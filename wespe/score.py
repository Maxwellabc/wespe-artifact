"""Independent CUGR2-unit scorer for an ISPD'18 .guide.

printStatistics in GlobalRouter.cpp scores the *routing tree*, not the guide
patches. This scorer therefore:

- Counts wirelength and usage from 1-D wire rectangles (the basic guides
  written for same-layer tree edges), *additively*: overlapping boxes inside
  one net are counted once per box, matching the tree preorder walk. 1-cell
  boxes and 2-D pin-access patches contribute no wire edges.
- Counts overflow as usage > 0 && usage > capacity with
  capacity = max(edge.capacity, 0), using capacity3D.txt (already max'ed).
- Wirelength is sum of getEdgeLength(direction, edge_index), then integer
  divided by M2 pitch, matching `wireLength / gridGraph.getM2Pitch()`.
  `direction` is the layer's preferred direction from the run dump
  (capacity3D.txt / LayerGrid.direction), not a hardcoded Metal2=H stack.
  ispd18 sample and test1 have Metal2 vertical; a vertical Metal2 box at
  x=nx-1 is a legal V-edge and must use edge_len_v[y], not edge_len_h[x].
- Via count: `vias_basic_incr` is `printStatistics`'s tree via total
  (`sum abs(node.layer-child.layer)` over via edges). getGuides writes each
  via as a min..max point-box stack in the basic-guide prefix (before the
  first 2-D pin-access patch). A strictly increasing-by-1 walk of those
  stacks matches the tree count except when a zero-length same-layer tree
  edge at the same (x,y) is written as a 1-cell box immediately after a
  downward via and is absorbed into that stack. The scorer detects that
  case (see `_zerolen_via_corrections`) and subtracts the absorbed layer.
  Validated exactly against route.log on sample (53), ispd18 test1 (42914),
  test5 (879492), ispd18 test8 (2189474), ispd18 test10 (2312318), and
  ispd19 test8 (7232757). On ispd19_test7 the first-2-D-patch cutoff is not
  enough: padding-0 pin-access is a 1-cell 0–2 point stack, identical to a
  tree via, and is counted whenever it appears before the first 2-D patch
  (or when a net has no 2-D patch). Passing a CUGR2 pin-access dump
  (`load_pin_access`, from `route -dump_pin_access` / `-dump_pins_only`)
  cuts the basic prefix at the true start of section (2). That cutoff is
  exact on all of the benches above and on ispd19_test7 (3774226), and it
  also reproduces the logged pin-access and wire-patch areas on test7.

Always validate on the original guide against route.log before publishing
a repaired official-unit score.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .grid_geom import GridGeom, box_kind, wire_edges_of_box
from .guide import iter_guide_nets, parse_body_boxes
from .io_run import Cugr2Run


@dataclass
class Score:
    n_nets: int
    n_empty: int
    n_boxes: int
    n_wire_boxes: int
    n_point_boxes: int
    n_patch2d_boxes: int
    n_unmapped_boxes: int
    wire_edges: int
    wl_dbu: int
    wl_metric: int
    vias_endpoint_span: int
    vias_all_point_span: int
    vias_wire_layer_span: int
    vias_basic_incr: int
    vias_zerolen_corrections: int
    overflow_edges: int
    overflow_units: float
    overflow_list: list[tuple[int, int, int, float, float]] = field(default_factory=list)
    usage_mismatch_cells: int = -1
    usage_mismatch_max_abs: float = -1.0
    notes: list[str] = field(default_factory=list)


def _via_span(layers: set[int]) -> int:
    if len(layers) <= 1:
        return 0
    return max(layers) - min(layers)


def _basic_incr_vias(cells_list: list) -> tuple[int, int]:
    """Official tree via count from the basic-guide prefix.

    getGuides writes each via edge as point boxes at (node.x, node.y) on
    every layer from min(parent,child) to max inclusive, in preorder, and
    stops that section at the first 2-D pin-access patch (padding=1 is 3x3
    in the interior; clipped at the die is still patch2d). Summing
    (max-min) over strictly increasing-by-1 same-(x,y) runs reproduces
    `viaCount += abs(layer diff)` except for one CUGR2 quirk:

    PatternRoute can emit a zero-length same-layer tree edge (parent and
    child at the same (x,y,layer)). getGuides writes that as a 1-cell box.
    If the upper node of a via column has such an edge, preorder writes it
    immediately after the parent's downward via stack, on the next layer,
    so a naive increasing walk absorbs it and overcounts by 1. Confirmed
    against per-net printStatistics dumps (ispd18_test5: 4 nets, +1 each,
    168 zero-len edges total; ispd18_test8: 8 nets, +1 each).

    Signature: two consecutive span>=1 via stacks S1,S2 at the same (x,y)
    with S2.max == S1.max, S2.min < S1.min, S1.min in S2, AND the upper
    node continues with either a third same-(x,y) via stack S3 containing
    S1.max, or a 1-D wire on layer S1.max that touches (x,y) immediately
    after S2 (before the next via stack). The wire alternative covers the
    case where the upper node leaves (x,y) on S1.max instead of via-ing
    in place (ispd18_test8 net479). Subtract 1 from S2. A same-xy S1,S2
    pair without S3 or an S1.max wire is a real via from S1.max down, and
    must not be corrected.

    Returns (via_count, n_corrections).
    """
    stacks: list[tuple[int, int, list[int], int, int, int]] = []  # x,y,layers,span,i,j
    i = 0
    n = len(cells_list)
    basic_end = n
    while i < n:
        L, x0, y0, x1, y1 = cells_list[i]
        k = box_kind(x0, y0, x1, y1)
        if k == "patch2d":
            basic_end = i
            break
        if k != "point":
            i += 1
            continue
        layers = [L]
        j = i + 1
        while (
            j < n
            and box_kind(cells_list[j][1], cells_list[j][2], cells_list[j][3], cells_list[j][4]) == "point"
            and cells_list[j][1] == x0
            and cells_list[j][2] == y0
            and cells_list[j][0] == layers[-1] + 1
        ):
            layers.append(cells_list[j][0])
            j += 1
        span = layers[-1] - layers[0]
        if span >= 1:
            stacks.append((x0, y0, layers, span, i, j))
        i = j
    corrections = _zerolen_via_corrections(stacks, cells_list, basic_end)
    total = sum(s[3] for s in stacks) - corrections
    return total, corrections


def _wire_touches(cells: list, start: int, end: int, layer: int, x: int, y: int) -> bool:
    for b in cells[start:end]:
        L, x0, y0, x1, y1 = b
        if L != layer or box_kind(x0, y0, x1, y1) != "wire":
            continue
        if y0 == y1 == y and x0 <= x <= x1:
            return True
        if x0 == x1 == x and y0 <= y <= y1:
            return True
    return False


def _zerolen_via_corrections(
    stacks: list[tuple[int, int, list[int], int, int, int]],
    cells: list,
    basic_end: int,
) -> int:
    """Count absorbed zero-length same-layer boxes in via stacks (see above)."""
    ncorr = 0
    for k in range(len(stacks) - 1):
        x1, y1, L1, _s1, _i1, j1 = stacks[k]
        x2, y2, L2, _s2, i2, j2 = stacks[k + 1]
        if (x1, y1) != (x2, y2):
            continue
        if not (L2[-1] == L1[-1] and L2[0] < L1[0] and L1[0] in L2):
            continue
        nxt = stacks[k + 2] if k + 2 < len(stacks) else None
        has_s3 = (
            nxt is not None
            and (nxt[0], nxt[1]) == (x1, y1)
            and L1[-1] in nxt[2]
        )
        nxt_start = nxt[4] if nxt is not None else basic_end
        has_top_wire = _wire_touches(cells, j2, nxt_start, L1[-1], x1, y1)
        if has_s3 or has_top_wire:
            ncorr += 1
    return ncorr




PIN_PATCH_TOP = 2  # min_routing_layer + 1 with min_routing_layer = 1


def load_pin_access(path: str | Path) -> dict[str, list[tuple[int, int, int]]]:
    """Load a `-dump_pin_access` file: net -> [(x, y, layer), ...] in getGuides order.

    Only access points with layer < min_routing_layer are written; each becomes
    one pin-access stack (layers layer..min_routing+1, padding 0 or 1).
    """
    nets: dict[str, list[tuple[int, int, int]]] = {}
    name: str | None = None
    pts: list[tuple[int, int, int]] = []
    with Path(path).open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("net "):
                if name is not None:
                    nets[name] = pts
                parts = line.split()
                name = parts[1]
                pts = []
            else:
                x, y, layer = map(int, line.split())
                pts.append((x, y, layer))
        if name is not None:
            nets[name] = pts
    return nets


def _resolve_pin_access(
    pin_access: dict[str, list[tuple[int, int, int]]] | str | Path | None,
    run: Cugr2Run,
) -> dict[str, list[tuple[int, int, int]]] | None:
    if pin_access is None:
        cand = run.path / "pin_access.txt"
        return load_pin_access(cand) if cand.is_file() else None
    if isinstance(pin_access, (str, Path)):
        return load_pin_access(pin_access)
    return pin_access


def _is_point_box(b: tuple) -> bool:
    return box_kind(b[1], b[2], b[3], b[4]) == "point"


def _matches_pin_access_item(
    cells: list, i: int, x: int, y: int, layer: int, nx: int, ny: int, pin_top: int
) -> bool:
    """boxes starting at i are the getGuides pin-access stack for (x,y,layer)."""
    nboxes = pin_top - layer + 1
    if i + nboxes - 1 >= len(cells):
        return False
    rect = cells[i][1:]
    for k, L in enumerate(range(layer, pin_top + 1)):
        b = cells[i + k]
        if b[0] != L or b[1:] != rect:
            return False
    if (x, y, x, y) == rect:
        return True
    r1 = (max(x - 1, 0), max(y - 1, 0), min(x + 1, nx - 1), min(y + 1, ny - 1))
    return rect == r1


def _pin_access_basic_end(
    cells: list,
    pts: list[tuple[int, int, int]],
    nx: int,
    ny: int,
    pin_top: int = PIN_PATCH_TOP,
) -> int | None:
    """Start index of pin-access boxes, or None if the suffix does not match.

    getGuides writes [basic tree][pin-access stacks in pin order][wire-patch
    points]. Wire patches are 1-cell points and may form incidental span>=1
    stacks, so the trailing point suffix is shrunk just enough for every
    dumped access point to match a padding-0 or clipped padding-1 stack in
    reverse order.
    """
    end = len(cells)

    def try_match(end: int) -> int | None:
        i = end
        for x, y, layer in reversed(pts):
            nboxes = pin_top - layer + 1
            start = i - nboxes
            if start < 0 or not _matches_pin_access_item(cells, start, x, y, layer, nx, ny, pin_top):
                return None
            i = start
        for b in cells[end:]:
            if not _is_point_box(b):
                return None
        return i

    while True:
        got = try_match(end)
        if got is not None:
            return got
        if end >= 1 and _is_point_box(cells[end - 1]):
            end -= 1
            continue
        return None


def score_guide(
    guide_path: str | Path,
    geom: GridGeom,
    run: Cugr2Run,
    compare_usage: bool = True,
    pin_access: dict[str, list[tuple[int, int, int]]] | str | Path | None = None,
) -> Score:
    """Score a guide in CUGR2 units. Does not load the whole file into one string.

    If `pin_access` is a dump path or net->access-point map (or `pin_access.txt`
    exists in the run directory), the basic-guide via walk stops at the true
    start of getGuides section (2). Otherwise the walk stops at the first
    2-D pin-access patch, which is exact on every checked bench except
    ispd19_test7.
    """
    usage: dict[tuple[int, int, int], int] = defaultdict(int)
    wl_dbu = 0
    n_nets = n_empty = n_boxes = 0
    n_wire = n_point = n_patch = n_bad = 0
    wire_edges = 0
    vias_endpoint = 0
    vias_point = 0
    vias_wire = 0
    vias_basic = 0
    vias_zerolen = 0
    pins = _resolve_pin_access(pin_access, run)
    n_pin_cut = 0
    n_pin_fallback = 0
    pin_top = run.min_routing_layer + 1

    for _name, body in iter_guide_nets(guide_path):
        n_nets += 1
        if not body:
            n_empty += 1
            continue
        raw_boxes = parse_body_boxes(body)
        n_boxes += len(raw_boxes)
        cells_list = []
        for b in raw_boxes:
            mapped = geom.box_to_cells(b[1], b[2], b[3], b[4])
            if mapped is None:
                n_bad += 1
                continue
            cells_list.append((b[0],) + mapped)
        from .guide import classify_cell_boxes

        groups = classify_cell_boxes(cells_list)
        n_wire += len(groups["wire"])
        n_point += len(groups["point"])
        n_patch += len(groups["patch2d"])

        net_edge_mult: dict[tuple[int, int, int], int] = defaultdict(int)
        wire_cells_by_xy: dict[tuple[int, int], set[int]] = defaultdict(set)
        endpoints: dict[tuple[int, int], set[int]] = defaultdict(set)
        for L, x0, y0, x1, y1 in groups["wire"]:
            for e in wire_edges_of_box(L, x0, y0, x1, y1):
                net_edge_mult[e] += 1
            if y0 == y1:
                for x in range(x0, x1 + 1):
                    wire_cells_by_xy[x, y0].add(L)
                endpoints[x0, y0].add(L)
                endpoints[x1, y1].add(L)
            elif x0 == x1:
                for y in range(y0, y1 + 1):
                    wire_cells_by_xy[x0, y].add(L)
                endpoints[x0, y0].add(L)
                endpoints[x1, y1].add(L)
        point_by_xy: dict[tuple[int, int], set[int]] = defaultdict(set)
        for L, x0, y0, x1, y1 in groups["point"]:
            point_by_xy[x0, y0].add(L)

        for e, mult in net_edge_mult.items():
            L, x, y = e
            usage[e] += mult
            # Official printStatistics uses getLayerDirection(layer), which
            # follows the LEF and is dumped as the first integer of each
            # capacity3D layer. A hardcoded Metal2=H map indexes edge_len_h[x]
            # for vertical Metal2 wires and raises IndexError at x=nx-1 on
            # benches whose Metal2 is vertical (ispd18 sample, test1).
            if L not in run.layers:
                raise ValueError(
                    f"wire edge {e} is on a layer with no capacity/direction dump"
                )
            d = run.layers[L].direction
            wl_dbu += geom.edge_length(d, x if d == 0 else y) * mult
            wire_edges += mult

        for xy, ls in endpoints.items():
            layers = set(ls) | point_by_xy.get(xy, set())
            vias_endpoint += _via_span(layers)
        for xy, ls in point_by_xy.items():
            vias_point += _via_span(ls)
        for xy, ls in wire_cells_by_xy.items():
            vias_wire += _via_span(ls)
        if pins is not None and _name in pins:
            cut = _pin_access_basic_end(cells_list, pins[_name], geom.nx, geom.ny, pin_top)
            if cut is None:
                vb, vc = _basic_incr_vias(cells_list)
                n_pin_fallback += 1
            else:
                vb, vc = _basic_incr_vias(cells_list[:cut])
                n_pin_cut += 1
        else:
            vb, vc = _basic_incr_vias(cells_list)
        vias_basic += vb
        vias_zerolen += vc

    of_edges = 0
    of_units = 0.0
    of_list: list[tuple[int, int, int, float, float]] = []
    for L, g in run.layers.items():
        for x, row in enumerate(g.cap):
            for y, c in enumerate(row):
                u = float(usage.get((L, x, y), 0))
                if u > 0 and u > c:
                    of_edges += 1
                    of_units += u - c
                    of_list.append((L, x, y, u, c))

    mismatch = -1
    mismatch_max = -1.0
    if compare_usage:
        mismatch = 0
        mismatch_max = 0.0
        for L, g in run.layers.items():
            for x, row in enumerate(g.usage):
                for y, u_dump in enumerate(row):
                    u = float(usage.get((L, x, y), 0))
                    if u != u_dump:
                        mismatch += 1
                        mismatch_max = max(mismatch_max, abs(u - u_dump))

    notes = [
        "WL/OF/usage come from 1-D wire rectangles counted additively per box (tree same-layer edges).",
        "vias_basic_incr matches printStatistics (tree via edges) after correcting "
        "zero-length same-layer boxes absorbed into a downward via stack. "
        f"zerolen_corrections={vias_zerolen}.",
        "capacity3D.txt already stores max(capacity, 0).",
    ]
    if pins is not None:
        notes.append(
            "via cutoff: pin-access dump "
            f"(nets_cut={n_pin_cut}, fallback_to_first_patch2d={n_pin_fallback})."
        )
    else:
        notes.append(
            "via cutoff: first 2-D pin-access patch (no pin-access dump; "
            "not exact on ispd19_test7)."
        )
    return Score(
        n_nets=n_nets,
        n_empty=n_empty,
        n_boxes=n_boxes,
        n_wire_boxes=n_wire,
        n_point_boxes=n_point,
        n_patch2d_boxes=n_patch,
        n_unmapped_boxes=n_bad,
        wire_edges=wire_edges,
        wl_dbu=wl_dbu,
        wl_metric=wl_dbu // geom.m2_pitch,
        vias_endpoint_span=vias_endpoint,
        vias_all_point_span=vias_point,
        vias_wire_layer_span=vias_wire,
        vias_basic_incr=vias_basic,
        vias_zerolen_corrections=vias_zerolen,
        overflow_edges=of_edges,
        overflow_units=of_units,
        overflow_list=of_list,
        usage_mismatch_cells=mismatch,
        usage_mismatch_max_abs=mismatch_max,
        notes=notes,
    )


def score_as_dict(s: Score) -> dict:
    return {
        "n_nets": s.n_nets,
        "n_empty": s.n_empty,
        "n_boxes": s.n_boxes,
        "n_wire_boxes": s.n_wire_boxes,
        "n_point_boxes": s.n_point_boxes,
        "n_patch2d_boxes": s.n_patch2d_boxes,
        "n_unmapped_boxes": s.n_unmapped_boxes,
        "wire_edges": s.wire_edges,
        "wl_dbu": s.wl_dbu,
        "wl_metric": s.wl_metric,
        "vias_endpoint_span": s.vias_endpoint_span,
        "vias_all_point_span": s.vias_all_point_span,
        "vias_wire_layer_span": s.vias_wire_layer_span,
        "vias_basic_incr": s.vias_basic_incr,
        "vias_zerolen_corrections": s.vias_zerolen_corrections,
        "overflow_edges": s.overflow_edges,
        "overflow_units": s.overflow_units,
        "overflow_list": [
            {"layer": L, "x": x, "y": y, "usage": u, "capacity": c}
            for L, x, y, u, c in s.overflow_list
        ],
        "usage_mismatch_cells": s.usage_mismatch_cells,
        "usage_mismatch_max_abs": s.usage_mismatch_max_abs,
        "notes": s.notes,
    }
