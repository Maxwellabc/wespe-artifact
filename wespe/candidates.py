"""Candidate Steiner attachments and 3D path detours inside a congestion window."""
from __future__ import annotations

import heapq
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .io_run import Cugr2Run, OverflowEdge

Edge = tuple[int, int, int]  # layer, x, y
Node = tuple[int, int, int]  # layer, x, y (grid point, not edge)


def nets_using_overflows(run: Cugr2Run, ofs: Iterable[OverflowEdge] | None = None) -> dict[str, list[Edge]]:
    want = {(e.layer, e.x, e.y) for e in (ofs if ofs is not None else run.overflows)}
    return {n: segs for n, segs in run.window_nets.items() if any(s in want for s in segs)}


def _neighbors(run: Cugr2Run, node: Node, bbox: tuple[int, int, int, int]) -> list[tuple[Node, Edge | None, str]]:
    """Return (next_node, used_wire_edge_or_None, kind) steps from a grid point.

    Wire steps consume the G-cell edge that starts at the lower endpoint.
    Via steps do not consume a wire edge (via cost is counted separately).
    """
    L, x, y = node
    x0, y0, x1, y1 = bbox
    out: list[tuple[Node, Edge | None, str]] = []
    layers = run.layers
    if L in layers:
        g = layers[L]
        if g.direction == 0:  # H: edge (L,x,y) connects (x,y)--(x+1,y)
            if 0 <= x < g.nx and 0 <= y < g.ny:
                out.append(((L, x + 1, y), (L, x, y), "wire"))
            if 0 <= x - 1 < g.nx and 0 <= y < g.ny:
                out.append(((L, x - 1, y), (L, x - 1, y), "wire"))
        else:  # V: edge (L,x,y) connects (x,y)--(x,y+1)
            if 0 <= x < g.nx and 0 <= y < g.ny:
                out.append(((L, x, y + 1), (L, x, y), "wire"))
            if 0 <= x < g.nx and 0 <= y - 1 < g.ny:
                out.append(((L, x, y - 1), (L, x, y - 1), "wire"))
    # vias
    for nL in (L - 1, L + 1):
        if nL in layers and x0 <= x <= x1 and y0 <= y <= y1:
            out.append(((nL, x, y), None, "via"))
    return out


def maze_paths(
    run: Cugr2Run,
    src: Node,
    dst: Node,
    bbox: tuple[int, int, int, int],
    ripped_bonus: dict[Edge, float],
    max_paths: int = 8,
    max_expansions: int = 20000,
) -> list[tuple[list[Edge], int, int]]:
    """Yen-like k-shortest paths on residual cost (overflow first, then vias, then length).

    Cost of entering a wire edge e: M_OF * max(0, 1 - residual_after_rip(e)) + 1
    (unit wire) plus a large penalty if residual_after_rip < 1, so we prefer
    residual-feasible detours. Via cost is 8 units of wire-equivalent (4 / 0.5).
    Returns list of (wire_edges, n_vias, wire_count).
    """
    # Dijkstra + simple deviation by forbidding first-used overflow / first-used edges
    def residual_of(e: Edge) -> float:
        L, x, y = e
        g = run.layers[L]
        if not (0 <= x < g.nx and 0 <= y < g.ny):
            return -1e9
        return g.cap[x][y] - g.usage[x][y] + ripped_bonus.get(e, 0.0)

    def edge_cost(e: Edge | None, kind: str) -> tuple[int, int, int]:
        if kind == "via":
            return (0, 1, 0)
        assert e is not None
        res = residual_of(e)
        of = 1 if res < 1 else 0  # one more unit of demand would overflow if res < 1
        return (of, 0, 1)

    def dijkstra(forbidden: set[Edge]) -> tuple[list[Edge], int, int] | None:
        # state cost is (overflow_units, vias, wl)
        start = (0, 0, 0, src)
        heap: list[tuple[int, int, int, Node]] = [start]
        prev: dict[Node, tuple[Node, Edge | None, str]] = {}
        best: dict[Node, tuple[int, int, int]] = {src: (0, 0, 0)}
        exp = 0
        while heap and exp < max_expansions:
            of, vias, wl, u = heapq.heappop(heap)
            if best.get(u, (1 << 30,)) < (of, vias, wl):
                continue
            exp += 1
            if u == dst:
                edges: list[Edge] = []
                nv = 0
                cur = u
                while cur != src:
                    p, e, kind = prev[cur]
                    if kind == "via":
                        nv += 1
                    elif e is not None:
                        edges.append(e)
                    cur = p
                edges.reverse()
                return (edges, nv, len(edges))
            for v, e, kind in _neighbors(run, u, bbox):
                if e is not None and e in forbidden:
                    continue
                # stay in bbox
                if not (bbox[0] <= v[1] <= bbox[2] and bbox[1] <= v[2] <= bbox[3]):
                    continue
                if v[0] not in run.layers:
                    continue
                dof, dv, dw = edge_cost(e, kind)
                nc = (of + dof, vias + dv, wl + dw)
                if nc < best.get(v, (1 << 30, 0, 0)):
                    best[v] = nc
                    prev[v] = (u, e, kind)
                    heapq.heappush(heap, (nc[0], nc[1], nc[2], v))
        return None

    found: list[tuple[list[Edge], int, int]] = []
    forbidden: set[Edge] = set()
    first = dijkstra(forbidden)
    if first is None:
        return []
    found.append(first)
    # diversify: forbid each edge of the first path in turn (esp. overflow edges)
    of_set = {(e.layer, e.x, e.y) for e in run.overflows}
    prefer = [e for e in first[0] if e in of_set] + list(first[0])
    seen = {tuple(first[0])}
    for e in prefer:
        alt = dijkstra({e})
        if alt and tuple(alt[0]) not in seen:
            seen.add(tuple(alt[0]))
            found.append(alt)
        if len(found) >= max_paths:
            break
    return found


def terminals_for_net(
    segs: list[Edge], ofs: list[OverflowEdge], pad: int = 8
) -> tuple[Node, Node, tuple[int, int, int, int]] | None:
    """Heuristic 2-point terminals: endpoints of the overflow-touching local chain."""
    if not ofs:
        return None
    ox = [e.x for e in ofs]
    oy = [e.y for e in ofs]
    x0, x1 = min(ox) - pad, max(ox) + pad
    y0, y1 = min(oy) - pad, max(oy) + pad
    ofset = {(e.layer, e.x, e.y) for e in ofs}
    local = [s for s in segs if x0 <= s[1] <= x1 and y0 <= s[2] <= y1]
    if not local:
        return None
    # Build adjacency of local wire edges as grid points
    # H edge (L,x,y) connects (L,x,y)-(L,x+1,y); V (dir 1) (L,x,y)-(L,x,y+1)
    # We do not know direction here; caller should pass run. Use both interpretations
    # only for grouping: an overflow edge must lie on the chain.
    if not any(s in ofset for s in local):
        return None
    # Use the geometrically extreme local points as terminals (min/max x+y, include layer)
    pts: list[Node] = []
    for L, x, y in local:
        pts.append((L, x, y))
        pts.append((L, x + 1, y))
        pts.append((L, x, y + 1))
    # restrict to bbox
    pts = [p for p in pts if x0 <= p[1] <= x1 + 1 and y0 <= p[2] <= y1 + 1]
    if len(pts) < 2:
        return None
    src = min(pts, key=lambda p: (p[1] + p[2], p[0], p[1], p[2]))
    dst = max(pts, key=lambda p: (p[1] + p[2], p[0], p[1], p[2]))
    if src == dst:
        return None
    return src, dst, (x0, y0, x1, y1)


# ---------------------------------------------------------------------------
# Pattern L/Z paths and Steiner-topology operators on a real 3-D G-cell grid
# ---------------------------------------------------------------------------

def _h_edges(layer: int, x0: int, x1: int, y: int) -> list[Edge]:
    lo, hi = (x0, x1) if x0 <= x1 else (x1, x0)
    return [(layer, x, y) for x in range(lo, hi)]


def _v_edges(layer: int, x: int, y0: int, y1: int) -> list[Edge]:
    lo, hi = (y0, y1) if y0 <= y1 else (y1, y0)
    return [(layer, x, y) for y in range(lo, hi)]


def _layer_for_direction(run: Cugr2Run, direction: int, allowed: set[int]) -> int | None:
    """Preferred layer in `allowed` that runs in `direction` (0=H, 1=V)."""
    # Prefer the lowest index (Metal2 for H, Metal3 for V on a 5-layer stack).
    cands = [L for L in sorted(allowed) if L in run.layers and run.layers[L].direction == direction]
    return cands[0] if cands else None


def _vias_between(a: Node, b: Node) -> int:
    if a[1] != b[1] or a[2] != b[2]:
        raise ValueError("via must be vertical")
    return abs(a[0] - b[0])


def _in_bbox(x: int, y: int, bbox: tuple[int, int, int, int]) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def pattern_lz_paths(
    run: Cugr2Run,
    src: Node,
    dst: Node,
    bbox: tuple[int, int, int, int],
    allowed_layers: set[int] | None = None,
) -> list[tuple[list[Edge], int, int, str]]:
    """Monotone L and Z pattern embeddings of a 2-pin tree edge.

    Uses one preferred-direction H layer and one V layer from `allowed_layers`
    (default: every layer present in the run that lies between src and dst
    inclusive, plus those two preferred directions). Does *not* search
    unrestricted maze, and does not introduce a third routing layer (e.g. a
    Metal4 lift when the incumbent used only Metal2/Metal3).

    L-shapes: HV and VH (via at the single corner). Z-shapes: HZH and VZV
    with the intermediate coordinate ranging over the bbox Hanan line
    (exclusive of the terminals' own coordinate). Returns
    (wire_edges, n_vias, wire_count, label).
    """
    if allowed_layers is None:
        lo, hi = min(src[0], dst[0]), max(src[0], dst[0])
        allowed_layers = {L for L in run.layers if lo <= L <= hi}
        # also keep the preferred-direction layers among those
    allowed = {L for L in allowed_layers if L in run.layers}
    hL = _layer_for_direction(run, 0, allowed)
    vL = _layer_for_direction(run, 1, allowed)
    if hL is None or vL is None:
        return []

    sx, sy = src[1], src[2]
    dx, dy = dst[1], dst[2]
    out: list[tuple[list[Edge], int, int, str]] = []
    seen: set[tuple[tuple[Edge, ...], int]] = set()

    def emit(edges: list[Edge], vias: int, label: str) -> None:
        # drop empty / out-of-bbox
        if any(not _in_bbox(e[1], e[2], bbox) and not _in_bbox(e[1] + (1 if run.layers[e[0]].direction == 0 else 0),
                                                                e[2] + (1 if run.layers[e[0]].direction == 1 else 0),
                                                                bbox) for e in edges):
            # allow edges whose start is in bbox; skip if completely outside
            pass
        key = (tuple(sorted(set(edges))), vias)
        if key in seen:
            return
        seen.add(key)
        out.append((edges, vias, len(edges), label))

    def stack_vias(x: int, y: int, l0: int, l1: int) -> int:
        return abs(l1 - l0)

    # L: HV  src --H--> (dx,sy) --V--> dst, vias at src, corner, dst
    def path_hv(mid_x: int, mid_y: int, label: str) -> None:
        # src -> (via to hL) -> H to (mid_x, sy) -> via to vL -> V to (mid_x, mid_y)
        # then, if mid is not dst, H or nothing; we use mid as the single corner for L
        edges: list[Edge] = []
        edges.extend(_h_edges(hL, sx, mid_x, sy))
        edges.extend(_v_edges(vL, mid_x, sy, mid_y))
        if mid_x != dx:
            # second H for Z (HZH): mid is (mid_x, mid_y) then H to (dx, mid_y)
            edges.extend(_h_edges(hL, mid_x, dx, mid_y))
        vias = stack_vias(sx, sy, src[0], hL)
        # corner vias: after first H we are on hL; to take V we via to vL
        if hL != vL:
            vias += 1  # at (mid_x, sy)
            if mid_x != dx:
                vias += 1  # back to hL at (mid_x, mid_y)
        vias += stack_vias(dx, mid_y if mid_x != dx else (mid_y), (hL if mid_x != dx else vL), dst[0])
        emit(edges, vias, label)

    def path_vh(mid_y: int, mid_x: int, label: str) -> None:
        edges: list[Edge] = []
        edges.extend(_v_edges(vL, sx, sy, mid_y))
        edges.extend(_h_edges(hL, sx, mid_x, mid_y))
        if mid_y != dy:
            edges.extend(_v_edges(vL, mid_x, mid_y, dy))
        vias = stack_vias(sx, sy, src[0], vL)
        if hL != vL:
            vias += 1
            if mid_y != dy:
                vias += 1
        vias += stack_vias(mid_x, dy if mid_y != dy else mid_y, (vL if mid_y != dy else hL), dst[0])
        emit(edges, vias, label)

    # L-shapes (corner at dst x or dst y)
    path_hv(dx, dy, "pattern_L_HV")
    path_vh(dy, dx, "pattern_L_VH")
    # Z-shapes: intermediate x in bbox, exclusive of sx,dx
    for mx in range(bbox[0], bbox[2] + 1):
        if mx == sx or mx == dx:
            continue
        if not _in_bbox(mx, sy, bbox) or not _in_bbox(mx, dy, bbox):
            continue
        path_hv(mx, dy, f"pattern_Z_HZH_{mx}")
    for my in range(bbox[1], bbox[3] + 1):
        if my == sy or my == dy:
            continue
        if not _in_bbox(sx, my, bbox) or not _in_bbox(dx, my, bbox):
            continue
        path_vh(my, dx, f"pattern_Z_VZV_{my}")
    return out


def hanan_grid(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Hanan grid of 2-D points (x, y): all (x_i, y_j) combinations."""
    xs = sorted({p[0] for p in points})
    ys = sorted({p[1] for p in points})
    return [(x, y) for x in xs for y in ys]


def same_side_topologies(
    pins: list[tuple[int, int]], cut_x: int
) -> list[list[tuple[tuple[int, int], tuple[int, int]]]]:
    """Same-side Steiner topologies: chain each side of cut_x, then one bridge.

    Returns a list of topologies, each a list of 2-pin (u, v) segments in
    the plane (no layer). Lift of `alns_refiner.operators.same_side_trees`
    onto real pin coordinates from a guide / tree.txt window.
    """
    left = [p for p in pins if p[0] < cut_x]
    right = [p for p in pins if p[0] >= cut_x]
    if not left or not right:
        # degenerate: chain all pins
        if len(pins) < 2:
            return []
        return [[(pins[i], pins[i + 1]) for i in range(len(pins) - 1)]]
    out = []
    for a in left:
        for b in right:
            segs = []
            for i in range(len(left) - 1):
                segs.append((left[i], left[i + 1]))
            for i in range(len(right) - 1):
                segs.append((right[i], right[i + 1]))
            segs.append((a, b))
            out.append(segs)
    return out


def flute_style_topologies(
    pins: list[tuple[int, int]]
) -> list[tuple[str, list[tuple[tuple[int, int], tuple[int, int]]]]]:
    """A small set of RSMT-style topologies on the pin set, not a FLUTE call.

    Variants: pin-chain in x-order, pin-chain in y-order, star through the
    Hanan median, and same-side flips about the median x. These can be
    selected as Steiner candidates; they are *not* the FLUTE library output.
    """
    if len(pins) < 2:
        return []
    variants: list[tuple[str, list[tuple[tuple[int, int], tuple[int, int]]]]] = []
    byx = sorted(pins, key=lambda p: (p[0], p[1]))
    variants.append(("chain_x", [(byx[i], byx[i + 1]) for i in range(len(byx) - 1)]))
    byy = sorted(pins, key=lambda p: (p[1], p[0]))
    variants.append(("chain_y", [(byy[i], byy[i + 1]) for i in range(len(byy) - 1)]))
    # star at Hanan median
    xs = sorted(p[0] for p in pins)
    ys = sorted(p[1] for p in pins)
    mid = (xs[len(xs) // 2], ys[len(ys) // 2])
    if mid not in pins or len(pins) > 2:
        variants.append(("star_hanan_median", [(mid, p) for p in pins if p != mid] or [(pins[0], pins[1])]))
    cut = xs[len(xs) // 2]
    for i, segs in enumerate(same_side_topologies(pins, cut)):
        variants.append((f"same_side_x{cut}_{i}", segs))
    return variants


def hanan_slide_steiner(
    pins: list[tuple[int, int]],
    steiner: list[tuple[int, int]],
) -> list[list[tuple[int, int]]]:
    """Move each existing Steiner point to other Hanan-grid locations of the pins.

    Returns a list of alternative Steiner-point sets (the pins are not moved).
    Empty if there are no Steiner points (2-pin nets).
    """
    if not steiner:
        return []
    h = [p for p in hanan_grid(pins) if p not in pins]
    if not h:
        return []
    # For one Steiner point, try every Hanan location; for several, try moving
    # one point at a time (linear, not a Cartesian product).
    alts: list[list[tuple[int, int]]] = []
    for i, s in enumerate(steiner):
        for q in h:
            if q == s:
                continue
            nxt = list(steiner)
            nxt[i] = q
            alts.append(nxt)
    return alts


# ---------------------------------------------------------------------------
# Lift Steiner topologies onto legal 3-D layers; optional real Flute LUT
# ---------------------------------------------------------------------------

_FLUTE_LIB = None
_FLUTE_STATUS: str | None = None  # "ok", "unavailable: ...", or None (not yet attempted)
_DEFAULT_LUT_DIRS = (
    Path(__file__).resolve().parents[2] / "code/CUGR2-wadmes/flute",
    Path(__file__).resolve().parents[1] / "CUGR2-wadmes/flute",
    Path(__file__).resolve().parent / "native",
)


def _candidates_root() -> Path:
    # code/wespe/candidates.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def flute_library_status() -> str:
    """Whether the real Flute shared library + LUT is callable ('ok' or a reason)."""
    _load_flute_lib()
    return _FLUTE_STATUS or "unavailable: not attempted"


def _load_flute_lib():
    global _FLUTE_LIB, _FLUTE_STATUS
    if _FLUTE_STATUS is not None:
        return _FLUTE_LIB
    import ctypes
    import os

    so = Path(__file__).resolve().parent / "native" / "libflute_wespe.so"
    if not so.exists():
        _FLUTE_STATUS = f"unavailable: missing {so}"
        _FLUTE_LIB = False
        return None
    lut = os.environ.get("WESPE_FLUTE_LUT", "")
    lut_dir = Path(lut) if lut else None
    if lut_dir is None or not (lut_dir / "POWV9.dat").exists():
        for d in _DEFAULT_LUT_DIRS:
            if (d / "POWV9.dat").exists() and (d / "POST9.dat").exists():
                lut_dir = d
                break
    if lut_dir is None or not (lut_dir / "POWV9.dat").exists():
        _FLUTE_STATUS = "unavailable: POWV9.dat / POST9.dat not found"
        _FLUTE_LIB = False
        return None
    try:
        lib = ctypes.CDLL(str(so))
        lib.flute_init.argtypes = [ctypes.c_char_p]
        lib.flute_init.restype = ctypes.c_int
        lib.flute_rsmt.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
        ]
        lib.flute_rsmt.restype = ctypes.c_int
        rc = lib.flute_init(str(lut_dir).encode())
        if rc != 0:
            _FLUTE_STATUS = f"unavailable: flute_init({lut_dir}) returned {rc}"
            _FLUTE_LIB = False
            return None
        _FLUTE_LIB = lib
        _FLUTE_STATUS = f"ok (LUT {lut_dir})"
        return lib
    except OSError as exc:
        _FLUTE_STATUS = f"unavailable: {exc}"
        _FLUTE_LIB = False
        return None


def flute_rsmt_segments(pins: list[tuple[int, int]]) -> tuple[str, list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    """Call the real Flute LUT library. Returns (label, 2-pin segments) or None.

    Not the flute_style_* approximation. High-degree flutes_HD is not linked;
    only d in the LUT / flutes_MD range is supported. Duplicate (x, y) pins
    are collapsed first. Zero-length branches (pin coincident with a Steiner
    point) are dropped.
    """
    import ctypes

    uniq: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for p in pins:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    if len(uniq) < 2:
        return None
    lib = _load_flute_lib()
    if not lib:
        return None
    d = len(uniq)
    if d > 150:
        return None
    xs = (ctypes.c_int * d)(*[p[0] for p in uniq])
    ys = (ctypes.c_int * d)(*[p[1] for p in uniq])
    out_len = ctypes.c_int()
    out_nb = ctypes.c_int()
    cap = max(2 * d, 2)
    ox = (ctypes.c_int * cap)()
    oy = (ctypes.c_int * cap)()
    on = (ctypes.c_int * cap)()
    rc = lib.flute_rsmt(d, xs, ys, ctypes.byref(out_len), ctypes.byref(out_nb), ox, oy, on)
    if rc != 0 or out_nb.value <= 0:
        return None
    segs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    seen_e: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for i in range(out_nb.value):
        j = on[i]
        if j < 0 or j >= out_nb.value or j == i:
            continue
        a = (int(ox[i]), int(oy[i]))
        b = (int(ox[j]), int(oy[j]))
        if a == b:
            continue
        key = (a, b) if a <= b else (b, a)
        if key in seen_e:
            continue
        seen_e.add(key)
        segs.append(key)
    if not segs:
        return None
    return (f"flute_rsmt_d{len(uniq)}_len{out_len.value}", segs)


def _endpoints_of(run: Cugr2Run, e: Edge) -> tuple[Node, Node]:
    L, x, y = e
    if run.layers[L].direction == 0:
        return (L, x, y), (L, x + 1, y)
    return (L, x, y), (L, x, y + 1)


def via_aware_local_graph(run: Cugr2Run, local: list[Edge]) -> dict[Node, set[Node]]:
    """Adjacency of a local wire chain, with via columns at shared (x, y)."""
    adj: dict[Node, set[Node]] = defaultdict(set)
    layers_at: dict[tuple[int, int], set[int]] = defaultdict(set)
    for e in local:
        a, b = _endpoints_of(run, e)
        adj[a].add(b)
        adj[b].add(a)
        layers_at[a[1], a[2]].add(a[0])
        layers_at[b[1], b[2]].add(b[0])
    for (x, y), ls in layers_at.items():
        sl = sorted(ls)
        for i in range(len(sl) - 1):
            a, b = (sl[i], x, y), (sl[i + 1], x, y)
            adj[a].add(b)
            adj[b].add(a)
    return adj


def classify_local_points(adj: dict[Node, set[Node]]) -> tuple[list[Node], list[Node]]:
    """Degree-1 attachments (terminals) and degree >= 3 Steiner points."""
    terminals = [p for p, nbrs in adj.items() if len(nbrs) == 1]
    steiners = [p for p, nbrs in adj.items() if len(nbrs) >= 3]
    terminals.sort()
    steiners.sort()
    return terminals, steiners


def contracted_tree_segments(adj: dict[Node, set[Node]]) -> list[tuple[Node, Node]]:
    """Tree edges on terminals + Steiner points after contracting degree-2 nodes."""
    keys = [p for p, nbrs in adj.items() if len(nbrs) != 2]
    keyset = set(keys)
    segs: list[tuple[Node, Node]] = []
    seen: set[tuple[Node, Node]] = set()
    for start in keys:
        for first in adj[start]:
            prev, cur = start, first
            hops = 0
            while cur not in keyset and hops < 10000:
                nbrs = [n for n in adj[cur] if n != prev]
                if len(nbrs) != 1:
                    break
                prev, cur = cur, nbrs[0]
                hops += 1
            if cur in keyset and cur != start:
                key = (start, cur) if start <= cur else (cur, start)
                if key not in seen:
                    seen.add(key)
                    segs.append(key)
    return segs


def ensure_hv_layers(run: Cugr2Run, allowed: set[int]) -> set[int]:
    """Add the preferred missing H or V layer so L/Z embeddings are legal."""
    allowed = {L for L in allowed if L in run.layers}
    for direction in (0, 1):
        if _layer_for_direction(run, direction, allowed) is None:
            L = _layer_for_direction(run, direction, set(run.layers))
            if L is not None:
                allowed.add(L)
    return allowed


def expand_bbox_to_points(
    bbox: tuple[int, int, int, int], points: Iterable[tuple[int, int] | Node]
) -> tuple[int, int, int, int]:
    xs, ys = [], []
    for p in points:
        if len(p) == 3:
            xs.append(p[1])
            ys.append(p[2])
        else:
            xs.append(p[0])
            ys.append(p[1])
    if not xs:
        return bbox
    return (min(bbox[0], *xs), min(bbox[1], *ys), max(bbox[2], *xs), max(bbox[3], *ys))


def _node_at_xy(
    xy: tuple[int, int],
    terminals_by_xy: dict[tuple[int, int], list[Node]],
    default_layer: int,
) -> Node:
    if xy in terminals_by_xy:
        return min(terminals_by_xy[xy], key=lambda n: (abs(n[0] - default_layer), n[0]))
    return (default_layer, xy[0], xy[1])


def embed_topology(
    run: Cugr2Run,
    terminals: list[Node],
    segments: list[tuple[tuple[int, int], tuple[int, int]]],
    bbox: tuple[int, int, int, int],
    allowed_layers: set[int],
    label_prefix: str,
    max_product: int = 64,
) -> list[tuple[list[Edge], int, int, str]]:
    """Lift a 2-D Steiner topology onto legal layers with pattern L/Z per tree edge."""
    if not segments:
        return []
    allowed = ensure_hv_layers(run, set(allowed_layers))
    hL = _layer_for_direction(run, 0, allowed)
    default_L = hL if hL is not None else (min(allowed) if allowed else 1)
    by_xy: dict[tuple[int, int], list[Node]] = defaultdict(list)
    for t in terminals:
        by_xy[t[1], t[2]].append(t)
    bbox = expand_bbox_to_points(bbox, [p for seg in segments for p in seg] + [(t[1], t[2]) for t in terminals])

    per_seg: list[list[tuple[list[Edge], int, int, str]]] = []
    for u, v in segments:
        su = _node_at_xy(u, by_xy, default_L)
        sv = _node_at_xy(v, by_xy, default_L)
        if su == sv:
            continue
        opts = pattern_lz_paths(run, su, sv, bbox, allowed_layers=allowed)
        # Prefer L-shapes first so a bounded product stays meaningful
        opts.sort(key=lambda o: (0 if o[3].startswith("pattern_L") else 1, o[2], o[1], o[3]))
        if not opts:
            return []
        per_seg.append(opts)
    if not per_seg:
        return []

    from itertools import product as iproduct

    def n_L(opts: list) -> int:
        return sum(1 for o in opts if o[3].startswith("pattern_L")) or 1

    product_L = 1
    for opts in per_seg:
        product_L *= n_L(opts)
        if product_L > max_product:
            break

    combos: list[tuple[tuple[list[Edge], int, int, str], ...]] = []
    if product_L <= max_product:
        trimmed = [opts[: n_L(opts)] for opts in per_seg]
        combos.extend(iproduct(*trimmed))
        # single-segment Z deviations around the first all-L embedding
        base_L = tuple(opts[0] for opts in trimmed)
        for i, opts in enumerate(per_seg):
            for alt in opts[n_L(opts) :]:
                row = list(base_L)
                row[i] = alt
                combos.append(tuple(row))
                if len(combos) >= max_product:
                    break
            if len(combos) >= max_product:
                break
    else:
        # too many tree edges: one all-preferred-L embedding, plus one deviation per option
        base = tuple(opts[0] for opts in per_seg)
        combos.append(base)
        for i, opts in enumerate(per_seg):
            for alt in opts[1:]:
                row = list(base)
                row[i] = alt
                combos.append(tuple(row))
                if len(combos) >= max_product:
                    break
            if len(combos) >= max_product:
                break

    out: list[tuple[list[Edge], int, int, str]] = []
    seen: set[tuple[tuple[Edge, ...], int]] = set()
    for combo in combos:
        edges: list[Edge] = []
        vias = 0
        labels = []
        for edges_i, nv, _wl, lab in combo:
            edges.extend(edges_i)
            vias += nv
            labels.append(lab)
        uniq_edges = list(dict.fromkeys(edges))
        key = (tuple(sorted(set(uniq_edges))), vias)
        if key in seen:
            continue
        seen.add(key)
        tag = "+".join(labels)
        out.append((uniq_edges, vias, len(uniq_edges), f"{label_prefix}|{tag}"))
    return out


def steiner_topology_candidates(
    run: Cugr2Run,
    terminals: list[Node],
    steiners: list[Node],
    adj: dict[Node, set[Node]],
    bbox: tuple[int, int, int, int],
    allowed_layers: set[int],
    max_embs_per_topo: int = 64,
) -> tuple[list[tuple[list[Edge], int, int, str]], dict[str, int]]:
    """Hanan / flute_style / same-side / real-Flute embeddings on local terminals.

    Pins are the 2-D projections of via-aware degree-1 terminals. Existing
    degree >= 3 points are only used by hanan_slide (they are not forced
    into a new RSMT, which would freeze the incumbent Steiner locations).
    """
    counts = {
        "flute_style_topologies": 0,
        "flute_style_embeddings": 0,
        "flute_rsmt_topologies": 0,
        "flute_rsmt_embeddings": 0,
        "hanan_slide_topologies": 0,
        "hanan_slide_embeddings": 0,
        "n_terminals_3d": len(terminals),
        "n_steiners_3d": len(steiners),
    }
    pins = list(dict.fromkeys((t[1], t[2]) for t in terminals))
    if len(pins) < 2:
        return [], counts
    out: list[tuple[list[Edge], int, int, str]] = []

    for name, segs in flute_style_topologies(pins):
        counts["flute_style_topologies"] += 1
        embs = embed_topology(run, terminals, segs, bbox, allowed_layers, f"flute_style_{name}", max_embs_per_topo)
        counts["flute_style_embeddings"] += len(embs)
        out.extend(embs)

    real = flute_rsmt_segments(pins)
    if real is not None:
        counts["flute_rsmt_topologies"] += 1
        embs = embed_topology(run, terminals, real[1], bbox, allowed_layers, real[0], max_embs_per_topo)
        counts["flute_rsmt_embeddings"] += len(embs)
        out.extend(embs)

    inc_segs = contracted_tree_segments(adj) if steiners else []
    if inc_segs and steiners:
        pin_set = set(pins)
        steiner_xy = list(dict.fromkeys((s[1], s[2]) for s in steiners))
        # incidence as 2-D segments among pins + incumbent Steiners
        inc_2d: list[tuple[tuple[int, int], tuple[int, int]]] = []
        seen_s: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        for a, b in inc_segs:
            u, v = (a[1], a[2]), (b[1], b[2])
            if u == v:
                continue
            key = (u, v) if u <= v else (v, u)
            if key not in seen_s:
                seen_s.add(key)
                inc_2d.append(key)
        alts = hanan_slide_steiner(pins, steiner_xy)
        # cap slides
        if len(alts) > 16:
            alts = alts[:16]
        for i, new_s in enumerate(alts):
            remap = {old: new for old, new in zip(steiner_xy, new_s)}
            moved: list[tuple[tuple[int, int], tuple[int, int]]] = []
            seen_m: set[tuple[tuple[int, int], tuple[int, int]]] = set()
            for u, v in inc_2d:
                uu = remap.get(u, u)
                vv = remap.get(v, v)
                if uu == vv:
                    continue
                key = (uu, vv) if uu <= vv else (vv, uu)
                if key in seen_m:
                    continue
                seen_m.add(key)
                moved.append(key)
            if not moved:
                continue
            counts["hanan_slide_topologies"] += 1
            embs = embed_topology(
                run, terminals, moved, bbox, allowed_layers, f"hanan_slide_{i}", max_embs_per_topo
            )
            counts["hanan_slide_embeddings"] += len(embs)
            out.extend(embs)

    counts["embeddings_total"] = len(out)
    counts["flute_library"] = flute_library_status()
    return out, counts
