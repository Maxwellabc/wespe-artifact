"""Write a legal CUGR2 .guide for a WESPE window repair and re-score it.

Splices only the ripped local chains of the repaired nets into a copy of the
original guide. Maze terminals must stay connected to pins and non-ripped wires.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

from .grid_geom import (
    LAYER_DIRECTION,
    LAYER_NAME,
    GridGeom,
    box_kind,
    load_grid_geom,
    merge_collinear_edges,
    wire_edges_of_box,
)
from .guide import (
    CellBox,
    GuideBox,
    boxes_to_cells,
    cells_to_boxes,
    classify_cell_boxes,
    extract_named_nets,
    parse_body_boxes,
    write_guide_replacing,
)
from .io_run import Cugr2Run, load_run
from .repair import repair_run
from .score import score_as_dict, score_guide


def _endpoints(layer: int, x: int, y: int) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if LAYER_DIRECTION[layer] == 0:
        return (layer, x, y), (layer, x + 1, y)
    return (layer, x, y), (layer, x, y + 1)


def _via_boxes_for_path(
    edges: list[tuple[int, int, int]], src: tuple[int, int, int], dst: tuple[int, int, int]
) -> list[CellBox]:
    """Point boxes for layer changes along a maze path, including src/dst stacks."""
    # Build adjacency of path grid-points and infer via columns at each (x,y).
    layers_at: dict[tuple[int, int], set[int]] = defaultdict(set)
    layers_at[src[1], src[2]].add(src[0])
    layers_at[dst[1], dst[2]].add(dst[0])
    for e in edges:
        a, b = _endpoints(*e)
        layers_at[a[1], a[2]].add(a[0])
        layers_at[b[1], b[2]].add(b[0])
    boxes: list[CellBox] = []
    for (x, y), ls in layers_at.items():
        if max(ls) == min(ls):
            continue
        for L in range(min(ls), max(ls) + 1):
            boxes.append((L, x, y, x, y))
    return boxes


def _box_cells(b: CellBox) -> list[tuple[int, int, int]]:
    L, x0, y0, x1, y1 = b
    return [(L, x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


def _graph_from_boxes(boxes: list[CellBox]) -> dict[tuple[int, int, int], set[tuple[int, int, int]]]:
    adj: dict[tuple[int, int, int], set[tuple[int, int, int]]] = defaultdict(set)
    present: set[tuple[int, int, int]] = set()
    for b in boxes:
        cells = _box_cells(b)
        present.update(cells)
        L, x0, y0, x1, y1 = b
        # 4-adjacent in-plane
        for x in range(x0, x1 + 1):
            for y in range(y0, y1 + 1):
                if x + 1 <= x1:
                    adj[L, x, y].add((L, x + 1, y))
                    adj[L, x + 1, y].add((L, x, y))
                if y + 1 <= y1:
                    adj[L, x, y].add((L, x, y + 1))
                    adj[L, x, y + 1].add((L, x, y))
    # vias between adjacent layers at the same (x,y)
    by_xy: dict[tuple[int, int], set[int]] = defaultdict(set)
    for L, x, y in present:
        by_xy[x, y].add(L)
    for (x, y), ls in by_xy.items():
        for L in ls:
            if L + 1 in ls:
                adj[L, x, y].add((L + 1, x, y))
                adj[L + 1, x, y].add((L, x, y))
    # ensure isolated nodes exist
    for n in present:
        adj.setdefault(n, set())
    return adj


def _component(adj, start) -> set:
    if start not in adj:
        return set()
    seen = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def _split_wire_box(b, ripped: set) -> list:
    """Keep a 1-D wire box, or the leftover segments after removing ripped edges."""
    L, x0, y0, x1, y1 = b
    edges = wire_edges_of_box(L, x0, y0, x1, y1)
    if not any(e in ripped for e in edges):
        return [b]
    if all(e in ripped for e in edges):
        return []
    keep = [e for e in edges if e not in ripped]
    # merge only the leftovers of THIS box (do not collapse overlaps with others)
    return [(L, a, b0, c, d) for (L, a, b0, c, d) in merge_collinear_edges(keep)]


def splice_net(
    geom: GridGeom,
    raw_body: list[str],
    incumbent_edges: list[tuple[int, int, int]],
    new_edges: list[tuple[int, int, int]],
    src: tuple[int, int, int],
    dst: tuple[int, int, int],
    extra_terminals: list[tuple[int, int, int]] | None = None,
) -> tuple[list[GuideBox], dict]:
    """Replace the ripped local-chain wire rectangles; keep patches and other wires.

    Output box order matches getGuides: basic wires+vias, then 3x3 pin patches,
    then remaining 1-cell patches. This keeps vias_basic_incr well-defined.
    Kept (non-ripped) wire boxes are preserved as-is so intra-net overlaps
    outside the window are not collapsed.
    """
    raw_boxes = parse_body_boxes(raw_body)
    cells = boxes_to_cells(geom, raw_boxes)
    ripped = set(map(tuple, incumbent_edges))
    new = list(dict.fromkeys(map(tuple, new_edges)))  # unique, stable

    # split original boxes at the first 3x3 (start of pin-access patches)
    first_patch = next(
        (i for i, b in enumerate(cells) if box_kind(b[1], b[2], b[3], b[4]) == "patch2d"),
        len(cells),
    )
    basic, patch_tail = cells[:first_patch], cells[first_patch:]

    kept_wires: list[CellBox] = []
    leftover_splits: list[CellBox] = []
    kept_basic_points: list[CellBox] = []
    removed_boxes = 0
    split_boxes = 0
    for b in basic:
        k = box_kind(b[1], b[2], b[3], b[4])
        if k == "wire":
            parts = _split_wire_box(b, ripped)
            if parts == [b]:
                kept_wires.append(b)
            elif not parts:
                removed_boxes += 1
            else:
                removed_boxes += 1
                split_boxes += 1
                leftover_splits.extend(parts)
        elif k == "point":
            kept_basic_points.append(b)
        else:
            # unexpected 2-D in the basic prefix; keep it
            kept_wires.append(b)

    # also split any 1-D wire that (unexpectedly) sits in the patch tail
    patch2d: list[CellBox] = []
    patch_points: list[CellBox] = []
    for b in patch_tail:
        k = box_kind(b[1], b[2], b[3], b[4])
        if k == "wire":
            parts = _split_wire_box(b, ripped)
            if parts == [b]:
                kept_wires.append(b)
            elif not parts:
                removed_boxes += 1
            else:
                split_boxes += 1
                leftover_splits.extend(parts)
        elif k == "patch2d":
            patch2d.append(b)
        else:
            patch_points.append(b)

    new_wire_boxes: list[CellBox] = [
        (L, x0, y0, x1, y1) for L, x0, y0, x1, y1 in merge_collinear_edges(new)
    ]
    via_new = _via_boxes_for_path(new, src, dst)

    new_cells: set[tuple[int, int, int]] = set()
    for e in new:
        a, bpt = _endpoints(*e)
        new_cells.add(a)
        new_cells.add(bpt)
    ripped_xy: set[tuple[int, int]] = set()
    for e in ripped:
        a, bpt = _endpoints(*e)
        ripped_xy.add((a[1], a[2]))
        ripped_xy.add((bpt[1], bpt[2]))
    kept_xy_layers: dict[tuple[int, int], set[int]] = defaultdict(set)
    for b in kept_wires + leftover_splits + new_wire_boxes:
        for L, x, y in _box_cells(b):
            kept_xy_layers[x, y].add(L)
    pin_xy: set[tuple[int, int]] = set()
    for b in patch2d:
        for c in _box_cells(b):
            pin_xy.add((c[1], c[2]))
    for b in kept_basic_points + patch_points:
        if b[0] == 0:
            pin_xy.add((b[1], b[2]))

    def keep_point(b: CellBox) -> bool:
        L, x, y = b[0], b[1], b[2]
        on_new = (L, x, y) in new_cells
        on_kept = L in kept_xy_layers.get((x, y), set())
        is_pin = (x, y) in pin_xy or L == 0
        if (x, y) in ripped_xy and not on_new and not on_kept and not is_pin:
            return False
        return True

    dropped_points = 0
    kept_bp, kept_pp = [], []
    for b in kept_basic_points:
        if keep_point(b):
            kept_bp.append(b)
        else:
            dropped_points += 1
    for b in patch_points:
        if keep_point(b):
            kept_pp.append(b)
        else:
            dropped_points += 1

    have = set(kept_wires + leftover_splits + new_wire_boxes + kept_bp)
    added_vias = [b for b in via_new if b not in have]

    # CUGR2 order: basic wires and vias, then 3x3 pin patches, then other points
    basic_out = kept_wires + leftover_splits + new_wire_boxes + kept_bp + added_vias
    final_cells: list[CellBox] = basic_out + patch2d + kept_pp
    guide_boxes = cells_to_boxes(geom, final_cells)

    adj = _graph_from_boxes(final_cells)
    pin_cells = []
    for b in patch2d:
        pin_cells.extend(_box_cells(b))
    for b in kept_bp + kept_pp:
        if b[0] == 0:
            pin_cells.append((b[0], b[1], b[2]))
    must = set(pin_cells)
    for b in kept_wires + leftover_splits + new_wire_boxes:
        must.update(_box_cells(b))
    must.add(src)
    must.add(dst)
    for pt in extra_terminals or ():
        must.add(pt)

    missing_nodes = [n for n in must if n not in adj]
    seed = src if src in adj else (next(iter(must & set(adj))) if (must & set(adj)) else None)
    comp = _component(adj, seed) if seed is not None else set()
    disconnected = [n for n in must if n not in comp]
    connected = (not missing_nodes) and (not disconnected)

    info = {
        "n_original_boxes": len(cells),
        "n_final_boxes": len(final_cells),
        "removed_wire_boxes": removed_boxes,
        "split_wire_boxes": split_boxes,
        "dropped_point_boxes": dropped_points,
        "added_via_boxes": len(added_vias),
        "n_new_wire_boxes": len(new_wire_boxes),
        "src": list(src),
        "dst": list(dst),
        "connected": connected,
        "missing_nodes": [list(n) for n in missing_nodes],
        "disconnected_must_nodes": [list(n) for n in disconnected],
        "n_must_nodes": len(must),
        "n_component_from_src": len(comp),
    }
    return guide_boxes, info



def residual_recount(
    run: Cugr2Run,
    ripped_local: dict[str, list[tuple[int, int, int]]],
    chosen: dict[str, list[tuple[int, int, int]]],
) -> dict:
    """Full-grid residual usage after local-only rip + replace of the given nets."""
    # start from dump usage
    of_before = []
    of_after = []
    # apply deltas
    delta: dict[tuple[int, int, int], int] = defaultdict(int)
    for segs in ripped_local.values():
        for e in set(map(tuple, segs)):
            delta[e] -= 1
    for segs in chosen.values():
        for e in set(map(tuple, segs)):
            delta[e] += 1

    units_before = 0.0
    units_after = 0.0
    new_ofs = []
    cleared = []
    for L, g in run.layers.items():
        for x, row in enumerate(g.cap):
            for y, c in enumerate(row):
                u0 = g.usage[x][y]
                u1 = u0 + delta.get((L, x, y), 0)
                if u0 > 0 and u0 > c:
                    of_before.append((L, x, y, u0, c))
                    units_before += u0 - c
                if u1 > 0 and u1 > c:
                    of_after.append((L, x, y, u1, c))
                    units_after += u1 - c
                    if not (u0 > 0 and u0 > c):
                        new_ofs.append((L, x, y, u1, c, u0))
                elif u0 > 0 and u0 > c:
                    cleared.append((L, x, y, u0, u1, c))
    return {
        "overflow_edges_before": len(of_before),
        "overflow_units_before": units_before,
        "overflow_edges_after": len(of_after),
        "overflow_units_after": units_after,
        "cleared": [
            {"layer": L, "x": x, "y": y, "usage_before": u0, "usage_after": u1, "capacity": c}
            for L, x, y, u0, u1, c in cleared
        ],
        "new_overflow": [
            {"layer": L, "x": x, "y": y, "usage_after": u1, "capacity": c, "usage_before": u0}
            for L, x, y, u1, c, u0 in new_ofs
        ],
        "remaining": [
            {"layer": L, "x": x, "y": y, "usage": u, "capacity": c}
            for L, x, y, u, c in of_after
        ],
    }


def _t(pt) -> tuple[int, int, int]:
    return (int(pt[0]), int(pt[1]), int(pt[2]))


def _el(edges) -> list[tuple[int, int, int]]:
    return [(int(e[0]), int(e[1]), int(e[2])) for e in edges]


def official_delta(orig, repaired, via_field: str | None) -> dict:
    out = {
        "wl_metric_before": orig.wl_metric,
        "wl_metric_after": repaired.wl_metric,
        "delta_wl_metric": repaired.wl_metric - orig.wl_metric,
        "wl_dbu_before": orig.wl_dbu,
        "wl_dbu_after": repaired.wl_dbu,
        "delta_wl_dbu": repaired.wl_dbu - orig.wl_dbu,
        "overflow_edges_before": orig.overflow_edges,
        "overflow_edges_after": repaired.overflow_edges,
        "overflow_units_before": orig.overflow_units,
        "overflow_units_after": repaired.overflow_units,
    }
    if via_field:
        b = getattr(orig, via_field)
        a = getattr(repaired, via_field)
        out["via_field"] = via_field
        out["vias_before"] = b
        out["vias_after"] = a
        out["delta_vias"] = a - b
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WESPE legal guide write-back + CUGR2-unit scoring")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--guide", type=Path, help="original guide (default: <run>/<run-name>.guide)")
    p.add_argument("--out-guide", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--pad", type=int, default=8)
    p.add_argument("--max-paths", type=int, default=8)
    p.add_argument("--time", type=float, default=30.0)
    p.add_argument("--die", type=int, nargs=4, metavar=("XL", "YL", "XH", "YH"), default=[0, 0, 1857000, 1840000])
    p.add_argument("--expected-wl", type=int, default=28002207)
    p.add_argument("--expected-vias", type=int, default=879492)
    p.add_argument("--expected-of-edges", type=int, default=4)
    p.add_argument("--expected-of-units", type=float, default=6.0)
    p.add_argument("--skip-repair", action="store_true", help="score original guide only")
    args = p.parse_args(argv)

    run = load_run(args.run)
    geom = load_grid_geom(args.run, *args.die)
    guide = args.guide or (args.run / f"{args.run.name}.guide")
    if not guide.exists():
        raise SystemExit(f"guide not found: {guide}")

    report: dict = {
        "run": str(run.path),
        "guide": str(guide),
        "grid": f"{geom.nx} x {geom.ny} x 5",
        "die": args.die,
        "m2_pitch": geom.m2_pitch,
        "log_metrics": run.log_metrics,
        "expected": {
            "wl": args.expected_wl,
            "vias": args.expected_vias,
            "overflow_edges": args.expected_of_edges,
            "overflow_units": args.expected_of_units,
        },
    }

    print("scoring original guide...", flush=True)
    orig = score_guide(guide, geom, run, compare_usage=True)
    report["original_score"] = score_as_dict(orig)
    wl_ok = orig.wl_metric == args.expected_wl
    of_ok = orig.overflow_edges == args.expected_of_edges and orig.overflow_units == args.expected_of_units
    via_field = None
    for field in ("vias_basic_incr", "vias_endpoint_span", "vias_all_point_span", "vias_wire_layer_span"):
        if getattr(orig, field) == args.expected_vias:
            via_field = field
            break
    via_ok = via_field is not None
    usage_ok = orig.usage_mismatch_cells == 0
    # WL and OF may be official even if the via estimator is not exact.
    official_wl_of = bool(wl_ok and of_ok and usage_ok)
    report["scorer_validation"] = {
        "wl_match": wl_ok,
        "wl_scored": orig.wl_metric,
        "wl_expected": args.expected_wl,
        "of_match": of_ok,
        "of_edges_scored": orig.overflow_edges,
        "of_units_scored": orig.overflow_units,
        "via_match": via_ok,
        "via_field": via_field,
        "vias_basic_incr": orig.vias_basic_incr,
        "vias_endpoint_span": orig.vias_endpoint_span,
        "vias_all_point_span": orig.vias_all_point_span,
        "vias_wire_layer_span": orig.vias_wire_layer_span,
        "vias_expected": args.expected_vias,
        "usage_grid_exact_match": usage_ok,
        "usage_mismatch_cells": orig.usage_mismatch_cells,
        "official_wl_of": official_wl_of,
        "official_units_published": bool(official_wl_of and via_ok),
        "comment": (
            "Scorer reproduced route.log WL, vias, and OF on the original guide."
            if (official_wl_of and via_ok)
            else (
                "Scorer reproduced route.log WL and OF (and the usage grid) on the original guide; "
                "the via estimator is not exact, so repaired via totals are not official CUGR2 vias."
                if official_wl_of
                else "Scorer did not fully reproduce route.log; do not publish official-unit repaired scores for the mismatched metrics."
            )
        ),
    }
    print("validation:", json.dumps(report["scorer_validation"], indent=2), flush=True)

    if args.skip_repair:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
        return 0

    print("re-running WESPE to regenerate chosen maze paths...", flush=True)
    repair = repair_run(run, pad=args.pad, max_paths=args.max_paths, time_limit_s=args.time)
    # drop bulky path lists from the copy we embed under "repair" if needed — keep them
    report["repair_summary"] = repair.get("summary")
    report["repair_chosen_labels"] = repair.get("cpsat", {}).get("chosen")
    report["candidate_notes"] = repair.get("candidate_notes")
    report["log_metrics"] = repair.get("log_metrics") or run.log_metrics

    names = list(repair.get("chosen_paths", {}))
    if not names:
        report["status"] = "NO_CHOSEN_PATHS"
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
        return 1

    print(f"extracting {len(names)} nets from original guide...", flush=True)
    raw_nets = extract_named_nets(guide, set(names))

    replacements: dict[str, list[GuideBox]] = {}
    connectivity = {}
    ripped_local = {}
    chosen_edges = {}
    all_connected = True
    for name in names:
        inc = _el(repair["incumbent_paths"][name]["edges"])
        ch = _el(repair["chosen_paths"][name]["edges"])
        src = _t(repair["terminals"][name]["src"])
        dst = _t(repair["terminals"][name]["dst"])
        extra_pts = [_t(pt) for pt in repair["terminals"][name].get("points") or []]
        ripped_local[name] = inc
        chosen_edges[name] = ch
        boxes, info = splice_net(geom, raw_nets[name], inc, ch, src, dst, extra_terminals=extra_pts)
        replacements[name] = boxes
        info["label"] = repair["chosen_paths"][name]["label"]
        info["incumbent_wl"] = repair["incumbent_paths"][name]["wl"]
        info["chosen_wl"] = repair["chosen_paths"][name]["wl"]
        info["incumbent_vias"] = repair["incumbent_paths"][name]["vias"]
        info["chosen_vias"] = repair["chosen_paths"][name]["vias"]
        connectivity[name] = info
        if not info["connected"]:
            all_connected = False
            print(f"CONNECTIVITY FAIL {name}: {info}", flush=True)
        else:
            print(f"connectivity OK {name}: boxes {info['n_original_boxes']}->{info['n_final_boxes']}", flush=True)

    report["connectivity"] = connectivity
    report["all_nets_connected"] = all_connected

    print("residual-grid recount...", flush=True)
    report["residual_recount"] = residual_recount(run, ripped_local, chosen_edges)

    if not all_connected:
        report["status"] = "CONNECTIVITY_FAILED"
        report["comment"] = "Refusing to write a success guide; connectivity failed for at least one net."
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
        return 1

    print(f"writing repaired guide to {args.out_guide}...", flush=True)
    args.out_guide.parent.mkdir(parents=True, exist_ok=True)
    write_stats = write_guide_replacing(guide, args.out_guide, replacements)
    report["write"] = write_stats
    report["out_guide"] = str(args.out_guide)

    print("scoring repaired guide...", flush=True)
    repaired = score_guide(args.out_guide, geom, run, compare_usage=False)
    report["repaired_score"] = score_as_dict(repaired)
    report["official_unit_delta"] = official_delta(orig, repaired, via_field if via_ok else None)
    report["official_wl_of_validated"] = bool(wl_ok and of_ok)
    report["official_units_validated"] = bool(wl_ok and of_ok and via_ok)
    if wl_ok and of_ok:
        # ICCAD cost only if WL and OF validated; vias only if via_ok
        def cost(of_u, vias, wl):
            return 500 * of_u + 4 * vias + 0.5 * wl

        if via_ok:
            b = cost(orig.overflow_units, getattr(orig, via_field), orig.wl_metric)
            a = cost(repaired.overflow_units, getattr(repaired, via_field), repaired.wl_metric)
            report["iccad_cost"] = {
                "formula": "500*OF_units + 4*vias + 0.5*WL",
                "before": b,
                "after": a,
                "delta": a - b,
                "note": "Secondary. Headline remains OF/vias/WL. Reconstruction, not Table 2.",
            }
    report["status"] = "OK"
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    # also write a compact sidecar of replacements
    sidecar = args.out_guide.with_suffix(".changes.json")
    sidecar.write_text(
        json.dumps(
            {
                "source_guide": str(guide),
                "repaired_guide": str(args.out_guide),
                "nets": {
                    n: {
                        "label": repair["chosen_paths"][n]["label"],
                        "n_boxes": len(replacements[n]),
                        "connectivity": connectivity[n],
                    }
                    for n in names
                },
            },
            indent=2,
            default=str,
        )
        + "\n"
    )
    report["changes_sidecar"] = str(sidecar)

    # keep chosen/incumbent edges in the main json (small for test5)
    report["chosen_paths"] = repair["chosen_paths"]
    report["incumbent_paths"] = repair["incumbent_paths"]
    report["terminals"] = repair["terminals"]
    report["cpsat"] = {k: v for k, v in repair.get("cpsat", {}).items() if k != "chosen" or True}

    args.out_json.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("wrote", args.out_json, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
