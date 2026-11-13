#!/usr/bin/env python3
"""Constructed Claim-1 instance on real ispd18_test5 G-cell geometry.

This is NOT a native CUGR2 residual. We take the reconstructed test5
618 x 613 x 5 grid (layer directions and pitch from layout.txt), site a
3 x 3 window at an interior G-cell origin that is not one of the four
native overflow edges, freeze residual capacity in the window to a
declared cut (two unit gates, blocked middle crossing, all H-layers),
and place two 3-pin nets that copy the synthetic same-side example.

Label: constructed instance on real geometry, not a native CUGR2 residual.

The interesting certificate would be: 3-D maze on the current pin-set /
tree-edge terminals cannot clear overflow, while a Steiner-topology
change reaches OF 0. If maze on those terminals still clears, the
instance is not Steiner-necessary in that strong sense.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from itertools import product
from pathlib import Path

ROOT = Path("/workspace/dac2027-3793")
sys.path.insert(0, str(ROOT / "code"))

from wespe.candidates import maze_paths, pattern_lz_paths, same_side_topologies
from wespe.cpsat import Candidate, lexmin_pick
from wespe.io_run import Cugr2Run, LayerGrid, OverflowEdge, load_run

# Interior origin on the honest test5 618 x 613 grid, away from the native
# Metal2 overflow edges (318,198), (319,198), (322,192), (591,192).
X0, Y0 = 400, 300
W = 3  # 3 x 3 G-cells, local coordinates 0..2
BBOX = (0, 0, 2, 2)  # maze / pattern bbox in local coordinates
CUT_X = 1  # cut is the H-edges at local x=0 (between x=0 and x=1)
PIN_LAYER = 1  # Metal2
H_LAYERS = (1, 3)  # Metal2, Metal4
V_LAYERS = (2, 4)  # Metal3, Metal5
GATE_LAYER = 1
DEFAULT_CAP = 4.0

# Pins in local (x, y), then world (X0+x, Y0+y).
# Net A: two on the left, one on the right.
PINS_A = [(0, 0), (0, 2), (2, 1)]
# Net B: one on the left, two on the right. Shares the G-cell (0,0) with A
# as in the 2-D synthetic (different nets, same G-cell).
PINS_B = [(0, 0), (2, 0), (2, 2)]

# Bad forest: A is a star at the right pin; B is a star at the left pin.
BAD_A = [(0, 2), (1, 2)]  # pin index pairs
BAD_B = [(1, 0), (2, 0)]
# Good forest: same-side then one bridge.
GOOD_A = [(0, 1), (1, 2)]
GOOD_B = [(1, 2), (0, 1)]


def world(p: tuple[int, int]) -> tuple[int, int]:
    return (X0 + p[0], Y0 + p[1])


def node(p: tuple[int, int], layer: int = PIN_LAYER) -> tuple[int, int, int]:
    return (layer, p[0], p[1])


def make_constructed_run(test5: Cugr2Run) -> Cugr2Run:
    """Tiny 3 x 3 x 5 residual with test5 layer directions and a blocked cut."""
    assert test5.nx >= X0 + W and test5.ny >= Y0 + W, "origin out of test5 grid"
    layers: dict[int, LayerGrid] = {}
    # Confirm layer directions on the real run.
    directions = {L: test5.layers[L].direction for L in (1, 2, 3, 4)}
    assert directions[1] == 0 and directions[3] == 0, "expected M2/M4 horizontal"
    assert directions[2] == 1 and directions[4] == 1, "expected M3/M5 vertical"

    # Record native residual at the cut before we overwrite (documentation).
    native_cut = []
    for L, d in directions.items():
        g = test5.layers[L]
        if d != 0:
            continue
        for ly in range(W):
            wx, wy = X0 + 0, Y0 + ly  # H-edge at world (X0, Y0+ly)
            if wx < g.nx and wy < g.ny:
                native_cut.append(
                    {
                        "layer": L,
                        "world": [wx, wy],
                        "cap": g.cap[wx][wy],
                        "usage": g.usage[wx][wy],
                        "residual": g.cap[wx][wy] - g.usage[wx][wy],
                    }
                )

    for L, d in directions.items():
        if d == 0:  # H: cap[x][y] for x in 0..1, y in 0..2
            nx_e, ny_e = W - 1, W
        else:  # V: cap[x][y] for x in 0..2, y in 0..1
            nx_e, ny_e = W, W - 1
        cap = [[DEFAULT_CAP for _ in range(ny_e)] for _ in range(nx_e)]
        usage = [[0.0 for _ in range(ny_e)] for _ in range(nx_e)]
        if d == 0:
            # The cut: H-edges at local x=0, all rows, all H-layers.
            for ly in range(W):
                if ly == 1:
                    cap[0][ly] = 0.0  # blocked middle
                elif L == GATE_LAYER:
                    cap[0][ly] = 1.0  # unit gate on Metal2 only
                else:
                    cap[0][ly] = 0.0  # other H-layers at the gates also closed
        layers[L] = LayerGrid(L, d, cap, usage)

    run = Cugr2Run(
        path=Path("constructed_on_test5_geometry"),
        nx=W,
        ny=W,
        m2_pitch=test5.m2_pitch,
        min_routing_layer=1,
        layers=layers,
        overflows=[],  # filled after we score the bad incumbent
        window_nets={},
        log_metrics={
            "source_grid": test5.log_metrics.get("gcell_grid", "618 x 613 x 5"),
            "origin_world": f"{X0},{Y0}",
            "label": "constructed_instance_on_real_geometry",
        },
    )
    run._native_cut = native_cut  # type: ignore[attr-defined]
    return run


def all_walks(
    run: Cugr2Run,
    src: tuple[int, int, int],
    dst: tuple[int, int, int],
    bbox: tuple[int, int, int, int],
    max_wl: int = 8,
    max_paths: int = 400,
) -> list[tuple[list[tuple[int, int, int]], int, int]]:
    """Simple walks in a tiny window, shortest wirelength first (BFS)."""
    from collections import deque
    from wespe.candidates import _neighbors

    found: list[tuple[list[tuple[int, int, int]], int, int]] = []
    seen_edge_sets: set[tuple[tuple[int, int, int], ...]] = set()
    # state: node, frozenset of visited (L,x,y), edges, vias, wl
    start = (src, frozenset([src]), tuple(), 0, 0)
    q = deque([start])
    while q and len(found) < max_paths:
        u, vis, edges, vias, wl = q.popleft()
        if u == dst:
            key = tuple(sorted(set(edges)))
            if key not in seen_edge_sets:
                seen_edge_sets.add(key)
                found.append((list(edges), vias, wl))
            continue
        if wl >= max_wl:
            continue
        for v, e, kind in _neighbors(run, u, bbox):
            if not (bbox[0] <= v[1] <= bbox[2] and bbox[1] <= v[2] <= bbox[3]):
                continue
            if v in vis:
                continue
            if kind == "via":
                q.append((v, vis | {v}, edges, vias + 1, wl))
            elif e is not None:
                q.append((v, vis | {v}, edges + (e,), vias, wl + 1))
    return found


def cand_of(edges, vias, wl, label) -> Candidate:
    # unique edges for unique-per-net demand (lexmin_pick already uses set())
    return Candidate(edges=list(edges), vias=int(vias), wl=int(wl), label=label)


def _is_strictly_monotone(src, dst, label: str) -> bool:
    """True for L-shapes, and for Z-shapes whose mid lies between the endpoints.

    wespe.pattern_lz_paths also emits Z midpoints *outside* the endpoint
    interval (e.g. VZV with mid y=2 on a y=0-->y=1 terminal pair). Those
    walks are real maze paths, but they are not monotone L/Z in the sense
    of Claim 1 and of the 2-D synthetic. This experiment filters them out
    of the pattern set; they remain available to the maze enumerator.
    """
    if "pattern_L_" in label:
        return True
    sx, sy = src[1], src[2]
    dx, dy = dst[1], dst[2]
    if "pattern_Z_HZH_" in label:
        mid = int(label.rsplit("_", 1)[-1])
        return min(sx, dx) < mid < max(sx, dx)
    if "pattern_Z_VZV_" in label:
        mid = int(label.rsplit("_", 1)[-1])
        return min(sy, dy) < mid < max(sy, dy)
    return False


def pattern_cands(run, src, dst, bbox, allowed, prefix: str) -> list[Candidate]:
    out = []
    for edges, vias, wl, lab in pattern_lz_paths(run, src, dst, bbox, allowed):
        if not _is_strictly_monotone(src, dst, lab):
            continue
        out.append(cand_of(edges, vias, wl, f"{prefix}:{lab}"))
    return out


def embeddings_for_tree(
    run,
    pins,
    pairs,
    bbox,
    allowed,
    kind: str,
) -> list[Candidate]:
    """Cartesian product of per-tree-edge candidates, unique-per-net union."""
    per_edge: list[list[Candidate]] = []
    for i, (a, b) in enumerate(pairs):
        s, t = node(pins[a]), node(pins[b])
        if kind == "pattern":
            cs = pattern_cands(run, s, t, bbox, allowed, f"e{i}")
        elif kind == "maze_k":
            cs = []
            for edges, vias, wl in maze_paths(run, s, t, bbox, {}, max_paths=12):
                cs.append(cand_of(edges, vias, wl, f"e{i}:maze"))
        elif kind == "maze_all":
            cs = []
            seen = set()
            for edges, vias, wl in all_walks(run, s, t, bbox):
                key = tuple(sorted(set(edges)))
                seen.add(key)
                cs.append(cand_of(edges, vias, wl, f"e{i}:walk"))
            # also keep k-shortest maze paths (Dijkstra), in case the walk cap drops any
            for edges, vias, wl in maze_paths(run, s, t, bbox, {}, max_paths=12):
                key = tuple(sorted(set(edges)))
                if key not in seen:
                    seen.add(key)
                    cs.append(cand_of(edges, vias, wl, f"e{i}:maze"))
        else:
            raise ValueError(kind)
        if not cs:
            return []
        per_edge.append(cs)

    merged: dict[tuple, Candidate] = {}
    for combo in product(*per_edge):
        edges = []
        vias = 0
        labels = []
        for c in combo:
            edges.extend(c.edges)
            vias += c.vias
            labels.append(c.label)
        # unique edges, unique-per-net
        uniq = list(dict.fromkeys(edges))
        key = tuple(sorted(set(uniq)))
        lab = "+".join(labels)
        if key not in merged or (vias, len(uniq)) < (merged[key].vias, merged[key].wl):
            merged[key] = cand_of(uniq, vias, len(uniq), lab)
    return list(merged.values())


def solve(run, cand_by_net: dict[str, list[Candidate]], overflows, ripped=None):
    run = deepcopy(run)
    run.overflows = list(overflows)
    return lexmin_pick(run, ripped or {n: [] for n in cand_by_net}, cand_by_net, time_limit_s=20.0)


def score_embedding(run, net_edges: dict[str, list[tuple[int, int, int]]]):
    demand: dict[tuple[int, int, int], int] = {}
    for edges in net_edges.values():
        for e in set(edges):
            demand[e] = demand.get(e, 0) + 1
    of_edges = 0
    of_units = 0.0
    of_list = []
    for e, d in demand.items():
        L, x, y = e
        g = run.layers[L]
        cap = g.cap[x][y] if 0 <= x < g.nx and 0 <= y < g.ny else 0.0
        if d > 0 and d > cap:
            of_edges += 1
            of_units += d - cap
            of_list.append({"edge": [L, x, y], "demand": d, "capacity": cap})
    return {
        "overflow_edges": of_edges,
        "overflow_units": of_units,
        "overflow_list": of_list,
        "demand_cut": {
            "y0": demand.get((GATE_LAYER, 0, 0), 0),
            "y1": demand.get((GATE_LAYER, 0, 1), 0) + demand.get((3, 0, 1), 0),
            "y2": demand.get((GATE_LAYER, 0, 2), 0),
        },
    }


def first_pattern_embedding(run, pins, pairs, bbox, allowed):
    """One L-shape (or first pattern) per tree edge, for an incumbent picture."""
    edges = []
    for a, b in pairs:
        cs = pattern_cands(run, node(pins[a]), node(pins[b]), bbox, allowed, "inc")
        if not cs:
            return []
        # prefer a shortest pattern
        cs = sorted(cs, key=lambda c: (c.wl, c.vias, c.label))
        edges.extend(cs[0].edges)
    return list(dict.fromkeys(edges))


def result_dict(res) -> dict:
    return {
        "status": res.status,
        "overflow_edges_after": res.overflow_edges_after,
        "overflow_units_after": res.overflow_units_after,
        "delta_vias": res.delta_vias,
        "delta_wl": res.delta_wl,
        "chosen": res.chosen,
        "residual_reason": res.residual_reason,
        "wall_s": res.wall_s,
    }


def main() -> int:
    test5 = load_run(ROOT / "runs" / "ispd18_test5_metal5")
    native_ofs = [(e.layer, e.x, e.y) for e in test5.overflows]
    window_world = [(X0 + x, Y0 + y) for x in range(W) for y in range(W)]
    overlap = [e for e in native_ofs if (e[1], e[2]) in {(X0 + x, Y0 + y) for x in range(W) for y in range(W)}]

    run = make_constructed_run(test5)
    allowed_inc = {1, 2}  # incumbent layers Metal2/Metal3, as in the pattern-only tests
    allowed_all = {1, 2, 3, 4}

    # Incumbent embeddings for the bad forest (pattern L/Z, one each).
    inc_A = first_pattern_embedding(run, PINS_A, BAD_A, BBOX, allowed_inc)
    inc_B = first_pattern_embedding(run, PINS_B, BAD_B, BBOX, allowed_inc)
    bad_score = score_embedding(run, {"A": inc_A, "B": inc_B})
    overflows = [
        OverflowEdge(e["edge"][0], e["edge"][1], e["edge"][2], e["demand"], e["capacity"])
        for e in bad_score["overflow_list"]
    ]

    good_inc_A = first_pattern_embedding(run, PINS_A, GOOD_A, BBOX, allowed_inc)
    good_inc_B = first_pattern_embedding(run, PINS_B, GOOD_B, BBOX, allowed_inc)
    good_score = score_embedding(run, {"A": good_inc_A, "B": good_inc_B})

    # --- Experiment 1: pattern L/Z on the BAD forest only ---
    pat_bad = {
        "A": embeddings_for_tree(run, PINS_A, BAD_A, BBOX, allowed_inc, "pattern"),
        "B": embeddings_for_tree(run, PINS_B, BAD_B, BBOX, allowed_inc, "pattern"),
    }
    res_pat_bad = solve(run, pat_bad, overflows)

    # --- Experiment 2: pattern L/Z on BAD + same-side (good) topologies ---
    pat_exp = {
        "A": embeddings_for_tree(run, PINS_A, BAD_A, BBOX, allowed_inc, "pattern")
        + embeddings_for_tree(run, PINS_A, GOOD_A, BBOX, allowed_inc, "pattern"),
        "B": embeddings_for_tree(run, PINS_B, BAD_B, BBOX, allowed_inc, "pattern")
        + embeddings_for_tree(run, PINS_B, GOOD_B, BBOX, allowed_inc, "pattern"),
    }
    # tag labels so we can see which topology was chosen
    for n, extra_pairs, pins in (
        ("A", GOOD_A, PINS_A),
        ("B", GOOD_B, PINS_B),
    ):
        pass
    # Relabel: first block bad, second block good — rebuild with prefixes
    def tagged(run, pins, pairs, tag, allowed=allowed_inc, kind="pattern"):
        cs = embeddings_for_tree(run, pins, pairs, BBOX, allowed, kind)
        for c in cs:
            c.label = f"{tag}:{c.label}"
        return cs

    pat_exp = {
        "A": tagged(run, PINS_A, BAD_A, "bad") + tagged(run, PINS_A, GOOD_A, "good"),
        "B": tagged(run, PINS_B, BAD_B, "bad") + tagged(run, PINS_B, GOOD_B, "good"),
    }
    res_pat_exp = solve(run, pat_exp, overflows)

    # --- Experiment 3: 3-D maze (all layers) on BAD forest terminals ---
    # k-shortest first, then exhaustive walks (the window is 3 x 3).
    maze_k = {
        "A": tagged(run, PINS_A, BAD_A, "bad", allowed_all, "maze_k"),
        "B": tagged(run, PINS_B, BAD_B, "bad", allowed_all, "maze_k"),
    }
    res_maze_k = solve(run, maze_k, overflows)

    maze_all = {
        "A": tagged(run, PINS_A, BAD_A, "bad", allowed_all, "maze_all"),
        "B": tagged(run, PINS_B, BAD_B, "bad", allowed_all, "maze_all"),
    }
    res_maze_all = solve(run, maze_all, overflows)

    # Maze on the GOOD forest (sanity): should also clear.
    maze_good = {
        "A": tagged(run, PINS_A, GOOD_A, "good", allowed_all, "maze_all"),
        "B": tagged(run, PINS_B, GOOD_B, "good", allowed_all, "maze_all"),
    }
    res_maze_good = solve(run, maze_good, overflows)

    # Did exhaustive maze-on-bad-terminals find a zero-OF solution?
    maze_clears = (
        res_maze_all.status in ("OPTIMAL", "FEASIBLE")
        and res_maze_all.overflow_edges_after == 0
    )
    pattern_bad_clears = (
        res_pat_bad.status in ("OPTIMAL", "FEASIBLE")
        and res_pat_bad.overflow_edges_after == 0
    )
    steiner_clears = (
        res_pat_exp.status in ("OPTIMAL", "FEASIBLE")
        and res_pat_exp.overflow_edges_after == 0
    )

    if (not pattern_bad_clears) and (not maze_clears) and steiner_clears:
        certificate = (
            "STEINER_NECESSARY_CONSTRUCTED: both monotone pattern L/Z and "
            "3-D maze on the current tree-edge terminals cannot clear OF, "
            "and a same-side Steiner topology reaches OF 0. Constructed "
            "instance on real geometry, not a native CUGR2 residual."
        )
        label = "constructed_steiner_necessary"
    elif (not pattern_bad_clears) and maze_clears and steiner_clears:
        certificate = (
            "NOT_STEINER_NECESSARY: monotone L/Z path reselection on the "
            "bad forest does not clear overflow, and a same-side Steiner "
            "change does (pattern-DAG incompleteness, as in the 2-D "
            "synthetic). But 3-D maze on the current tree-edge terminals, "
            "with unique-per-net demand and the window bbox, also reaches "
            "OF 0 — typically by a non-monotone walk that geometrically "
            "mimics the good tree. This is a constructed instance on real "
            "test5 G-cell coordinates, not a native CUGR2 residual, and it "
            "is not a Steiner-necessary certificate against 3-D maze."
        )
        label = "constructed_pattern_incompleteness_maze_also_clears"
    elif pattern_bad_clears:
        certificate = (
            "NO_CERTIFICATE: even monotone L/Z path reselection on the "
            "declared bad forest already reaches OF 0, so this construction "
            "does not certify Claim 1. Do not invent a certificate."
        )
        label = "constructed_no_certificate"
    else:
        certificate = (
            "NO_CERTIFICATE: the constructed instance did not produce a "
            "clean pattern-infeasible / Steiner-clears (or maze-fails / "
            "Steiner-clears) result. Do not invent a certificate."
        )
        label = "constructed_no_certificate"

    report = {
        "label": label,
        "certificate_kind": certificate.split(":")[0],
        "comment": certificate,
        "honesty": {
            "native_cugr2_residual": False,
            "constructed_on_real_geometry": True,
            "source_run": str(test5.path),
            "source_grid": test5.log_metrics.get("gcell_grid"),
            "m2_pitch": test5.m2_pitch,
            "origin_world": [X0, Y0],
            "window_local": [0, 0, 2, 2],
            "window_world": {"x": [X0, X0 + 2], "y": [Y0, Y0 + 2]},
            "native_overflow_edges_in_window": overlap,
            "native_overflow_edges_all": [
                {"layer": e.layer, "x": e.x, "y": e.y, "usage": e.usage, "capacity": e.capacity}
                for e in test5.overflows
            ],
            "native_residual_on_cut_column_before_overwrite": run._native_cut,  # type: ignore
            "capacity_model": (
                "In the 3x3 window: default cap 4; the H-cut at local x=0 has "
                "Metal2 cap 1,0,1 on rows y=0,1,2 and cap 0 on Metal4 and on "
                "the middle Metal2 crossing. Maze bbox is the window, so a "
                "path cannot walk around the cut outside the 3x3. This is a "
                "derived residual (freeze + blocked cut), not native usage."
            ),
        },
        "nets": {
            "A": {"pins_local": PINS_A, "pins_world": [world(p) for p in PINS_A],
                  "bad_tree": BAD_A, "good_tree": GOOD_A},
            "B": {"pins_local": PINS_B, "pins_world": [world(p) for p in PINS_B],
                  "bad_tree": BAD_B, "good_tree": GOOD_B},
        },
        "bad_forest_incumbent": bad_score,
        "good_forest_incumbent": good_score,
        "n_candidates": {
            "pattern_bad_A": len(pat_bad["A"]),
            "pattern_bad_B": len(pat_bad["B"]),
            "pattern_exp_A": len(pat_exp["A"]),
            "pattern_exp_B": len(pat_exp["B"]),
            "maze_k_A": len(maze_k["A"]),
            "maze_k_B": len(maze_k["B"]),
            "maze_all_A": len(maze_all["A"]),
            "maze_all_B": len(maze_all["B"]),
        },
        "experiments": {
            "pattern_LZ_on_bad_forest": result_dict(res_pat_bad),
            "pattern_LZ_plus_same_side_steiner": result_dict(res_pat_exp),
            "maze_kshort_on_bad_terminals": result_dict(res_maze_k),
            "maze_exhaustive_on_bad_terminals": result_dict(res_maze_all),
            "maze_exhaustive_on_good_forest": result_dict(res_maze_good),
        },
        "summary": {
            "pattern_on_bad_forest_clears": pattern_bad_clears,
            "pattern_on_bad_forest_best_OF": res_pat_bad.overflow_edges_after,
            "same_side_steiner_clears": steiner_clears,
            "maze_on_bad_terminals_clears": maze_clears,
            "maze_on_bad_terminals_best_OF": res_maze_all.overflow_edges_after,
            "steiner_necessary_vs_3d_maze": bool((not maze_clears) and steiner_clears),
        },
    }

    out = ROOT / "runs" / "wespe" / "claim1_constructed_test5.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"out": str(out), "label": label,
                       "summary": report["summary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
