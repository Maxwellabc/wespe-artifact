#!/usr/bin/env python3
"""Constructed Claim-1 instance with >=3 via-aware terminals on real test5 geometry.

This is NOT a native CUGR2 overflow residual, and it is not a second legal
write-back. Native overflow-local fragments on test5/test8 are inspected
first; if none has >=3 via-aware degree-1 terminals, a residual is sited on
the honest 618 x 613 x 5 grid at the Steiner neighborhood of test5 net36665
(world cell (425, 192)), away from the four native Metal2 overflow edges.

Two 3-pin nets copy the synthetic same-side example, with via-aware 3-D
terminals (Metal2 / Metal3). Residual capacity in the 3 x 3 window is frozen
to a declared cut. We compare:

  (a) monotone L/Z path reselection on a declared bad forest / pattern DAG
  (b) Steiner candidates: same-side, flute_style, real Flute, Hanan slides
  (c) unique-per-net 3-D maze on the same terminals: k-shortest and exhaustive
      s-t walks on the bad forest, AND (because maze_paths is only an s-t
      enumerator and no multi-terminal maze exists in the code) exhaustive
      s-t maze on every spanning tree of the same 3 terminals.

Success for a strong Steiner-necessary certificate is only: Steiner (or
FLUTE/Hanan) clears OF to 0 AND maze on the same terminals does not.
If maze still clears, the report is labeled honestly. Do not invent a
certificate, weaken maze, or cook a GCELLGRID.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from copy import deepcopy
from itertools import combinations, product
from pathlib import Path

ROOT = Path("/workspace/dac2027-3793")
sys.path.insert(0, str(ROOT / "code"))

from wespe.candidates import (
    classify_local_points,
    contracted_tree_segments,
    flute_library_status,
    maze_paths,
    nets_using_overflows,
    pattern_lz_paths,
    steiner_topology_candidates,
    via_aware_local_graph,
)
from wespe.cpsat import Candidate, lexmin_pick
from wespe.io_run import Cugr2Run, LayerGrid, OverflowEdge, load_run
from wespe.repair import _endpoints, _incumbent_local

# Site the 3 x 3 so it contains net36665's Steiner cell (425, 192).
X0, Y0 = 424, 191
W = 3
BBOX = (0, 0, 2, 2)
PIN_LAYER_DEFAULT = 1
H_LAYERS = (1, 3)
V_LAYERS = (2, 4)
GATE_LAYER = 1
DEFAULT_CAP = 4.0

# Pins as via-aware 3-D terminals (layer, local_x, local_y). World = (X0+x, Y0+y).
# Net A: two on the left (Metal2), one on the right (Metal3) — 3 terminals.
PINS_A = [(1, 0, 0), (1, 0, 2), (2, 2, 1)]
# Net B: one on the left (Metal2), two on the right (Metal3 / Metal2).
PINS_B = [(1, 0, 0), (2, 2, 0), (1, 2, 2)]

# Tree edges as pin-index pairs.
BAD_A = [(0, 2), (1, 2)]  # star at the right pin
BAD_B = [(1, 0), (2, 0)]  # star at the left pin
GOOD_A = [(0, 1), (1, 2)]  # same-side then one bridge
GOOD_B = [(1, 2), (0, 1)]


def xy(p: tuple[int, int, int]) -> tuple[int, int]:
    return (p[1], p[2])


def world_xy(p: tuple[int, int, int]) -> tuple[int, int]:
    return (X0 + p[1], Y0 + p[2])


def node(p: tuple[int, int, int]) -> tuple[int, int, int]:
    return (p[0], p[1], p[2])


def inspect_native(run: Cugr2Run, pads=(2, 4, 8, 16, 32, 64, 128)) -> dict:
    ripped = nets_using_overflows(run, run.overflows)
    nets = {}
    any_local_ge3 = False
    any_full_ge3 = False
    for name, segs in sorted(ripped.items()):
        of_used = [e for e in run.overflows if (e.layer, e.x, e.y) in set(map(tuple, segs))]
        adj = via_aware_local_graph(run, segs)
        terms, steiners = classify_local_points(adj)
        if len(terms) >= 3:
            any_full_ge3 = True
        windows = []
        for pad in pads:
            inc = _incumbent_local(run, segs, of_used or run.overflows, pad)
            if inc is None:
                continue
            local, src, dst, bbox, _vias = inc
            t, s = classify_local_points(via_aware_local_graph(run, local))
            rec = {
                "pad": pad,
                "bbox": list(bbox),
                "n_local_edges": len(local),
                "n_deg1": len(t),
                "n_steiner": len(s),
            }
            if len(t) >= 3 or s:
                rec["terminals"] = [list(p) for p in t]
                rec["steiners"] = [list(p) for p in s]
                any_local_ge3 = True
            windows.append(rec)
        of_xy = [(e.layer, e.x, e.y) for e in of_used]
        steiner_dist = []
        for S in steiners:
            dists = [
                {"of": [e.layer, e.x, e.y], "chebyshev": max(abs(e.x - S[1]), abs(e.y - S[2]))}
                for e in of_used or run.overflows
            ]
            steiner_dist.append({"steiner": list(S), "to_own_or_all_of": dists})
        # compact neighborhood around each full-tree Steiner (not an OF window)
        nbhds = []
        for S in steiners:
            for radius in (2, 4, 8):
                sx, sy = S[1], S[2]
                local = []
                for e in segs:
                    a, b = _endpoints(run, e)
                    if any(max(abs(p[1] - sx), abs(p[2] - sy)) <= radius for p in (a, b)):
                        local.append(e)
                t, s = classify_local_points(via_aware_local_graph(run, local))
                nbhds.append({
                    "steiner": list(S),
                    "radius": radius,
                    "n_local_edges": len(local),
                    "n_deg1": len(t),
                    "n_steiner": len(s),
                    "terminals": [list(p) for p in t],
                    "steiners": [list(p) for p in s],
                    "min_chebyshev_to_any_native_of": min(
                        max(abs(e.x - sx), abs(e.y - sy)) for e in run.overflows
                    ) if run.overflows else None,
                })
        nets[name] = {
            "n_window_net_edges": len(segs),
            "overflow_edges": of_xy,
            "full_tree_n_deg1": len(terms),
            "full_tree_n_steiner": len(steiners),
            "full_tree_terminals": [list(p) for p in terms],
            "full_tree_steiners": [list(p) for p in steiners],
            "contracted_segments": [
                [list(a), list(b)] for a, b in contracted_tree_segments(adj)
            ],
            "steiner_distance_to_overflow": steiner_dist,
            "overflow_padded_windows": windows,
            "steiner_neighborhoods_not_at_overflow": nbhds,
        }
    return {
        "run": str(run.path),
        "grid": f"{run.nx} x {run.ny} x {len(run.layers) + run.min_routing_layer}",
        "log_gcell_grid": run.log_metrics.get("gcell_grid"),
        "n_overflow_edges": len(run.overflows),
        "overflows": [
            {"layer": e.layer, "x": e.x, "y": e.y, "usage": e.usage, "capacity": e.capacity}
            for e in run.overflows
        ],
        "any_overflow_padded_window_with_ge3_deg1": any_local_ge3,
        "any_full_dumped_tree_with_ge3_deg1": any_full_ge3,
        "nets": nets,
    }


def make_constructed_run(test5: Cugr2Run) -> Cugr2Run:
    assert test5.nx >= X0 + W and test5.ny >= Y0 + W, "origin out of test5 grid"
    directions = {L: test5.layers[L].direction for L in (1, 2, 3, 4)}
    assert directions[1] == 0 and directions[3] == 0, "expected M2/M4 horizontal"
    assert directions[2] == 1 and directions[4] == 1, "expected M3/M5 vertical"

    native_cut = []
    for L, d in directions.items():
        if d != 0:
            continue
        g = test5.layers[L]
        for ly in range(W):
            wx, wy = X0 + 0, Y0 + ly
            if wx < g.nx and wy < g.ny:
                native_cut.append({
                    "layer": L,
                    "world": [wx, wy],
                    "cap": g.cap[wx][wy],
                    "usage": g.usage[wx][wy],
                    "residual": g.cap[wx][wy] - g.usage[wx][wy],
                })

    layers: dict[int, LayerGrid] = {}
    for L, d in directions.items():
        if d == 0:
            nx_e, ny_e = W - 1, W
        else:
            nx_e, ny_e = W, W - 1
        cap = [[DEFAULT_CAP for _ in range(ny_e)] for _ in range(nx_e)]
        usage = [[0.0 for _ in range(ny_e)] for _ in range(nx_e)]
        if d == 0:
            for ly in range(W):
                if ly == 1:
                    cap[0][ly] = 0.0
                elif L == GATE_LAYER:
                    cap[0][ly] = 1.0
                else:
                    cap[0][ly] = 0.0
        layers[L] = LayerGrid(L, d, cap, usage)

    run = Cugr2Run(
        path=Path("constructed_3term_on_test5_geometry"),
        nx=W,
        ny=W,
        m2_pitch=test5.m2_pitch,
        min_routing_layer=1,
        layers=layers,
        overflows=[],
        window_nets={},
        log_metrics={
            "source_grid": test5.log_metrics.get("gcell_grid", "618 x 613 x 5"),
            "origin_world": f"{X0},{Y0}",
            "label": "constructed_3term_on_real_geometry",
        },
    )
    run._native_cut = native_cut  # type: ignore[attr-defined]
    return run


def all_walks(run, src, dst, bbox, max_wl: int = 8, max_paths: int = 400):
    from collections import deque
    from wespe.candidates import _neighbors

    found = []
    seen_edge_sets: set[tuple] = set()
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
    return Candidate(edges=list(edges), vias=int(vias), wl=int(wl), label=label)


def _is_strictly_monotone(src, dst, label: str) -> bool:
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


def embeddings_for_tree(run, pins, pairs, bbox, allowed, kind: str) -> list[Candidate]:
    per_edge: list[list[Candidate]] = []
    for i, (a, b) in enumerate(pairs):
        s, t = node(pins[a]), node(pins[b])
        if kind == "pattern":
            cs = pattern_cands(run, s, t, bbox, allowed, f"e{i}")
        elif kind == "maze_k":
            cs = [cand_of(edges, vias, wl, f"e{i}:maze")
                  for edges, vias, wl in maze_paths(run, s, t, bbox, {}, max_paths=12)]
        elif kind == "maze_all":
            cs = []
            seen = set()
            for edges, vias, wl in all_walks(run, s, t, bbox):
                key = tuple(sorted(set(edges)))
                seen.add(key)
                cs.append(cand_of(edges, vias, wl, f"e{i}:walk"))
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
        uniq = list(dict.fromkeys(edges))
        key = tuple(sorted(set(uniq)))
        lab = "+".join(labels)
        if key not in merged or (vias, len(uniq)) < (merged[key].vias, merged[key].wl):
            merged[key] = cand_of(uniq, vias, len(uniq), lab)
    return list(merged.values())


def spanning_trees(n: int) -> list[list[tuple[int, int]]]:
    """All distinct spanning trees on n labeled pins (stars / paths for n=3)."""
    # For n=3 this is the three stars (equivalently the three 2-edge paths).
    verts = list(range(n))
    edges = list(combinations(verts, 2))
    trees = []
    for combo in combinations(edges, n - 1):
        # check connectedness / acyclicity via Union-Find
        parent = {v: v for v in verts}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        ok = True
        for a, b in combo:
            ra, rb = find(a), find(b)
            if ra == rb:
                ok = False
                break
            parent[ra] = rb
        if ok and len({find(v) for v in verts}) == 1:
            trees.append([tuple(e) for e in combo])
    return trees


def tagged(run, pins, pairs, tag, allowed, kind="pattern"):
    cs = embeddings_for_tree(run, pins, pairs, BBOX, allowed, kind)
    for c in cs:
        c.label = f"{tag}:{c.label}"
    return cs


def steiner_cands(run, pins, allowed) -> tuple[list[Candidate], dict]:
    terminals = [node(p) for p in pins]
    # No incumbent local Steiner in the declared bad forest (stars at a pin).
    embs, counts = steiner_topology_candidates(
        run, terminals, [], {}, BBOX, allowed, max_embs_per_topo=64
    )
    out = [cand_of(e, v, w, lab) for e, v, w, lab in embs]
    return out, counts


def solve(run, cand_by_net, overflows):
    run = deepcopy(run)
    run.overflows = list(overflows)
    return lexmin_pick(run, {n: [] for n in cand_by_net}, cand_by_net, time_limit_s=20.0)


def score_embedding(run, net_edges):
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
    edges = []
    for a, b in pairs:
        cs = pattern_cands(run, node(pins[a]), node(pins[b]), bbox, allowed, "inc")
        if not cs:
            return []
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


def clears(res) -> bool:
    return res.status in ("OPTIMAL", "FEASIBLE") and res.overflow_edges_after == 0


def main() -> int:
    test5 = load_run(ROOT / "runs" / "ispd18_test5_metal5")
    test8 = load_run(ROOT / "runs" / "ispd19_test8_metal5")
    native = {
        "test5": inspect_native(test5),
        "test8": inspect_native(test8),
    }
    native_of_local_ge3 = (
        native["test5"]["any_overflow_padded_window_with_ge3_deg1"]
        or native["test8"]["any_overflow_padded_window_with_ge3_deg1"]
    )

    native_ofs = [(e.layer, e.x, e.y) for e in test5.overflows]
    window_cells = {(X0 + x, Y0 + y) for x in range(W) for y in range(W)}
    overlap = [e for e in native_ofs if (e[1], e[2]) in window_cells]

    run = make_constructed_run(test5)
    allowed_inc = {1, 2}
    allowed_all = {1, 2, 3, 4}

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

    # (a) monotone L/Z on the BAD forest only
    pat_bad = {
        "A": embeddings_for_tree(run, PINS_A, BAD_A, BBOX, allowed_inc, "pattern"),
        "B": embeddings_for_tree(run, PINS_B, BAD_B, BBOX, allowed_inc, "pattern"),
    }
    res_pat_bad = solve(run, pat_bad, overflows)

    # (b) pattern on BAD + same-side good trees
    pat_exp = {
        "A": tagged(run, PINS_A, BAD_A, "bad", allowed_inc) + tagged(run, PINS_A, GOOD_A, "good", allowed_inc),
        "B": tagged(run, PINS_B, BAD_B, "bad", allowed_inc) + tagged(run, PINS_B, GOOD_B, "good", allowed_inc),
    }
    res_pat_exp = solve(run, pat_exp, overflows)

    # (b') Hanan / flute_style / real Flute (and same-side inside flute_style)
    steiner_A, counts_A = steiner_cands(run, PINS_A, allowed_inc)
    steiner_B, counts_B = steiner_cands(run, PINS_B, allowed_inc)
    for c in steiner_A:
        c.label = "steiner:" + c.label
    for c in steiner_B:
        c.label = "steiner:" + c.label
    steiner_set = {
        "A": tagged(run, PINS_A, BAD_A, "bad", allowed_inc) + steiner_A,
        "B": tagged(run, PINS_B, BAD_B, "bad", allowed_inc) + steiner_B,
    }
    res_steiner = solve(run, steiner_set, overflows)

    # (c) 3-D maze on BAD forest terminals (s-t enumerator, unique-per-net)
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

    # (c') s-t maze on EVERY spanning tree of the same 3 terminals.
    # maze_paths / all_walks are s-t only; no multi-terminal maze/forest
    # enumerator exists in code/wespe. This is the honest strongest forest
    # we can build from the existing s-t enumerator without inventing a
    # weaker maze. For 3 terminals it includes the good topology.
    trees = spanning_trees(3)
    maze_forest_A = []
    maze_forest_B = []
    for i, tree in enumerate(trees):
        maze_forest_A.extend(tagged(run, PINS_A, tree, f"tree{i}{tree}", allowed_all, "maze_all"))
        maze_forest_B.extend(tagged(run, PINS_B, tree, f"tree{i}{tree}", allowed_all, "maze_all"))
    # dedup by edge set, keep best vias/wl
    def dedup(cs: list[Candidate]) -> list[Candidate]:
        m = {}
        for c in cs:
            key = tuple(sorted(set(c.edges)))
            if key not in m or (c.vias, c.wl) < (m[key].vias, m[key].wl):
                m[key] = c
        return list(m.values())

    maze_forest = {"A": dedup(maze_forest_A), "B": dedup(maze_forest_B)}
    res_maze_forest = solve(run, maze_forest, overflows)

    pattern_bad_clears = clears(res_pat_bad)
    same_side_clears = clears(res_pat_exp)
    steiner_op_clears = clears(res_steiner)
    maze_clears = clears(res_maze_all)
    maze_k_clears = clears(res_maze_k)
    maze_forest_clears = clears(res_maze_forest)
    steiner_any_clears = same_side_clears or steiner_op_clears

    # Strong form: Steiner clears AND (exhaustive s-t maze on the declared
    # bad forest AND the all-spanning-tree maze forest) do not.
    # If only the bad-forest maze fails but the all-tree maze clears, that
    # is still not Steiner-necessary against 3-D maze on the same terminals.
    maze_any_clears = maze_clears or maze_k_clears or maze_forest_clears
    strong = (not maze_any_clears) and steiner_any_clears and (not pattern_bad_clears)

    if strong:
        certificate_kind = "STEINER_NECESSARY_CONSTRUCTED"
        label = "constructed_3term_steiner_necessary"
        certificate = (
            "STEINER_NECESSARY_CONSTRUCTED: monotone pattern L/Z on the "
            "declared bad forest cannot clear OF, and 3-D maze on the same "
            "via-aware terminals (k-shortest and exhaustive s-t on the bad "
            "forest, and s-t maze on every spanning tree) also cannot, while "
            "a Steiner topology (same-side / Hanan / flute_style / real Flute) "
            "reaches OF 0. Constructed instance on real geometry, not a "
            "native CUGR2 residual."
        )
    elif (not pattern_bad_clears) and steiner_any_clears and maze_any_clears:
        certificate_kind = "NOT_STEINER_NECESSARY"
        label = "constructed_3term_pattern_incompleteness_maze_also_clears"
        certificate = (
            "NOT_STEINER_NECESSARY: monotone L/Z path reselection on the "
            "declared bad forest does not clear overflow, and a Steiner "
            "change (same-side and/or Hanan/flute_style/real Flute) does "
            "(pattern-DAG incompleteness, as in the 2-D synthetic). But 3-D "
            "maze on the same via-aware terminals, with unique-per-net demand "
            "and the window bbox, also reaches OF 0. wespe.maze_paths is an "
            "s-t path enumerator only; no multi-terminal maze/forest "
            "enumerator exists in the code. Exhaustive s-t walks on the bad "
            "tree edges already suffice to mimic the good tree, because a "
            "Steiner tree on these terminals is a union of walks between the "
            "same endpoints (unique-per-net demand shares the stem). s-t maze "
            "on every spanning tree was also run and likewise clears. This is "
            "a constructed instance on real test5 G-cell coordinates, not a "
            "native CUGR2 residual, and it is not a Steiner-necessary "
            "certificate against 3-D maze."
        )
    elif pattern_bad_clears:
        certificate_kind = "NO_CERTIFICATE"
        label = "constructed_3term_no_certificate"
        certificate = (
            "NO_CERTIFICATE: even monotone L/Z path reselection on the "
            "declared bad forest already reaches OF 0, so this construction "
            "does not certify Claim 1. Do not invent a certificate."
        )
    else:
        certificate_kind = "NO_CERTIFICATE"
        label = "constructed_3term_no_certificate"
        certificate = (
            "NO_CERTIFICATE: the constructed instance did not produce a "
            "clean pattern-infeasible / Steiner-clears (or maze-fails / "
            "Steiner-clears) result. Do not invent a certificate."
        )

    report = {
        "label": label,
        "certificate_kind": certificate_kind,
        "comment": certificate,
        "native_fragment_search": {
            "found_overflow_local_ge3_terminals": native_of_local_ge3,
            "note": (
                "Every overflow-padded window (pad up to 128) on test5 and "
                "test8 still has deg1=2 and no local Steiner. Some full "
                "dumped trees of overflow-incident nets have >=3 via-aware "
                "terminals and a Steiner point, but those Steiner points lie "
                "far from the native overflow edges (test5 net36665 Steiner "
                "at (1,425,192) is 166 Chebyshev from its overflow at "
                "(1,591,192); nearest native OF is 103 away). Compact "
                "3-terminal neighborhoods exist around those distant "
                "Steiners; they are not native overflow residuals. Therefore "
                "this increment constructs a residual on real coordinates, "
                "sited so the 3x3 window contains (425,192)."
            ),
            "inspection": native,
        },
        "honesty": {
            "native_cugr2_residual": False,
            "constructed_on_real_geometry": True,
            "source_run": str(test5.path),
            "source_grid": test5.log_metrics.get("gcell_grid"),
            "m2_pitch": test5.m2_pitch,
            "origin_world": [X0, Y0],
            "window_local": [0, 0, 2, 2],
            "window_world": {"x": [X0, X0 + 2], "y": [Y0, Y0 + 2]},
            "contains_net36665_steiner_cell": [425, 192] == [X0 + 1, Y0 + 1],
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
                "derived residual (freeze + blocked cut), not native usage. "
                "Not a legal write-back."
            ),
            "maze_is_st_path_enumerator": True,
            "multi_terminal_maze_in_code": False,
            "all_spanning_tree_maze_added": True,
            "official_guides_untouched": True,
        },
        "nets": {
            "A": {
                "terminals_local_3d": PINS_A,
                "terminals_world_3d": [[p[0], X0 + p[1], Y0 + p[2]] for p in PINS_A],
                "n_via_aware_terminals": len(PINS_A),
                "bad_tree": BAD_A,
                "good_tree": GOOD_A,
            },
            "B": {
                "terminals_local_3d": PINS_B,
                "terminals_world_3d": [[p[0], X0 + p[1], Y0 + p[2]] for p in PINS_B],
                "n_via_aware_terminals": len(PINS_B),
                "bad_tree": BAD_B,
                "good_tree": GOOD_B,
            },
        },
        "bad_forest_incumbent": bad_score,
        "good_forest_incumbent": good_score,
        "steiner_operator_counts": {"A": counts_A, "B": counts_B, "flute_library": flute_library_status()},
        "n_candidates": {
            "pattern_bad_A": len(pat_bad["A"]),
            "pattern_bad_B": len(pat_bad["B"]),
            "pattern_exp_A": len(pat_exp["A"]),
            "pattern_exp_B": len(pat_exp["B"]),
            "steiner_ops_A": len(steiner_A),
            "steiner_ops_B": len(steiner_B),
            "maze_k_A": len(maze_k["A"]),
            "maze_k_B": len(maze_k["B"]),
            "maze_all_bad_A": len(maze_all["A"]),
            "maze_all_bad_B": len(maze_all["B"]),
            "maze_all_spanning_trees": [list(t) for t in trees],
            "maze_forest_A": len(maze_forest["A"]),
            "maze_forest_B": len(maze_forest["B"]),
        },
        "experiments": {
            "pattern_LZ_on_bad_forest": result_dict(res_pat_bad),
            "pattern_LZ_plus_same_side_steiner": result_dict(res_pat_exp),
            "pattern_LZ_plus_hanan_flute_style_rsmt": result_dict(res_steiner),
            "maze_kshort_on_bad_terminals": result_dict(res_maze_k),
            "maze_exhaustive_on_bad_terminals": result_dict(res_maze_all),
            "maze_exhaustive_on_all_spanning_trees": result_dict(res_maze_forest),
        },
        "summary": {
            "n_via_aware_terminals_per_net": 3,
            "pattern_on_bad_forest_clears": pattern_bad_clears,
            "pattern_on_bad_forest_best_OF": res_pat_bad.overflow_edges_after,
            "same_side_steiner_clears": same_side_clears,
            "hanan_flute_steiner_clears": steiner_op_clears,
            "maze_kshort_on_bad_clears": maze_k_clears,
            "maze_on_bad_terminals_clears": maze_clears,
            "maze_on_bad_terminals_best_OF": res_maze_all.overflow_edges_after,
            "maze_on_all_spanning_trees_clears": maze_forest_clears,
            "maze_on_all_spanning_trees_best_OF": res_maze_forest.overflow_edges_after,
            "steiner_necessary_vs_3d_maze": bool(strong),
        },
    }

    out = ROOT / "runs" / "wespe" / "claim1_constructed_3term_test5.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({
        "out": str(out),
        "label": label,
        "certificate_kind": certificate_kind,
        "summary": report["summary"],
        "native_of_local_ge3": native_of_local_ge3,
        "flute": flute_library_status(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
