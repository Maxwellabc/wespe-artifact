"""Drive windowed exact repair on a CUGR2 run or the synthetic incompleteness demo."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from .candidates import (
    classify_local_points,
    expand_bbox_to_points,
    flute_library_status,
    maze_paths,
    nets_using_overflows,
    pattern_lz_paths,
    steiner_topology_candidates,
    via_aware_local_graph,
)
from .cpsat import Candidate, lexmin_pick
from .io_run import Cugr2Run, OverflowEdge, load_run, overflow_edges_from_dumps

Edge = tuple[int, int, int]
Node = tuple[int, int, int]


def _dir(run: Cugr2Run, L: int) -> int:
    return run.layers[L].direction


def _endpoints(run: Cugr2Run, e: Edge) -> tuple[Node, Node]:
    L, x, y = e
    if _dir(run, L) == 0:
        return (L, x, y), (L, x + 1, y)
    return (L, x, y), (L, x, y + 1)


def _incumbent_local(
    run: Cugr2Run, segs: list[Edge], ofs: list[OverflowEdge], pad: int
) -> tuple[list[Edge], Node, Node, tuple[int, int, int, int], int] | None:
    """Local chain around overflow: edges inside the padded bbox, plus estimated vias/terminals."""
    ox = [e.x for e in ofs]
    oy = [e.y for e in ofs]
    x0, x1 = max(0, min(ox) - pad), max(ox) + pad
    y0, y1 = max(0, min(oy) - pad), max(oy) + pad
    local = [s for s in segs if x0 <= s[1] <= x1 and y0 <= s[2] <= y1]
    if not local:
        return None
    # Wire-edge endpoints plus via adjacency at the same (x,y). Vias must be
    # in the degree graph; otherwise each via landing looks like a terminal
    # and a 2-pin maze will rip window-boundary stubs of the same net.
    adj: dict[Node, set[Node]] = defaultdict(set)
    layers_at: dict[tuple[int, int], set[int]] = defaultdict(set)
    for e in local:
        a, b = _endpoints(run, e)
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
    odd = [p for p, nbrs in adj.items() if len(nbrs) % 2 == 1]
    if len(odd) >= 2:
        # stable: min / max in (x+y, layer, x, y) so the pair is deterministic
        src = min(odd, key=lambda p: (p[1] + p[2], p[0], p[1], p[2]))
        dst = max(odd, key=lambda p: (p[1] + p[2], p[0], p[1], p[2]))
    else:
        pts = list(adj)
        src, dst = min(pts), max(pts)
    vias = sum(max(0, len(ls) - 1) for ls in layers_at.values())
    return local, src, dst, (x0, y0, x1, y1), vias


def repair_run(run: Cugr2Run, pad: int = 8, max_paths: int = 8, time_limit_s: float = 30.0) -> dict:
    ofs = run.overflows or overflow_edges_from_dumps(run)
    report: dict = {
        "run": str(run.path),
        "log_metrics": run.log_metrics,
        "grid": f"{run.nx} x {run.ny} x {len(run.layers)+run.min_routing_layer}",
        "overflow_edges_dump": [e.__dict__ for e in ofs],
        "n_window_nets": len(run.window_nets),
    }
    if not ofs:
        report["status"] = "NOTHING_TO_REPAIR"
        report["comment"] = "CUGR2 dump has zero overflow G-cell edges."
        return report

    ripped = nets_using_overflows(run, ofs)
    report["n_overflow_incident_nets"] = len(ripped)
    report["overflow_incident_nets"] = sorted(ripped)

    ripped_bonus: dict[Edge, float] = defaultdict(float)
    for segs in ripped.values():
        for e in segs:
            ripped_bonus[e] += 1.0

    candidates: dict[str, list[Candidate]] = {}
    notes: dict[str, str] = {}
    terminals_map: dict[str, dict] = {}
    for name, segs in ripped.items():
        of_used = [e for e in ofs if (e.layer, e.x, e.y) in set(map(tuple, segs))]
        inc = _incumbent_local(run, segs, of_used or ofs, pad)
        if inc is None:
            notes[name] = "no local chain in the window"
            continue
        local, src, dst, bbox, vias = inc
        adj_local = via_aware_local_graph(run, local)
        terminals, steiners = classify_local_points(adj_local)
        if not terminals:
            terminals = [src, dst]
        # Edge-index windows can exclude a terminal grid point (an H-edge at x
        # covers points x and x+1). Expand so maze/pattern/Steiner can reach
        # every terminal grid point (never a single bbox around all OF edges).
        bbox = expand_bbox_to_points(bbox, list(terminals) + list(steiners) + [src, dst])
        cands = [Candidate(edges=list(local), vias=vias, wl=len(local), label="incumbent")]
        alts = maze_paths(run, src, dst, bbox, ripped_bonus, max_paths=max_paths)

        def splice(bypass: list[Edge], nv: int, drop: set[Edge]) -> tuple[list[Edge], int, int]:
            kept = [e for e in local if e not in drop]
            # vias: incumbent vias, minus vias implied by dropped edges, plus bypass vias — keep additive
            edges = kept + [e for e in bypass if e not in kept]
            return edges, vias + nv, len(edges)

        # Additional, more reliable candidates: bypass each overflow edge used by this net
        # by a short maze between that edge's two endpoints (the overflow edge itself forbidden).
        of_used = [e for e in ofs if (e.layer, e.x, e.y) in set(local)]
        for k, oe in enumerate(of_used):
            a, b = _endpoints(run, (oe.layer, oe.x, oe.y))
            # shrink bbox around this edge, still allowing a few G-cells of jog
            bx0 = max(bbox[0], min(a[1], b[1]) - pad)
            by0 = max(bbox[1], min(a[2], b[2]) - pad)
            bx1 = min(bbox[2], max(a[1], b[1]) + pad)
            by1 = min(bbox[3], max(a[2], b[2]) + pad)
            # forbid the overflow edge so maze must detour
            bonus = dict(ripped_bonus)
            byps = maze_paths(run, a, b, (bx0, by0, bx1, by1), bonus, max_paths=max_paths)
            for j, (edges, nv, wl) in enumerate(byps):
                if (oe.layer, oe.x, oe.y) in edges:
                    continue
                se, sv, sw = splice(edges, nv, {(oe.layer, oe.x, oe.y)})
                cands.append(Candidate(edges=se, vias=sv, wl=sw, label=f"bypass_{k}_{j}"))

        for i, (edges, nv, wl) in enumerate(alts):
            if set(edges) == set(local):
                continue
            cands.append(Candidate(edges=edges, vias=nv, wl=wl, label=f"maze_{i}"))
        # Monotone L/Z on the incumbent's layers only (selectable; not a new layer).
        allowed = {e[0] for e in local} | {src[0], dst[0]}
        for edges, nv, wl, label in pattern_lz_paths(run, src, dst, bbox, allowed_layers=allowed):
            if not edges or set(edges) == set(local):
                continue
            cands.append(Candidate(edges=edges, vias=nv, wl=wl, label=label))
        # Hanan / flute_style / same-side / real-Flute embeddings on the
        # via-aware degree-1 terminals of this net's own window. Dedup below
        # keeps the first label, so a Steiner embedding that matches a maze
        # or pattern path stays labeled maze_* / pattern_*.
        steiner_embs, steiner_counts = steiner_topology_candidates(
            run, terminals, steiners, adj_local, bbox, allowed
        )
        n_steiner_appended = 0
        for edges, nv, wl, label in steiner_embs:
            if not edges or set(edges) == set(local):
                continue
            cands.append(Candidate(edges=edges, vias=nv, wl=wl, label=label))
            n_steiner_appended += 1
        # unique by edge set, keep first label
        uniq = []
        seen = set()
        for c in cands:
            key = tuple(sorted(set(c.edges)))
            if key in seen:
                continue
            seen.add(key)
            uniq.append(c)
        src_counts = {"incumbent": 0, "maze": 0, "bypass": 0, "pattern": 0, "steiner": 0}
        for c in uniq:
            if c.label == "incumbent":
                src_counts["incumbent"] += 1
            elif c.label.startswith("maze_"):
                src_counts["maze"] += 1
            elif c.label.startswith("bypass_"):
                src_counts["bypass"] += 1
            elif c.label.startswith("pattern_"):
                src_counts["pattern"] += 1
            else:
                src_counts["steiner"] += 1
        notes[name] = (
            f"terminals {src}->{dst} (deg1={len(terminals)}, steiner={len(steiners)}); "
            f"{len(uniq)} unique candidates "
            f"(incumbent, end-to-end maze, overflow-edge bypass, pattern L/Z, "
            f"Hanan/flute_style/same-side/flute_rsmt); kept unique by source {src_counts}; "
            f"generated steiner embeddings {n_steiner_appended} {steiner_counts}"
        )
        candidates[name] = uniq
        terminals_map[name] = {
            "src": list(src),
            "dst": list(dst),
            "points": [list(p) for p in terminals],
            "steiners": [list(p) for p in steiners],
            "bbox": list(bbox),
            "incumbent_vias": vias,
            "candidate_sources": src_counts,
            "steiner_generation": steiner_counts,
        }

    report["candidate_notes"] = notes
    report["flute_library"] = flute_library_status()
    if len(candidates) != len(ripped):
        missing = set(ripped) - set(candidates)
        report["status"] = "INCOMPLETE_CANDIDATES"
        report["comment"] = f"missing candidates for {sorted(missing)}"
        # still try on those we have? safer to stop honestly
        return report

    result = lexmin_pick(run, ripped, candidates, time_limit_s=time_limit_s)
    report["cpsat"] = {k: v for k, v in result.__dict__.items()}
    # attach regenerable chosen / incumbent embeddings for guide write-back
    chosen_paths = {}
    incumbent_paths = {}
    for name, cs in candidates.items():
        incumbent_paths[name] = {
            "label": cs[0].label,
            "vias": cs[0].vias,
            "wl": cs[0].wl,
            "edges": [list(e) for e in cs[0].edges],
        }
        label = result.chosen.get(name)
        for c in cs:
            if c.label == label:
                chosen_paths[name] = {
                    "label": c.label,
                    "vias": c.vias,
                    "wl": c.wl,
                    "edges": [list(e) for e in c.edges],
                }
                break
    report["chosen_paths"] = chosen_paths
    report["incumbent_paths"] = incumbent_paths
    report["terminals"] = terminals_map
    # human summary
    before_e = result.overflow_edges_before
    after_e = result.overflow_edges_after
    report["summary"] = {
        "overflow_edges": {"before": before_e, "after": after_e},
        "overflow_units": {"before": result.overflow_units_before, "after": result.overflow_units_after},
        "delta_vias": result.delta_vias,
        "delta_wl_gcell_edges": result.delta_wl,
        "status": result.status,
        "reason": result.residual_reason,
    }
    return report


def repair_demo() -> dict:
    """CP-SAT exact repair on the synthetic incompleteness instance (not the enumerator)."""
    # local imports so the CUGR2 path does not require the demo package
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alns-refiner"))
    from alns_refiner.demo_incompleteness import CUT_X, bad_solution, good_solution, path_reselection_cannot_clear, score
    from alns_refiner.operators import same_side_trees
    from alns_refiner.grid import Grid2D
    from ortools.sat.python import cp_model
    import copy
    from itertools import product as iproduct

    bad = bad_solution()
    good = good_solution()
    infeasible, best_of = path_reselection_cannot_clear(bad)
    s_bad, s_good = score(bad), score(good)

    # Build an explicit candidate set: original + same-side topologies, each
    # 2-pin with all monotone L/Z paths, as complete embeddings (tree+paths).
    def embeddings(tree, grid: Grid2D):
        path_opts = []
        for tp in tree.two_pins:
            opts = grid.all_monotone_paths(tp.u, tp.v) or [list(tp.path)]
            path_opts.append(opts)
        uniq = {}
        for combo in iproduct(*path_opts):
            edges = []
            for path in combo:
                edges.extend(path)  # path is already a list of (x,y,dir) edges
            key = tuple(sorted(set(edges)))
            uniq.setdefault(key, combo)
        return list(uniq.items())

    cand_by_net = []
    for i, net in enumerate(bad.nets):
        opts = [copy.deepcopy(net)] + same_side_trees(net, CUT_X)
        uniq_trees = []
        seen = set()
        for t in opts:
            k = tuple(sorted(t.tree_edges))
            if k not in seen:
                seen.add(k)
                uniq_trees.append(t)
        embs = []
        for t in uniq_trees:
            embs.extend(embeddings(t, bad.grid))
        cand_by_net.append(embs)

    model = cp_model.CpModel()
    x = {}
    for n, embs in enumerate(cand_by_net):
        for i in range(len(embs)):
            x[n, i] = model.NewBoolVar(f"y_{n}_{i}")
        model.Add(sum(x[n, i] for i in range(len(embs))) == 1)

    # demand on every grid edge that appears
    all_edges = set()
    for embs in cand_by_net:
        for key, _ in embs:
            all_edges.update(key)
    overflow = {}
    for e in all_edges:
        demand = []
        for n, embs in enumerate(cand_by_net):
            for i, (key, _) in enumerate(embs):
                if e in key:
                    demand.append(x[n, i])
        cap = bad.grid.cap.get(e, bad.grid.default_cap)
        of = model.NewIntVar(0, len(cand_by_net), f"of_{e}")
        overflow[e] = of
        model.Add(of >= sum(demand) - int(cap))
        model.Add(of >= 0)
    # wl = sum of unique edges * already unit; we also compute via proxy after
    wl_terms = []
    for n, embs in enumerate(cand_by_net):
        for i, (key, _) in enumerate(embs):
            wl_terms.append(len(key) * x[n, i])
    total_of = sum(overflow.values())
    total_wl = sum(wl_terms)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    model.Minimize(total_of)
    st = solver.Solve(model)
    of_star = int(solver.Value(total_of)) if st in (cp_model.OPTIMAL, cp_model.FEASIBLE) else None
    if of_star is not None:
        model.Add(total_of == of_star)
        model.Minimize(total_wl)
        st = solver.Solve(model)
    report = {
        "bad_forest": s_bad.as_dict(),
        "path_reselection_only": {
            "cleared_overflow": (not infeasible),
            "best_overflow_edge_count": best_of,
        },
        "good_forest": s_good.as_dict(),
        "cpsat_status": solver.StatusName(st),
        "cpsat_overflow_edges": of_star,
        "cpsat_wl_unique_edges": int(solver.Value(total_wl)) if of_star is not None else None,
        "comment": "CP-SAT is exact for the enumerated same-side topologies × monotone L/Z paths.",
    }
    return report
