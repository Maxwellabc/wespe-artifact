#!/usr/bin/env python3
"""Official-v2-grid candidate-set ablation. Read-only on legal guides.

Writes JSON only under runs/wespe-ablation-official-v2/. Never writes a .guide.
Does not touch official-v2 legal write-backs or public-DEF guides.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path("/workspace/dac2027-3793")
OUT = ROOT / "runs/wespe-ablation-official-v2"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "code"))

from wespe.candidates import (  # noqa: E402
    classify_local_points,
    expand_bbox_to_points,
    flute_library_status,
    maze_paths,
    nets_using_overflows,
    pattern_lz_paths,
    steiner_topology_candidates,
    via_aware_local_graph,
)
from wespe.cpsat import Candidate, lexmin_pick  # noqa: E402
from wespe.io_run import overflow_edges_from_dumps, load_run  # noqa: E402
from wespe.repair import _endpoints, _incumbent_local  # noqa: E402

CASES = {
    "ispd18_test5_metal5": ROOT / "runs/official-v2-metal5/ispd18_test5_metal5",
    "ispd19_test8_metal5": ROOT / "runs/official-v2-metal5/ispd19_test8_metal5",
}

# Family filters. Incumbent is always kept.
SETTINGS = [
    ("full", {"incumbent", "maze", "bypass", "pattern", "steiner"}),
    ("maze_bypass", {"incumbent", "maze", "bypass"}),
    ("pattern_only", {"incumbent", "pattern"}),
    ("pattern_steiner", {"incumbent", "pattern", "steiner"}),
    ("incumbent_only", {"incumbent"}),
]


def family_of(label: str) -> str:
    if label == "incumbent":
        return "incumbent"
    if label.startswith("maze_"):
        return "maze"
    if label.startswith("bypass_"):
        return "bypass"
    if label.startswith("pattern_"):
        return "pattern"
    return "steiner"


def build_candidates(run, pad: int = 8, max_paths: int = 8):
    ofs = run.overflows or overflow_edges_from_dumps(run)
    ripped = nets_using_overflows(run, ofs)
    ripped_bonus: dict = defaultdict(float)
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
        bbox = expand_bbox_to_points(bbox, list(terminals) + list(steiners) + [src, dst])
        cands = [Candidate(edges=list(local), vias=vias, wl=len(local), label="incumbent")]
        alts = maze_paths(run, src, dst, bbox, ripped_bonus, max_paths=max_paths)

        def splice(bypass, nv, drop):
            kept = [e for e in local if e not in drop]
            edges = kept + [e for e in bypass if e not in kept]
            return edges, vias + nv, len(edges)

        of_used = [e for e in ofs if (e.layer, e.x, e.y) in set(local)]
        for k, oe in enumerate(of_used):
            a, b = _endpoints(run, (oe.layer, oe.x, oe.y))
            bx0 = max(bbox[0], min(a[1], b[1]) - pad)
            by0 = max(bbox[1], min(a[2], b[2]) - pad)
            bx1 = min(bbox[2], max(a[1], b[1]) + pad)
            by1 = min(bbox[3], max(a[2], b[2]) + pad)
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
        allowed = {e[0] for e in local} | {src[0], dst[0]}
        for edges, nv, wl, label in pattern_lz_paths(run, src, dst, bbox, allowed_layers=allowed):
            if not edges or set(edges) == set(local):
                continue
            cands.append(Candidate(edges=edges, vias=nv, wl=wl, label=label))
        steiner_embs, steiner_counts = steiner_topology_candidates(
            run, terminals, steiners, adj_local, bbox, allowed
        )
        n_steiner_appended = 0
        for edges, nv, wl, label in steiner_embs:
            if not edges or set(edges) == set(local):
                continue
            cands.append(Candidate(edges=edges, vias=nv, wl=wl, label=label))
            n_steiner_appended += 1
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
            src_counts[family_of(c.label)] += 1
        notes[name] = (
            f"terminals {src}->{dst} (deg1={len(terminals)}, steiner={len(steiners)}); "
            f"{len(uniq)} unique candidates; kept unique by source {src_counts}; "
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
            "n_unique": len(uniq),
        }
    return ofs, ripped, candidates, notes, terminals_map


def slim_result(result, families, n_cands):
    return {
        "families": sorted(families),
        "n_candidates_kept": n_cands,
        "status": result.status,
        "overflow_edges_before": result.overflow_edges_before,
        "overflow_units_before": result.overflow_units_before,
        "overflow_edges_after": result.overflow_edges_after,
        "overflow_units_after": result.overflow_units_after,
        "delta_vias": result.delta_vias,
        "delta_wl_gcell_edges": result.delta_wl,
        "chosen": result.chosen,
        "residual_reason": result.residual_reason,
        "wall_s": result.wall_s,
    }


def run_case(name: str, run_dir: Path) -> dict:
    print(f"=== load {name} ===", flush=True)
    run = load_run(run_dir)
    ofs, ripped, candidates, notes, terminals_map = build_candidates(run)
    report = {
        "case": name,
        "run": str(run_dir),
        "label": (
            "official-v2-DEF grids + our 5-layer truncation; "
            "candidate-set ablation; no guide written"
        ),
        "grid": f"{run.nx} x {run.ny} x {len(run.layers)+run.min_routing_layer}",
        "log_metrics": run.log_metrics,
        "overflow_edges_dump": [e.__dict__ for e in ofs],
        "overflow_incident_nets": sorted(ripped),
        "flute_library": flute_library_status(),
        "candidate_notes": notes,
        "terminals": terminals_map,
        "settings": {},
        "wrote_guide": False,
    }
    for tag, families in SETTINGS:
        filtered = {}
        n_kept = {}
        for net, cs in candidates.items():
            kept = [c for c in cs if family_of(c.label) in families]
            # guarantee incumbent
            if not any(c.label == "incumbent" for c in kept):
                inc = next(c for c in cs if c.label == "incumbent")
                kept = [inc] + kept
            filtered[net] = kept
            n_kept[net] = {
                "n": len(kept),
                "by_family": {
                    f: sum(1 for c in kept if family_of(c.label) == f)
                    for f in ("incumbent", "maze", "bypass", "pattern", "steiner")
                },
            }
        print(f"  solve {tag} ...", flush=True)
        result = lexmin_pick(run, ripped, filtered, time_limit_s=30.0)
        rec = slim_result(result, families, {k: v["n"] for k, v in n_kept.items()})
        rec["n_kept_detail"] = n_kept
        report["settings"][tag] = rec
        print(
            f"    {tag}: status={result.status} OF {result.overflow_edges_before}"
            f"->{result.overflow_edges_after} chosen={result.chosen} wall_s={result.wall_s}",
            flush=True,
        )
    out = OUT / f"{name}_ablation.json"
    out.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print("wrote", out, flush=True)
    return report


def main():
    summary = {"wrote_guide": False, "cases": {}}
    for name, run_dir in CASES.items():
        summary["cases"][name] = run_case(name, run_dir)
    out = OUT / "summary.json"
    # drop bulky notes from the rolled-up summary? keep settings only
    slim = {
        "wrote_guide": False,
        "label": "official-v2 candidate-set ablation; legal guides not written or modified",
        "cases": {},
    }
    for name, rep in summary["cases"].items():
        slim["cases"][name] = {
            "grid": rep["grid"],
            "overflow_incident_nets": rep["overflow_incident_nets"],
            "flute_library": rep["flute_library"],
            "terminals": {
                n: {
                    "candidate_sources": t["candidate_sources"],
                    "n_unique": t["n_unique"],
                    "steiners": t["steiners"],
                    "src": t["src"],
                    "dst": t["dst"],
                }
                for n, t in rep["terminals"].items()
            },
            "settings": {
                tag: {
                    k: v
                    for k, v in rec.items()
                    if k != "n_kept_detail"
                } | {"n_kept_detail": rec["n_kept_detail"]}
                for tag, rec in rep["settings"].items()
            },
        }
    out.write_text(json.dumps(slim, indent=2, default=str) + "\n")
    print("wrote", out, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
