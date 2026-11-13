from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="WESPE: Windowed Exact Repair with Adaptive Steiner Forest Expansion")
    p.add_argument("--demo", action="store_true", help="run the synthetic incompleteness instance with CP-SAT")
    p.add_argument("--run", type=Path, help="CUGR2 run directory (layout.txt, usage3D.txt, ...)")
    p.add_argument("--pad", type=int, default=8, help="G-cell padding around overflow edges")
    p.add_argument("--max-paths", type=int, default=8)
    p.add_argument("--time", type=float, default=30.0, help="CP-SAT time limit in seconds")
    p.add_argument("--out", type=Path, help="write JSON report here")
    args = p.parse_args(argv)

    reports = []
    if args.demo or args.run is None:
        from .repair import repair_demo
        demo = repair_demo()
        demo["instance"] = "synthetic_incompleteness"
        reports.append(demo)
    if args.run is not None:
        from .io_run import load_run
        from .repair import repair_run
        run = load_run(args.run)
        reports.append(repair_run(run, pad=args.pad, max_paths=args.max_paths, time_limit_s=args.time))

    text = json.dumps(reports if len(reports) > 1 else reports[0], indent=2, default=str)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")

    # non-zero only if the demo (when run) failed its theory certificate
    for r in reports:
        if r.get("instance") == "synthetic_incompleteness":
            of = r.get("cpsat_overflow_edges")
            if of != 0 or r.get("path_reselection_only", {}).get("cleared_overflow"):
                print("DEMO FAILED", file=sys.stderr)
                return 1
            print("DEMO OK: path reselection cannot clear overflow; CP-SAT topology expansion reaches OF=0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
