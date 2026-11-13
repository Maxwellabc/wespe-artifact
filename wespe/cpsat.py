"""OR-Tools CP-SAT window ILP: pick one candidate embedding per net, lexmin OF, vias, WL."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from ortools.sat.python import cp_model

from .io_run import Cugr2Run, OverflowEdge

Edge = tuple[int, int, int]


@dataclass
class Candidate:
    edges: list[Edge]
    vias: int
    wl: int
    label: str = ""


@dataclass
class RepairResult:
    status: str
    overflow_edges_before: int
    overflow_units_before: float
    overflow_edges_after: int
    overflow_units_after: float
    delta_vias: int
    delta_wl: int
    chosen: dict[str, str] = field(default_factory=dict)
    residual_reason: str = ""
    wall_s: float = 0.0


def _lookup(run: Cugr2Run, e: Edge) -> tuple[float, float]:
    L, x, y = e
    g = run.layers.get(L)
    if g is None or not (0 <= x < g.nx and 0 <= y < g.ny):
        return 0.0, 0.0
    return g.cap[x][y], g.usage[x][y]


def _edge_caps_and_ripped(
    run: Cugr2Run, ripped: dict[str, list[Edge]], window_edges: set[Edge] | None = None
) -> tuple[dict[Edge, float], dict[Edge, float], dict[Edge, float]]:
    """cap, original usage, usage of ripped nets — window-restricted, no full-grid dicts."""
    cap: dict[Edge, float] = {}
    usage: dict[Edge, float] = {}
    ripped_u: dict[Edge, float] = defaultdict(float)
    want: set[Edge] = set(window_edges or ())
    want.update((e.layer, e.x, e.y) for e in run.overflows)
    for segs in ripped.values():
        for e in segs:
            e = (e[0], e[1], e[2])
            ripped_u[e] += 1.0
            want.add(e)
    for e in want:
        c, u = _lookup(run, e)
        cap[e] = c
        usage[e] = u
    return cap, usage, dict(ripped_u)


def lexmin_pick(
    run: Cugr2Run,
    ripped: dict[str, list[Edge]],
    candidates: dict[str, list[Candidate]],
    window_edges: set[Edge] | None = None,
    time_limit_s: float = 30.0,
) -> RepairResult:
    if window_edges is None:
        window_edges = set()
        for cs in candidates.values():
            for c in cs:
                window_edges.update(c.edges)
        window_edges.update((e.layer, e.x, e.y) for e in run.overflows)
    cap, usage, ripped_u = _edge_caps_and_ripped(run, ripped, window_edges)

    def of_count_from_run() -> tuple[int, float]:
        n_e, units = 0, 0.0
        if run.overflows:
            for e in run.overflows:
                n_e += 1
                units += e.usage - e.capacity
            return n_e, units
        for L, g in run.layers.items():
            for x, row in enumerate(g.cap):
                for y, c in enumerate(row):
                    u = g.usage[x][y]
                    if u > 0 and u > c:
                        n_e += 1
                        units += u - c
        return n_e, units

    b_e, b_u = of_count_from_run()

    model = cp_model.CpModel()
    x: dict[tuple[str, int], cp_model.IntVar] = {}
    for n, cs in candidates.items():
        if not cs:
            return RepairResult(
                status="NO_CANDIDATES",
                overflow_edges_before=b_e,
                overflow_units_before=b_u,
                overflow_edges_after=b_e,
                overflow_units_after=b_u,
                delta_vias=0,
                delta_wl=0,
                residual_reason=f"net {n} has an empty candidate set",
            )
        for i, _c in enumerate(cs):
            x[n, i] = model.NewBoolVar(f"x_{n}_{i}")
        model.Add(sum(x[n, i] for i in range(len(cs))) == 1)

    # integer scaled residual: capacities may be fractional (0.138 etc.)
    # Use milli-units (round to 0.001) so CP-SAT stays integral.
    SCALE = 1000

    def milli(v: float) -> int:
        return int(round(v * SCALE))

    overflow_vars: dict[Edge, cp_model.IntVar] = {}
    for e in window_edges:
        c_left = cap.get(e, 0.0) - (usage.get(e, 0.0) - ripped_u.get(e, 0.0))
        demand = []
        for n, cs in candidates.items():
            for i, cand in enumerate(cs):
                # a candidate may use an edge more than once; unique-per-net demand counts once
                if e in set(cand.edges):
                    demand.append(x[n, i])
        # overflow_milli >= SCALE * demand - milli(c_left), overflow >= 0
        of = model.NewIntVar(0, SCALE * max(1, len(candidates)), f"of_{e}")
        overflow_vars[e] = of
        d = sum(demand) if demand else 0
        # SCALE * d - milli(c_left)
        model.Add(of >= SCALE * d - milli(c_left))

    # original vias/wl inside the window (for delta)
    # we only score candidates relative to candidate 0 if it is the incumbent
    via_expr = []
    wl_expr = []
    for n, cs in candidates.items():
        for i, cand in enumerate(cs):
            via_expr.append(cand.vias * x[n, i])
            wl_expr.append(cand.wl * x[n, i])

    total_of = sum(overflow_vars.values())
    total_via = sum(via_expr)
    total_wl = sum(wl_expr)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 8

    # lexmin: overflow, then vias, then wl
    model.Minimize(total_of)
    st1 = solver.Solve(model)
    if st1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return RepairResult(
            status=solver.StatusName(st1),
            overflow_edges_before=b_e,
            overflow_units_before=b_u,
            overflow_edges_after=b_e,
            overflow_units_after=b_u,
            delta_vias=0,
            delta_wl=0,
            residual_reason="CP-SAT failed on the overflow objective",
            wall_s=solver.WallTime(),
        )
    of_star = int(solver.Value(total_of))
    model.Add(total_of == of_star)
    model.Minimize(total_via)
    st2 = solver.Solve(model)
    if st2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return RepairResult(
            status=solver.StatusName(st2),
            overflow_edges_before=b_e,
            overflow_units_before=b_u,
            overflow_edges_after=b_e,
            overflow_units_after=b_u,
            delta_vias=0,
            delta_wl=0,
            residual_reason="CP-SAT failed on the via objective after fixing overflow",
            wall_s=solver.WallTime(),
        )
    via_star = int(solver.Value(total_via))
    model.Add(total_via == via_star)
    model.Minimize(total_wl)
    st3 = solver.Solve(model)
    if st3 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return RepairResult(
            status=solver.StatusName(st3),
            overflow_edges_before=b_e,
            overflow_units_before=b_u,
            overflow_edges_after=b_e,
            overflow_units_after=b_u,
            delta_vias=0,
            delta_wl=0,
            residual_reason="CP-SAT failed on the wirelength objective",
            wall_s=solver.WallTime(),
        )

    chosen = {}
    via_new = 0
    wl_new = 0
    via_old = 0
    wl_old = 0
    selected_edges: dict[Edge, int] = defaultdict(int)
    for n, cs in candidates.items():
        via_old += cs[0].vias
        wl_old += cs[0].wl
        for i, cand in enumerate(cs):
            if solver.Value(x[n, i]):
                chosen[n] = cand.label or f"{i}"
                via_new += cand.vias
                wl_new += cand.wl
                for e in set(cand.edges):
                    selected_edges[e] += 1
                break

    # global overflow after: original usage, minus ripped, plus selected.
    # Only edges whose usage changes, plus the original overflow set, can change the OF set.
    a_e, a_u = 0, 0.0
    touched = set(selected_edges) | set(ripped_u) | set((e.layer, e.x, e.y) for e in run.overflows)
    for e in touched:
        if e not in usage or e not in cap:
            c, u0 = _lookup(run, e)
        else:
            c, u0 = cap[e], usage[e]
        u = u0 - ripped_u.get(e, 0.0) + selected_edges.get(e, 0)
        if u > 0 and u > c:
            a_e += 1
            a_u += u - c

    return RepairResult(
        status=solver.StatusName(st3),
        overflow_edges_before=b_e,
        overflow_units_before=b_u,
        overflow_edges_after=a_e,
        overflow_units_after=a_u,
        delta_vias=via_new - via_old,
        delta_wl=wl_new - wl_old,
        chosen=chosen,
        residual_reason=(
            "cleared all overflow edges in the residual instance"
            if a_e == 0
            else f"{a_e} overflow edge(s) remain after an optimal candidate-set repair"
        ),
        wall_s=solver.WallTime(),
    )
