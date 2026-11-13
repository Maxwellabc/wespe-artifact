"""Parse a wadmes CUGR2 run directory (layout, capacity3D, usage3D, overflow, window nets)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LayerGrid:
    layer: int
    direction: int  # 0 = H (edge at (x,y) goes to (x+1,y)); 1 = V (to (x,y+1))
    cap: list[list[float]]
    usage: list[list[float]]

    @property
    def nx(self) -> int:
        return len(self.cap)

    @property
    def ny(self) -> int:
        return len(self.cap[0]) if self.cap else 0


@dataclass
class OverflowEdge:
    layer: int
    x: int
    y: int
    usage: float
    capacity: float


@dataclass
class Cugr2Run:
    path: Path
    nx: int
    ny: int
    m2_pitch: int
    min_routing_layer: int
    layers: dict[int, LayerGrid]
    overflows: list[OverflowEdge]
    window_nets: dict[str, list[tuple[int, int, int]]]  # name -> [(L,x,y), ...]
    log_metrics: dict[str, float | int | str] = field(default_factory=dict)


def _parse_layout(path: Path) -> tuple[int, int, int]:
    first = path.read_text().splitlines()[0].split()
    return int(first[0]), int(first[1]), int(first[2])


def _parse_stacked(path: Path, min_layer: int, nx: int, ny: int) -> dict[int, tuple[int, list[list[float]]]]:
    lines = path.read_text().splitlines()
    layers: dict[int, tuple[int, list[list[float]]]] = {}
    i = 0
    L = min_layer
    while i < len(lines):
        direction = int(lines[i])
        i += 1
        nlines = nx - 1 + direction
        ncols = ny - direction
        rows = [list(map(float, lines[i + k].split())) for k in range(nlines)]
        for row in rows:
            if len(row) != ncols:
                raise ValueError(f"layer {L}: expected {ncols} cols, got {len(row)}")
        i += nlines
        layers[L] = (direction, rows)
        L += 1
    return layers


def _parse_overflow(path: Path) -> list[OverflowEdge]:
    out: list[OverflowEdge] = []
    if not path.exists():
        return out
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        L, x, y, u, c = raw.split()[:5]
        out.append(OverflowEdge(int(L), int(x), int(y), float(u), float(c)))
    return out


def _parse_window_nets(path: Path) -> dict[str, list[tuple[int, int, int]]]:
    nets: dict[str, list[tuple[int, int, int]]] = {}
    if not path.exists():
        return nets
    name: str | None = None
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.startswith("#"):
            continue
        parts = raw.split()
        if parts[0] == "net":
            name = parts[1]
            nets[name] = []
            continue
        if name is None:
            raise ValueError(f"segment line before net header: {raw}")
        L, x, y = map(int, parts[:3])
        nets[name].append((L, x, y))
    return nets


def _parse_log(path: Path) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    if not path.exists():
        return metrics
    for raw in path.read_text().splitlines():
        if "gcell grid:" in raw:
            metrics["gcell_grid"] = raw.split("gcell grid:")[-1].strip()
        elif "num of cells:" in raw:
            metrics["cells"] = int(raw.split(":")[-1])
        elif "num of nets :" in raw or "num of nets:" in raw:
            metrics["nets"] = int(raw.split(":")[-1])
        elif "wire length (metric):" in raw:
            metrics["wl"] = int(raw.split(":")[-1])
        elif "total via count:" in raw:
            metrics["vias"] = int(raw.split(":")[-1])
        elif "total wire overflow:" in raw:
            metrics["overflow_units"] = int(raw.split(":")[-1])
        elif "overflow G-cell edges:" in raw:
            metrics["overflow_edges"] = int(raw.split(":")[-1])
        elif "peak=" in raw and "MEM:" in raw:
            # last occurrence wins
            try:
                metrics["peak_mb"] = float(raw.split("peak=")[-1].replace("MB", "").strip())
            except ValueError:
                pass
    return metrics


def load_run(run_dir: str | Path, min_routing_layer: int = 1) -> Cugr2Run:
    run_dir = Path(run_dir)
    nx, ny, m2 = _parse_layout(run_dir / "layout.txt")
    cap_raw = _parse_stacked(run_dir / "capacity3D.txt", min_routing_layer, nx, ny)
    use_raw = _parse_stacked(run_dir / "usage3D.txt", min_routing_layer, nx, ny)
    layers = {}
    for L, (d, cap) in cap_raw.items():
        if L not in use_raw:
            raise ValueError(f"usage3D missing layer {L}")
        du, usage = use_raw[L]
        if du != d:
            raise ValueError(f"direction mismatch on layer {L}")
        layers[L] = LayerGrid(L, d, cap, usage)
    log_path = run_dir / "route.log"
    return Cugr2Run(
        path=run_dir,
        nx=nx,
        ny=ny,
        m2_pitch=m2,
        min_routing_layer=min_routing_layer,
        layers=layers,
        overflows=_parse_overflow(run_dir / "overflow_edges.txt"),
        window_nets=_parse_window_nets(run_dir / "window_net_edges.txt"),
        log_metrics=_parse_log(log_path),
    )


def residual(run: Cugr2Run, layer: int, x: int, y: int) -> float:
    g = run.layers[layer]
    return g.cap[x][y] - g.usage[x][y]


def overflow_edges_from_dumps(run: Cugr2Run) -> list[OverflowEdge]:
    out = []
    for L, g in run.layers.items():
        for x, row in enumerate(g.cap):
            for y, c in enumerate(row):
                u = g.usage[x][y]
                if u > 0 and u > c:
                    out.append(OverflowEdge(L, x, y, u, c))
    return out
