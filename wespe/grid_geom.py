"""G-cell gridlines, DBU box conversion, and CUGR2 edge-length WL units.

Mirrors Design.cpp implicit GCELLGRID (defaultGridlineSpacing=3000) and
GridGraph::getEdgeLength / getCellBox / GlobalRouter::write.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# MetalLayer::H = 0, MetalLayer::V = 1. Metal2/4 are H, Metal1/3/5 are V.
LAYER_DIRECTION = {0: 1, 1: 0, 2: 1, 3: 0, 4: 1}
LAYER_NAME = {0: "Metal1", 1: "Metal2", 2: "Metal3", 3: "Metal4", 4: "Metal5"}
NAME_TO_LAYER = {v: k for k, v in LAYER_NAME.items()}


def implicit_gridlines(low: int, high: int, step: int = 3000) -> list[int]:
    """Design.cpp: for i = low; i + step < high; i += step, then append high."""
    lines: list[int] = []
    i = low
    while i + step < high:
        lines.append(i)
        i += step
    if not lines or lines[-1] != high:
        lines.append(high)
    return lines


@dataclass
class GridGeom:
    gx: list[int]  # x gridlines, length nx+1
    gy: list[int]  # y gridlines, length ny+1
    edge_len_h: list[int]  # length nx-1, index = x of an H-edge (L,x,y)
    edge_len_v: list[int]  # length ny-1, index = y of a V-edge (L,x,y)
    m2_pitch: int
    die_xl: int
    die_yl: int
    die_xh: int
    die_yh: int
    _xindex: dict[int, int] | None = None
    _yindex: dict[int, int] | None = None

    @property
    def nx(self) -> int:
        return len(self.gx) - 1

    @property
    def ny(self) -> int:
        return len(self.gy) - 1

    def xindex(self) -> dict[int, int]:
        if self._xindex is None:
            self._xindex = {v: i for i, v in enumerate(self.gx)}
        return self._xindex

    def yindex(self) -> dict[int, int]:
        if self._yindex is None:
            self._yindex = {v: i for i, v in enumerate(self.gy)}
        return self._yindex

    def cell_box(self, x: int, y: int) -> tuple[int, int, int, int]:
        return self.gx[x], self.gy[y], self.gx[x + 1], self.gy[y + 1]

    def box_to_cells(self, xl: int, yl: int, xh: int, yh: int) -> tuple[int, int, int, int] | None:
        """Map a CUGR2 guide box in DBU back to inclusive G-cell indices (x0,y0,x1,y1).

        write() emits getGridline(0, x.low), getGridline(1, y.low),
        getGridline(0, x.high+1), getGridline(1, y.high+1).
        Returns None if any coordinate is not an exact gridline.
        """
        ix, iy = self.xindex(), self.yindex()
        if xl not in ix or xh not in ix or yl not in iy or yh not in iy:
            return None
        x0, x1 = ix[xl], ix[xh] - 1
        y0, y1 = iy[yl], iy[yh] - 1
        if x0 > x1 or y0 > y1 or x0 < 0 or y0 < 0:
            return None
        return x0, y0, x1, y1

    def cells_to_box(self, x0: int, y0: int, x1: int, y1: int) -> tuple[int, int, int, int]:
        return self.gx[x0], self.gy[y0], self.gx[x1 + 1], self.gy[y1 + 1]

    def edge_length(self, direction: int, edge_index: int) -> int:
        return self.edge_len_h[edge_index] if direction == 0 else self.edge_len_v[edge_index]


def load_grid_geom(
    run_dir: str | Path,
    die_xl: int = 0,
    die_yl: int = 0,
    die_xh: int = 1857000,
    die_yh: int = 1840000,
    step: int = 3000,
) -> GridGeom:
    """Build gridlines from the implicit 3000-DBU rule and edge_length.txt / layout.txt.

    die_* defaults are ispd18_test5's logged die region. Callers for other
    benches must pass the die from that run's route.log.
    """
    run_dir = Path(run_dir)
    layout = (run_dir / "layout.txt").read_text().splitlines()[0].split()
    nx, ny, m2 = int(layout[0]), int(layout[1]), int(layout[2])
    gx = implicit_gridlines(die_xl, die_xh, step)
    gy = implicit_gridlines(die_yl, die_yh, step)
    if len(gx) - 1 != nx or len(gy) - 1 != ny:
        raise ValueError(
            f"implicit gridlines give {len(gx)-1}x{len(gy)-1} cells, layout.txt has {nx}x{ny}. "
            f"Check die region / step."
        )
    elines = (run_dir / "edge_length.txt").read_text().splitlines()
    eh = list(map(int, elines[0].split()))
    ev = list(map(int, elines[1].split()))
    if len(eh) != nx - 1 or len(ev) != ny - 1:
        raise ValueError(
            f"edge_length.txt has {len(eh)}/{len(ev)} entries, expected {nx-1}/{ny-1}"
        )
    return GridGeom(
        gx=gx,
        gy=gy,
        edge_len_h=eh,
        edge_len_v=ev,
        m2_pitch=m2,
        die_xl=die_xl,
        die_yl=die_yl,
        die_xh=die_xh,
        die_yh=die_yh,
    )


def wire_edges_of_box(
    layer: int, x0: int, y0: int, x1: int, y1: int
) -> list[tuple[int, int, int]]:
    """G-cell wire edges inside a 1-D (row or column) guide box on `layer`.

    A horizontal box (y0==y1, x0<x1) contributes H-edges (L,x,y0) for x in [x0,x1).
    A vertical box (x0==x1, y0<y1) contributes V-edges (L,x0,y) for y in [y0,y1).
    A 1-cell box contributes no wire edges. A 2-D box is not a basic wire guide
    (those are pin-access patches) and is rejected here.
    """
    if y0 == y1 and x0 < x1:
        return [(layer, x, y0) for x in range(x0, x1)]
    if x0 == x1 and y0 < y1:
        return [(layer, x0, y) for y in range(y0, y1)]
    return []


def box_kind(x0: int, y0: int, x1: int, y1: int) -> str:
    dx, dy = x1 - x0, y1 - y0
    if dx == 0 and dy == 0:
        return "point"
    if dx == 0 or dy == 0:
        return "wire"
    return "patch2d"


def merge_collinear_edges(
    edges: list[tuple[int, int, int]], layer_direction: dict[int, int] | None = None
) -> list[tuple[int, int, int, int, int]]:
    """Merge same-layer collinear unit edges into maximal 1-D boxes.

    Returns list of (layer, x0, y0, x1, y1) inclusive cell ranges.
    """
    dirs = layer_direction or LAYER_DIRECTION
    by_layer: dict[int, list[tuple[int, int, int]]] = {}
    for e in edges:
        by_layer.setdefault(e[0], []).append(e)
    boxes: list[tuple[int, int, int, int, int]] = []
    for L, es in by_layer.items():
        d = dirs[L]
        used: set[tuple[int, int, int]] = set()
        eset = set(es)
        for e in sorted(es):
            if e in used:
                continue
            _, x, y = e
            if d == 0:
                x0 = x
                while (L, x0 - 1, y) in eset and (L, x0 - 1, y) not in used:
                    x0 -= 1
                x1 = x
                while (L, x1, y) in eset:
                    used.add((L, x1, y))
                    x1 += 1
                # edges (L,x0,y)..(L,x1-1,y) cover cells x0..x1 at y
                boxes.append((L, x0, y, x1, y))
            else:
                y0 = y
                while (L, x, y0 - 1) in eset and (L, x, y0 - 1) not in used:
                    y0 -= 1
                y1 = y
                while (L, x, y1) in eset:
                    used.add((L, x, y1))
                    y1 += 1
                boxes.append((L, x, y0, x, y1))
    return boxes


def official_gridlines_from_gcellgrid(
    statements: list[tuple[str, int, int, int]], die_low: tuple[int, int], die_high: tuple[int, int]
) -> tuple[list[int], list[int]]:
    """Rebuild CUGR2 gridlines from DEF GCELLGRID statements (Design.cpp).

    Each statement is (axis, location, num_tracks, step) as in
    `GCELLGRID X loc DO num STEP step`. axis is 'X' or 'Y'.
    CUGR2 pushes location+step*i for i=1..num_tracks-1, then inserts die
    low/high and sorts. Do not invent a synthetic grid.
    """
    acc: dict[str, list[int]] = {"X": [], "Y": []}
    for axis, location, num, step in statements:
        axis = axis.upper()
        for i in range(1, num):
            acc[axis].append(location + step * i)
    gx = [die_low[0], *acc["X"]]
    gy = [die_low[1], *acc["Y"]]
    gx = sorted(set(gx))
    gy = sorted(set(gy))
    if gx[-1] != die_high[0]:
        gx.append(die_high[0])
    if gy[-1] != die_high[1]:
        gy.append(die_high[1])
    return gx, gy


def parse_gcellgrid_def(def_path: str | Path) -> list[tuple[str, int, int, int]]:
    """Parse GCELLGRID lines from a DEF (stream; stops after the DESIGN header)."""
    out: list[tuple[str, int, int, int]] = []
    with Path(def_path).open() as f:
        for raw in f:
            s = raw.strip()
            if not s.startswith("GCELLGRID"):
                # GCELLGRID is in the header; stop once COMPONENTS/NETS start
                if s.startswith("COMPONENTS") or s.startswith("NETS") or s.startswith("PINS"):
                    if out:
                        break
                continue
            # GCELLGRID X 100 DO 1804 STEP 2000 ;
            parts = s.replace(";", "").split()
            # ['GCELLGRID', 'X', loc, 'DO', num, 'STEP', step]
            axis = parts[1]
            loc = int(parts[2])
            num = int(parts[4])
            step = int(parts[6])
            out.append((axis, loc, num, step))
    return out


def load_grid_geom_from_def(
    run_dir: str | Path,
    def_path: str | Path,
    die_xl: int,
    die_yl: int,
    die_xh: int,
    die_yh: int,
) -> GridGeom:
    run_dir = Path(run_dir)
    layout = (run_dir / "layout.txt").read_text().splitlines()[0].split()
    nx, ny, m2 = int(layout[0]), int(layout[1]), int(layout[2])
    stmts = parse_gcellgrid_def(def_path)
    if not stmts:
        raise ValueError(f"no GCELLGRID in {def_path}")
    gx, gy = official_gridlines_from_gcellgrid(stmts, (die_xl, die_yl), (die_xh, die_yh))
    if len(gx) - 1 != nx or len(gy) - 1 != ny:
        raise ValueError(
            f"GCELLGRID gridlines give {len(gx)-1}x{len(gy)-1} cells, layout.txt has {nx}x{ny}"
        )
    elines = (run_dir / "edge_length.txt").read_text().splitlines()
    eh = list(map(int, elines[0].split()))
    ev = list(map(int, elines[1].split()))
    if len(eh) != nx - 1 or len(ev) != ny - 1:
        raise ValueError(
            f"edge_length.txt has {len(eh)}/{len(ev)} entries, expected {nx-1}/{ny-1}"
        )
    # verify first/last edge lengths against centers
    def elen(lines, i):
        c0 = (lines[i] + lines[i + 1]) // 2
        c1 = (lines[i + 1] + lines[i + 2]) // 2
        return c1 - c0
    if elen(gx, 0) != eh[0] or elen(gx, nx - 2) != eh[-1]:
        raise ValueError(f"X edge_length mismatch: file {eh[0]}..{eh[-1]} vs centers {elen(gx,0)}..{elen(gx,nx-2)}")
    if elen(gy, 0) != ev[0] or elen(gy, ny - 2) != ev[-1]:
        raise ValueError(f"Y edge_length mismatch: file {ev[0]}..{ev[-1]} vs centers {elen(gy,0)}..{elen(gy,ny-2)}")
    return GridGeom(
        gx=gx, gy=gy, edge_len_h=eh, edge_len_v=ev, m2_pitch=m2,
        die_xl=die_xl, die_yl=die_yl, die_xh=die_xh, die_yh=die_yh,
    )
