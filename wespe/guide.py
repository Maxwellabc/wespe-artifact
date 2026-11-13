"""Stream-parse and emit ISPD'18 CUGR2 .guide files."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from .grid_geom import NAME_TO_LAYER, LAYER_NAME, GridGeom, box_kind

GuideBox = tuple[int, int, int, int, int]  # layer, xl, yl, xh, yh in DBU
CellBox = tuple[int, int, int, int, int]  # layer, x0, y0, x1, y1 inclusive G-cells


def iter_guide_nets(path: str | Path) -> Iterator[tuple[str, list[str]]]:
    """Yield (netname, raw body lines including the boxes, excluding name/'('/')')."""
    with Path(path).open() as f:
        while True:
            name = f.readline()
            if not name:
                return
            name = name.strip()
            if not name:
                continue
            opener = f.readline()
            if opener.strip() != "(":
                raise ValueError(f"expected '(' after {name}, got {opener!r}")
            body: list[str] = []
            while True:
                line = f.readline()
                if not line:
                    raise ValueError(f"unterminated net {name}")
                if line.strip() == ")":
                    break
                body.append(line)
            yield name, body


def parse_body_boxes(body: Iterable[str]) -> list[GuideBox]:
    boxes: list[GuideBox] = []
    for raw in body:
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) != 5:
            raise ValueError(f"bad guide line: {s}")
        xl, yl, xh, yh = map(int, parts[:4])
        layer = NAME_TO_LAYER[parts[4]]
        boxes.append((layer, xl, yl, xh, yh))
    return boxes


def format_box(layer: int, xl: int, yl: int, xh: int, yh: int) -> str:
    return f"{xl} {yl} {xh} {yh} {LAYER_NAME[layer]}\n"


def boxes_to_cells(geom: GridGeom, boxes: list[GuideBox]) -> list[CellBox]:
    out: list[CellBox] = []
    for layer, xl, yl, xh, yh in boxes:
        cells = geom.box_to_cells(xl, yl, xh, yh)
        if cells is None:
            raise ValueError(
                f"box {(xl, yl, xh, yh, LAYER_NAME[layer])} is not on implicit gridlines"
            )
        x0, y0, x1, y1 = cells
        out.append((layer, x0, y0, x1, y1))
    return out


def cells_to_boxes(geom: GridGeom, cells: list[CellBox]) -> list[GuideBox]:
    out: list[GuideBox] = []
    for layer, x0, y0, x1, y1 in cells:
        xl, yl, xh, yh = geom.cells_to_box(x0, y0, x1, y1)
        out.append((layer, xl, yl, xh, yh))
    return out


def classify_cell_boxes(cells: list[CellBox]) -> dict[str, list[CellBox]]:
    groups = {"point": [], "wire": [], "patch2d": []}
    for b in cells:
        groups[box_kind(b[1], b[2], b[3], b[4])].append(b)
    return groups


def extract_named_nets(path: str | Path, names: set[str]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for name, body in iter_guide_nets(path):
        if name in names:
            found[name] = body
            if len(found) == len(names):
                break
    missing = names - set(found)
    if missing:
        raise KeyError(f"nets not found in guide: {sorted(missing)}")
    return found


def write_guide_replacing(
    src: str | Path,
    dst: str | Path,
    replacements: dict[str, list[GuideBox]],
) -> dict[str, int]:
    """Stream-copy src to dst, replacing the listed nets' bodies. Other nets untouched."""
    n_written = 0
    n_replaced = 0
    with Path(dst).open("w") as out:
        for name, body in iter_guide_nets(src):
            out.write(name + "\n(\n")
            if name in replacements:
                for layer, xl, yl, xh, yh in replacements[name]:
                    out.write(format_box(layer, xl, yl, xh, yh))
                n_replaced += 1
            else:
                out.writelines(body)
            out.write(")\n")
            n_written += 1
    return {"nets_written": n_written, "nets_replaced": n_replaced}
