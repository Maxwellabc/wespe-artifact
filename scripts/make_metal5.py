#!/usr/bin/env python3
"""Truncate an ISPD 2018/2019 LEF/DEF to the first N routing layers (default 5).

This reconstructs the congested '*_metal5' variants used by CUGR2 / DGR /
ICCAD'19 Problem C. The official ICCAD'19 tarball was not reachable
(404 / 500 as of 2026-11-12). We generate LEF/DEF locally from the public
ISPD contest files.

What is done:
  1. Drop top-level ROUTING layers above Metal{N} and CUT layers that only
     connect a dropped routing layer (Via5..Via8 when N=5).
  2. Drop VIA / VIARULE blocks that reference a dropped layer.
  3. Inside MACRO PIN/OBS, remap leftover Metal{N+1}+ geometry to Metal{N}
     (needed because many ISPD IO / macro pins sit on Metal7). Drop geometry
     on dropped CUT layers.
  4. In the DEF, drop TRACKS on dropped layers, and remap LAYER Metal{N+1}+
     (typically IO pins) to Metal{N}.

This is a reconstruction, not a byte-identical copy of the official ICCAD
metal5 files. After CUGR2 runs, compare G-cell grid and net/cell counts to
DGR Table 2 before treating numbers as official-bench results.

The first version of this script (same day) only truncated top-level LAYER/VIA
blocks. That was enough for ispd18_sample and ispd18_test1 (no Metal6+ inside
MACROs / no IO on Metal7), but ispd18_test5 segfaulted in Rsyn while parsing
because MACRO PIN ports and DEF PINS still named Metal6/Metal7. This version
remaps those shapes.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


ROUTING_RE = re.compile(r"^LAYER\s+(\S+)\s*$", re.I)
END_RE = re.compile(r"^END\s+(\S+)\s*$", re.I)
TYPE_RE = re.compile(r"^\s*TYPE\s+(\S+)\s*;", re.I)
LAYER_REF_RE = re.compile(r"\bLAYER\s+(\S+)\s*;", re.I)
LAYER_ANY_RE = re.compile(r"^(\s*)LAYER\s+(\S+)(\s*;?.*)$", re.I)


def split_lef_blocks(text: str) -> list[tuple[str, str, str]]:
    lines = text.splitlines(keepends=True)
    blocks: list[tuple[str, str, str]] = []
    i = 0
    header: list[str] = []
    start_named = re.compile(r"^(LAYER|VIA|VIARULE|SITE|MACRO|BEGINEXT)\s+(\S+)", re.I)
    start_bare = re.compile(r"^(UNITS|PROPERTYDEFINITIONS)\s*$", re.I)
    while i < len(lines):
        m_named = start_named.match(lines[i])
        m_bare = start_bare.match(lines[i])
        if not m_named and not m_bare:
            header.append(lines[i])
            i += 1
            continue
        if header:
            blocks.append(("HEADER", "", "".join(header)))
            header = []
        if m_bare:
            kind = m_bare.group(1).upper()
            name = kind
        else:
            kind = m_named.group(1).upper()
            name = m_named.group(2).strip()
        body = [lines[i]]
        i += 1
        while i < len(lines):
            body.append(lines[i])
            em = END_RE.match(lines[i])
            if em and em.group(1).upper() == name.upper():
                i += 1
                break
            i += 1
        else:
            raise RuntimeError(f"Unterminated LEF block {kind} {name}")
        blocks.append((kind, name, "".join(body)))
    if header:
        blocks.append(("HEADER", "", "".join(header)))
    return blocks


def layer_index(name: str) -> int | None:
    m = re.match(r"^(Metal|metal|M)(\d+)$", name)
    if m:
        return int(m.group(2))
    m = re.match(r"^(Via|via|V)(\d+)$", name)
    if m:
        return int(m.group(2))
    return None


def classify_layer_block(body: str) -> str:
    for line in body.splitlines():
        m = TYPE_RE.match(line)
        if m:
            return m.group(1).upper()
    return "UNKNOWN"


def keep_layer(name: str, typ: str, max_routing: int) -> bool:
    idx = layer_index(name)
    if typ == "ROUTING":
        return idx is None or idx <= max_routing
    if typ == "CUT":
        return idx is None or idx < max_routing
    return True


def referenced_layers(body: str) -> set[str]:
    return set(LAYER_REF_RE.findall(body))


def remap_embedded_layers(body: str, dropped: set[str], top: str) -> str:
    """Inside MACRO (or similar), remap dropped routing layers to top, drop cut geometry."""
    lines = body.splitlines(keepends=True)
    out: list[str] = []
    skipping = False  # skip geometry under a dropped CUT layer
    for line in lines:
        m = LAYER_ANY_RE.match(line.rstrip("\n"))
        if m:
            indent, name, rest = m.group(1), m.group(2), m.group(3)
            if name in dropped:
                typ_guess = "CUT" if re.match(r"^(Via|via|V)\d+$", name) else "ROUTING"
                if typ_guess == "CUT":
                    skipping = True
                    continue
                # remap routing layer to MetalN
                skipping = False
                nl = "\n" if line.endswith("\n") else ""
                # preserve a trailing ' ;' if present in rest
                if not rest.strip():
                    rest = " ;"
                elif not rest.strip().endswith(";"):
                    rest = rest.rstrip() + " ;"
                out.append(f"{indent}LAYER {top}{rest}{nl}")
                continue
            skipping = False
            out.append(line)
            continue
        if skipping:
            # geometry lines (RECT/POLYGON/PATH) belong to the dropped cut layer
            if re.match(r"^\s*(RECT|POLYGON|PATH)\b", line, re.I):
                continue
            skipping = False
        out.append(line)
    return "".join(out)


def truncate_lef(lef_text: str, max_routing: int = 5) -> tuple[str, dict]:
    blocks = split_lef_blocks(lef_text)
    dropped_layers: set[str] = set()
    layer_kept: dict[str, bool] = {}
    stats = {"dropped_layers": [], "dropped_vias": [], "kept_routing": [], "remapped_macro_lines": 0}

    for kind, name, body in blocks:
        if kind == "LAYER":
            typ = classify_layer_block(body)
            keep = keep_layer(name, typ, max_routing)
            layer_kept[name] = keep
            if not keep:
                dropped_layers.add(name)
                stats["dropped_layers"].append(f"{name} ({typ})")
            elif typ == "ROUTING":
                stats["kept_routing"].append(name)

    top = f"Metal{max_routing}"
    out: list[str] = []
    for kind, name, body in blocks:
        if kind == "LAYER":
            if layer_kept.get(name, True):
                out.append(body)
            continue
        if kind in {"VIA", "VIARULE"}:
            refs = referenced_layers(body)
            if any(r in dropped_layers for r in refs) or name in dropped_layers:
                stats["dropped_vias"].append(name)
                continue
        if kind == "MACRO":
            new_body = remap_embedded_layers(body, dropped_layers, top)
            if new_body != body:
                stats["remapped_macro_lines"] += 1
            out.append(new_body)
            continue
        out.append(body)
    return "".join(out), stats



DEFAULT_GCELL_SPACING = 3000  # matches CUGR2 Design::defaultGridlineSpacing


def gcell_do(high: int, spacing: int = DEFAULT_GCELL_SPACING, low: int = 0) -> int:
    """DEF GCELLGRID 'DO' (number of grid *lines*) so CUGR2 reproduces Table 2.

    CUGR2, when the DEF has no GCELLGRID, uses
        for (i = low; i + spacing < high; i += spacing) push i;
        if (back != high) push high;
    When (high-low) is an exact multiple of spacing this merges the last two
    steps into one double-width G-cell (ispd18_test5 X: 1857000/3000=619, but
    implicit grid is 618). DGR Table 2 lists 619×613, which is the split
    (inclusive) count. We emit an explicit GCELLGRID:
      * exact multiple: lines 0, S, ..., high  (N+1 lines, N cells)
      * otherwise: same interior lines as CUGR2 implicit, plus die high.
    CUGR2 then adds die low/high itself and iterates i=1..numStep-1, so DO
    must satisfy (DO-1)*S = last interior line.
    """
    if (high - low) % spacing == 0:
        return (high - low) // spacing + 1
    last_interior = low + spacing * ((high - low - spacing - 1) // spacing)
    return (last_interior - low) // spacing + 1


def insert_gcellgrid_UNUSED_DO_NOT_CALL(def_text: str, die: tuple[int, int, int, int] | None = None,
                     spacing: int = DEFAULT_GCELL_SPACING) -> tuple[str, dict]:
    import re
    if die is None:
        m = re.search(r"DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)", def_text)
        if not m:
            return def_text, {"inserted": False, "reason": "no DIEAREA"}
        xlo, ylo, xhi, yhi = map(int, m.groups())
    else:
        xlo, ylo, xhi, yhi = die
    do_x = gcell_do(xhi, spacing, xlo)
    do_y = gcell_do(yhi, spacing, ylo)
    # drop any existing GCELLGRID (public ISPD DEFs typically have none)
    lines = [ln for ln in def_text.splitlines(keepends=True) if not re.match(r"\s*GCELLGRID\b", ln)]
    block = (f"GCELLGRID X {xlo} DO {do_x} STEP {spacing} ;\n"
             f"GCELLGRID Y {ylo} DO {do_y} STEP {spacing} ;\n")
    out: list[str] = []
    inserted = False
    last_tracks = max((i for i, ln in enumerate(lines) if ln.lstrip().startswith("TRACKS")), default=-1)
    for i, ln in enumerate(lines):
        out.append(ln)
        if i == last_tracks:
            out.append(block)
            inserted = True
    if not inserted:
        out.insert(0, block)
        inserted = True
    cells_x, cells_y = do_x - 1, do_y - 1  # plus CUGR2 may append die high as an extra line
    # CUGR2 adds die high if not already the last line. For exact multiples,
    # last interior is high itself (DO-1)*S = high, so no extra line, cells = DO-1.
    # For leftover, last interior < high, CUGR2 appends high, cells = DO-1+1-1 = DO-1.
    # Wait: lines = {0, S, ..., (DO-1)*S} from GCELLGRID loop i=1..DO-1 plus die low
    # plus maybe die high. cells = nlines-1.
    # exact: (DO-1)*S = high, nlines = DO, cells = DO-1. Good.
    # leftover: (DO-1)*S = last_interior, plus high, nlines = DO+1, cells = DO.
    #
    # CORRECT leftover DO: we want cells = implicit = last_interior/S + 1
    # nlines = (last_interior/S + 1) + 1 (for high) = last_interior/S + 2
    # CUGR2 nlines = 1 (low) + (DO-1) interior starting at S + 1 (high) = DO+1
    # cells = DO. So DO should be last_interior/S + 1, cells = last_interior/S + 1. YES.
    # gcell_do leftover returns last_interior/S + 1, cells reported as DO-1 would be WRONG
    # for leftover! Report expected cells properly below.
    exp_x = do_x - 1 if (xhi - xlo) % spacing == 0 else do_x
    exp_y = do_y - 1 if (yhi - ylo) % spacing == 0 else do_y
    # leftover: cells = DO. exact: cells = DO-1.
    return "".join(out), {"inserted": True, "do_x": do_x, "do_y": do_y, "expected_cells": (exp_x, exp_y)}


def truncate_def(def_text: str, max_routing: int = 5) -> tuple[str, dict]:
    """Drop TRACKS, VIA defs, and SPECIALNET stripes on layers above Metal{N}.

    Do NOT invent a GCELLGRID. ispd19 public DEFs already have one (and it
    matches DGR Table 2). ispd18 public DEFs typically have none; CUGR2 then
    builds an implicit 3000-DBU grid, which for ispd18_test5 is 618 x 613,
    one column short of Table 2's 619. That is a reconstruction fact, not
    something we should paper over by inserting a synthetic grid.
    """
    top = f"Metal{max_routing}"
    dropped_metal = {f"Metal{i}" for i in range(max_routing + 1, 20)}
    dropped_cut = {f"Via{i}" for i in range(max_routing, 20)}
    dropped = dropped_metal | dropped_cut
    stats = {
        "dropped_tracks": 0,
        "remapped_layer": 0,
        "dropped_via_defs": 0,
        "dropped_special_stripes": 0,
    }
    lines = def_text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    via_kept = 0

    while i < n:
        line = lines[i]
        raw = line.lstrip()

        tm = re.search(r"\bLAYER\s+(\S+)\s*;", line)
        if raw.startswith("TRACKS") and tm and tm.group(1) in dropped_metal:
            stats["dropped_tracks"] += 1
            i += 1
            continue

        if re.match(r"^VIAS\s+\d+\s*;", line):
            header_idx = len(out)
            out.append(line)  # rewritten after the block
            i += 1
            via_kept = 0
            while i < n and not re.match(r"^END\s+VIAS", lines[i]):
                if re.match(r"^-\s+\S+", lines[i]):
                    chunk = [lines[i]]
                    i += 1
                    while (
                        i < n
                        and not re.match(r"^-\s+\S+", lines[i])
                        and not re.match(r"^END\s+VIAS", lines[i])
                    ):
                        chunk.append(lines[i])
                        i += 1
                    body = "".join(chunk)
                    refs = set(re.findall(r"\b(Metal\d+|Via\d+)\b", body))
                    if refs & dropped:
                        stats["dropped_via_defs"] += 1
                        continue
                    out.extend(chunk)
                    via_kept += 1
                else:
                    out.append(lines[i])
                    i += 1
            out[header_idx] = f"VIAS {via_kept} ;\n"
            if i < n:
                out.append(lines[i])  # END VIAS
                i += 1
            continue

        vm = re.search(r"\bNEW\s+(Metal\d+|Via\d+)\b", line)
        if vm and vm.group(1) in dropped:
            stats["dropped_special_stripes"] += 1
            i += 1
            continue

        def _sub(m, _dm=dropped_metal, _st=stats):
            name = m.group(1)
            if name in _dm:
                _st["remapped_layer"] += 1
                return m.group(0).replace(name, top)
            return m.group(0)

        out.append(re.sub(r"\bLAYER\s+(Metal\d+)\b", _sub, line))
        i += 1
    return "".join(out), stats


def convert_design(src_dir: Path, dst_dir: Path, src_name: str, dst_name: str, max_routing: int = 5) -> dict:
    dst_dir.mkdir(parents=True, exist_ok=True)
    src_lef = src_dir / f"{src_name}.input.lef"
    src_def = src_dir / f"{src_name}.input.def"
    dst_lef = dst_dir / f"{dst_name}.input.lef"
    dst_def = dst_dir / f"{dst_name}.input.def"
    lef_text, lef_stats = truncate_lef(src_lef.read_text(errors="replace"), max_routing=max_routing)
    def_text, def_stats = truncate_def(src_def.read_text(errors="replace"), max_routing=max_routing)
    # sanity: no dropped routing layer names remain as LAYER statements
    leftover = re.findall(r"\b(Metal[6-9]|Via[5-8])\b", lef_text + def_text)
    dst_lef.write_text(lef_text)
    dst_def.write_text(def_text)
    stats = {**lef_stats, **{f"def_{k}": v for k, v in def_stats.items()}}
    stats["leftover_layer_mentions"] = leftover
    stats["src_lef"] = str(src_lef)
    stats["dst_lef"] = str(dst_lef)
    stats["dst_def"] = str(dst_def)
    stats["src_bytes"] = src_lef.stat().st_size
    stats["dst_lef_bytes"] = dst_lef.stat().st_size
    stats["dst_def_bytes"] = dst_def.stat().st_size
    return stats


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extracted", type=Path, default=Path("/workspace/dac2027-3793/benches/extracted"))
    p.add_argument("--out", type=Path, default=Path("/workspace/dac2027-3793/benches/metal5"))
    p.add_argument("--max-routing", type=int, default=5)
    args = p.parse_args()
    cases = [
        ("ispd18_test5", "ispd18_test5_metal5"),
        ("ispd18_test8", "ispd18_test8_metal5"),
        ("ispd18_test10", "ispd18_test10_metal5"),
        ("ispd19_test7", "ispd19_test7_metal5"),
        ("ispd19_test8", "ispd19_test8_metal5"),
        ("ispd19_test9", "ispd19_test9_metal5"),
        ("ispd18_sample", "ispd18_sample_metal5"),
        ("ispd18_test1", "ispd18_test1_metal5"),
        ("ispd19_sample4", "ispd19_sample4_metal5"),
    ]
    for src, dst in cases:
        src_dir = args.extracted / src
        if not (src_dir / f"{src}.input.lef").exists():
            print(f"SKIP {src}: LEF not found in {src_dir}")
            continue
        stats = convert_design(src_dir, args.out / dst, src, dst, args.max_routing)
        print(
            f"OK {dst}: kept {stats['kept_routing']}; dropped {stats['dropped_layers']}; "
            f"vias {len(stats['dropped_vias'])}; macros_touched {stats['remapped_macro_lines']}; "
            f"def_tracks {stats['def_dropped_tracks']}; def_remap {stats['def_remapped_layer']}; "
            f"leftover {len(stats['leftover_layer_mentions'])}; gcell {stats.get('gcell_expected_cells')}"
        )


if __name__ == "__main__":
    main()
