# WESPE (anonymous review copy)

WESPE: Windowed Exact Repair for Global Routing. This repository is the
repair source, official-unit scorer, manuscript PDF, and scripts needed to
rebuild five-layer inputs from the official ICCAD 2019 Problem C v2 package.
It is provided as supplemental material for anonymous review.

## Official contest package (not redistributed)

We do not ship the official ICCAD 2019 Problem C tarball. Please download it
yourself from the contest problems page:

  https://www.iccad-contest.org/2019/problems.html

The Problem C row labeled (iccad19_benchmarks_v2) is the source used in the
paper. As of 13 November 2026 (Asia/Shanghai) the official page linked a Google
Drive copy of `iccad19_benchmarks_v2.tar.gz`. The copy we opened had:

  SHA256  2cecc39a05bb3a9a641ad866dfbf750dcb2e6cdf44fb07d07167715042a3aec7
  size    140569162 bytes

The tarball contains 9-layer LEF/DEF cases (`ispd18_test*` and `ispd19_test*`).
It is not a pre-built metal5 suite: there are no `*_metal5` names in the archive.
Official v2 DEFs carry a `GCELLGRID` that matches the published DGR Table 2
grids. Preserve that grid exactly. Do not insert a synthetic grid.

## Rebuilding five-layer inputs

The paper's Table I and legal write-backs use official v2 DEFs truncated to
Metal1 through Metal5 with these rules:

- Keep the official `GCELLGRID` verbatim, including leftover two-step lines
  (for example the leftover Y line on ispd18 test5).
- Drop routing layers above Metal5 and cut layers that only serve a dropped
  routing layer (Via5 and above).
- Drop VIA / VIARULE definitions that reference a dropped layer.
- Drop special-net stripes that sit on Metal6 or Via5 and above. Do not remap
  those stripes onto Metal5.
- Name the results `ispd18_test5_metal5` and similarly for the other cases, as
  CUGR2 expects.

`scripts/make_metal5.py` is the earlier public-ISPD reconstruction tool. It
remaps leftover MACRO / DEF pin geometry onto Metal5 and, on public DEFs that
have no official grid, may emit a CUGR2-readable grid. That path is the labeled
earlier public-DEF baseline in the paper (test5 618 x 613, ispd19 test7
1035 x 946, ispd19 test8 1805 x 1709). It is not the official-grid Table I
suite. Do not treat a public-DEF run as an official Table 2 result.

## Running the repair and scorer

Python 3.10+ and OR-Tools are required:

    pip install -r requirements.txt

Set `PYTHONPATH` to the directory that contains the `wespe` package.

    # Synthetic 3x3 certificate used for Claim 1 (declared path forest)
    python -m wespe --demo --out demo.json

    # Official-unit score of a CUGR2 route log / guide
    python -m wespe.score --help

    # Windowed repair on an instrumented CUGR2 run directory
    # (layout, capacity3D, usage3D, overflow_edges, window_net_edges, guide)
    python -m wespe --run /path/to/cugr2_run --out repair.json

The driver expects dumps from an instrumented CUGR2 binary. CUGR2 itself is
the published CPU global router; this archive does not include that binary.
The scorer is written to match CUGR2 printed statistics (wirelength, vias with
the documented zerolen correction, overflow edges and units, usage grid).

`scripts/ablation_official_v2.py` enumerates candidate families (incumbent,
maze, bypass, pattern, Steiner / FLUTE) on official-grid windows and records
which family CP-SAT selects. It must write to a new output directory. Do not
overwrite a legal guide.

`scripts/claim1_constructed.py` and `claim1_constructed_3term.py` rebuild the
synthetic 3x3 certificate and the constructed residuals on real test5
coordinates. Those constructed residuals are not a Steiner-necessary
certificate against exhaustive maze.

## Honesty limits

- Wirelength on our official-v2 5-layer CUGR2 runs is higher than published
  DGR Table 2 CUGR2 on every case we ran (2.41% to 14.04%). A lower raw
  overflow than Table 2 is not a Table 2 beat.
- The platform used in the paper had 8 CPU threads, 15 GiB RAM, no GPU, and
  no swap. ispd19 test9 was not run.
- WESPE is exact for the enumerated candidate set, not an optimal global router.
