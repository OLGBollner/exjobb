#!/usr/bin/env python
"""Regenerate the frozen T1 / transition-rate reference fixtures.

These fixtures are the value-level guard behind ``tests/test_t1_regression.py``:
they pin what the pipeline computes today for NV_64, NV_512 and ClV_128 so that a
later change which moves a published number has to move a tracked file too.

One command regenerates all of them::

    python scripts/regenerate_t1_reference.py

The script is deliberately free of timestamps and host details, so on the same
interpreter and library versions a behaviour-preserving change leaves the
fixtures byte-identical::

    python scripts/regenerate_t1_reference.py && git diff --exit-code tests/reference

A non-empty diff means the pipeline's numbers moved. That is not automatically
wrong, but it must be explained and re-blessed deliberately -- update
``tests/reference/README.md`` with the new commit and the reason in the same PR.

Note that this byte-for-byte check only holds within one interpreter. Across
numpy/scipy versions the last couple of digits move (~1e-15 relative), which is
why ``tests/test_t1_regression.py`` compares with a tolerance rather than by
hash. Regenerate on the interpreter recorded in ``tests/reference/README.md`` if
you want a clean diff.

Requires the Git LFS payloads: run ``git lfs pull`` first.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = REPO_ROOT / "tests" / "reference"

# Fixed temperature grid, in kelvin. Chosen to straddle every regime the pipeline
# shows: the low-T corner where NV_64 underflows to an infinite T1, the steep
# crossover around 30-100 K, and room temperature.
TEMPERATURE_GRID_K = (1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300)

# Systems to freeze. Each entry maps to one JSON fixture in tests/reference/.
# Paths are relative to the repository root and are all Git LFS payloads.
SYSTEMS = (
    {
        "system": "NV_64",
        "raw_1d": "NV_64/NV_64_raw_zfs_data_all_bands_1d.npz",
        "raw_2d": "NV_64/NV_64_raw_zfs_data_all_bands_2d.npz",
        "phonons": "NV_64/phonon_data.npz",
        "method": "all",
    },
    {
        "system": "NV_512",
        "raw_1d": "NV_512/NV_512_raw_zfs_data_all_bands_1d.npz",
        "raw_2d": "NV_512/NV_512_raw_zfs_data_all_bands_2d.npz",
        "phonons": "NV_512/phonon_data.npz",
        "method": "all",
    },
    {
        "system": "ClV_128",
        "raw_1d": "ClV_128/ClV_128_raw_zfs_data_all_bands_1d.npz",
        "raw_2d": "ClV_128/ClV_128_raw_zfs_data_all_bands_2d.npz",
        "phonons": "ClV_128/phonon_data.npz",
        "method": "all",
    },
)

RATE_ORDERS = ("first_order", "second_order", "two_phonon")
RATE_TRANSITIONS = ("0_1", "1_-1")

INIT_STATE = "ms_0"


def build_argv(spec: dict) -> list[str]:
    """Return the ``spinlabs`` argv that produces this system's reference run.

    The output location is intentionally omitted: the caller appends
    ``-o <dir> --run-name reference`` so the same argv can be replayed into a
    temporary directory by the regression test.
    """
    return [
        "run",
        "--raw-zfs-file-1d",
        spec["raw_1d"],
        "--raw-zfs-file-2d",
        spec["raw_2d"],
        "-ph",
        spec["phonons"],
        "--method",
        spec["method"],
        "--init-state",
        INIT_STATE,
        "--temperatures",
        *[str(t) for t in TEMPERATURE_GRID_K],
    ]


def floats(values) -> list[str]:
    """Serialise floats as exact round-trip strings.

    ``repr`` of a Python float round-trips exactly, reads cleanly in a diff, and
    -- unlike a bare JSON number -- represents ``inf`` without stepping outside
    standard JSON. The pipeline genuinely returns ``inf`` for NV_64 below 30 K,
    so that case has to survive the round trip rather than be smoothed away.
    """
    return [repr(float(v)) for v in np.asarray(values, dtype=float)]


def run_reference(spec: dict, output_root: Path) -> dict:
    """Run the pipeline for one system and collect the numbers worth freezing."""
    from spinlabs.cli import main

    argv = build_argv(spec)
    main([*argv, "-o", str(output_root), "--run-name", "reference"])

    run_dir = output_root / "reference"
    t1 = np.load(run_dir / "t1_relaxation.npz", allow_pickle=True)
    rates = np.load(run_dir / "transition_rates.npz", allow_pickle=True)

    rate_block = {}
    for order in RATE_ORDERS:
        channel = rates[order].item()
        rate_block[order] = {t: floats(channel[t]) for t in RATE_TRANSITIONS}

    return {
        "system": spec["system"],
        "defect": str(t1["defect"]),
        "cell_size": int(t1["cell_size"]),
        "calc_method": str(t1["calc_method"]),
        "order": int(t1["order"]),
        "pert_scale": repr(float(rates["pert_scale"])),
        "init_state": str(t1["init_state"]),
        "invocation": {
            "command": "spinlabs",
            "argv": argv,
            "cwd": "repository root",
            "output_args_appended_by_harness": ["-o", "<run dir>", "--run-name", "reference"],
        },
        "temperatures_K": floats(t1["temperatures"]),
        "t1_fit_s": floats(t1["t1_fit"]),
        "t1_eigenval_s": floats(t1["t1_eigenval"]),
        "transition_rates_per_s": rate_block,
    }


def fixture_path(system: str) -> Path:
    return REFERENCE_DIR / f"{system}.json"


def write_fixture(payload: dict) -> Path:
    path = fixture_path(payload["system"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--system",
        action="append",
        choices=[s["system"] for s in SYSTEMS],
        help="Regenerate only this system (repeatable). Default: all three.",
    )
    args = parser.parse_args(argv)

    selected = [s for s in SYSTEMS if args.system is None or s["system"] in args.system]

    for spec in selected:
        for key in ("raw_1d", "raw_2d", "phonons"):
            path = REPO_ROOT / spec[key]
            if not path.exists():
                parser.error(f"missing input {path} -- run 'git lfs pull'")

        print(f"== {spec['system']}: running pipeline over {len(TEMPERATURE_GRID_K)} temperatures ...")
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_reference(spec, Path(tmp))
        written = write_fixture(payload)
        print(f"== {spec['system']}: wrote {written.relative_to(REPO_ROOT)}\n")

    print("Done. Review 'git diff tests/reference' -- any change is a change to published numbers.")
    return 0


if __name__ == "__main__":
    # Run from the repository root so the relative input paths in SYSTEMS resolve.
    import os

    os.chdir(REPO_ROOT)
    sys.exit(main())
