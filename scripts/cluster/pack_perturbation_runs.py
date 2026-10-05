#!/usr/bin/env python3
"""pack_perturbation_runs.py -- collect perturbation runs into one .npz.

Scans runs/<mode>_pert_<p>/OUTCAR under the given root, parses the spin-spin
ZFS tensor and final energy from each finished run, and saves everything to a
single compressed .npz for offline analysis (scp it off the cluster).

Usage (from the folder holding runs/):
    python pack_perturbation_runs.py [-r RUN_ROOT] [-o OUT.npz] [--ground-state OUTCAR]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

from beyblade.parsers import parse_outcar_energy, parse_outcar_zfs


def discover_runs(run_root: Path) -> list[tuple[int, float, Path]]:
    """Find runs/<mode>_pert_<p>/OUTCAR; returns sorted (mode, pert, dir)."""
    runs = []
    for outcar in sorted(run_root.glob("runs/*/OUTCAR")):
        m = re.fullmatch(r"(\d+)_pert_(.+)", outcar.parent.name)
        if not m:
            print(f"warning: skipping unrecognised run dir {outcar.parent.name}", file=sys.stderr)
            continue
        try:
            runs.append((int(m.group(1)), float(m.group(2)), outcar))
        except ValueError:
            print(f"warning: skipping unparsable run dir {outcar.parent.name}", file=sys.stderr)
    return sorted(runs, key=lambda r: (r[0], r[1]))


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parse runs/<mode>_pert_<p>/OUTCAR files and pack ZFS tensors + energies into one .npz"
    )
    ap.add_argument("-r", "--run-root", type=Path, default=Path("."), help="folder containing runs/ (default: .)")
    ap.add_argument(
        "-o", "--output", type=Path, default=Path("pert_runs.npz"), help="output .npz path (default: pert_runs.npz)"
    )
    ap.add_argument(
        "--ground-state",
        type=Path,
        default=None,
        metavar="OUTCAR",
        help="ground-state OUTCAR; its ZFS tensor is stored as 'd0' (the Q=0 point) in the npz",
    )
    args = ap.parse_args()

    if not (args.run_root / "runs").is_dir():
        sys.exit(f"Error: no runs/ under {args.run_root}")

    runs = discover_runs(args.run_root)
    if not runs:
        sys.exit(f"Error: no runs/*/OUTCAR found under {args.run_root}")

    modes = sorted({m for m, _, _ in runs})
    perts = sorted({p for _, p, _ in runs})
    tensors = np.full((len(modes), len(perts), 3, 3), np.nan)
    energies = np.full((len(modes), len(perts)), np.nan)
    mode_idx = {m: i for i, m in enumerate(modes)}
    pert_idx = {p: i for i, p in enumerate(perts)}

    n_missing = n_empty = 0
    for mode, pert, outcar in runs:
        tensor = parse_outcar_zfs(outcar)
        if tensor is None:
            print(f"warning: no ZFS tensor in {outcar} (run unfinished?)", file=sys.stderr)
            n_empty += 1
            continue
        i, j = mode_idx[mode], pert_idx[pert]
        tensors[i, j] = tensor.matrix
        energy = parse_outcar_energy(outcar)
        if energy is not None:
            energies[i, j] = energy
        else:
            n_missing += 1

    extra = {}
    if args.ground_state is not None:
        d0 = parse_outcar_zfs(args.ground_state)
        if d0 is None:
            sys.exit(f"Error: no ZFS tensor in ground-state OUTCAR {args.ground_state}")
        extra["d0"] = d0.matrix  # (3, 3) MHz, the Q=0 point
        e0 = parse_outcar_energy(args.ground_state)
        if e0 is not None:
            extra["d0_energy"] = np.float64(e0)
        print(f"ground-state tensor from {args.ground_state}\n{extra['d0']}")

    np.savez_compressed(
        args.output,
        modes=np.array(modes, dtype=int),
        perts=np.array(perts, dtype=float),
        tensors=tensors,  # (n_modes, n_perts, 3, 3), MHz
        energies=energies,  # (n_modes, n_perts), eV
        **extra,
    )
    n_done = tensors.size // 9 - n_empty
    print(
        f"Packed {n_done}/{tensors.size // 9} runs ({n_empty} unfinished, {n_missing} without energy) -> {args.output}"
    )
    print(f"modes: {modes}")
    print(f"perts: {perts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
