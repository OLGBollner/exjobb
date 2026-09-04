#!/usr/bin/env python3
"""
Master Analysis Script for Defect Spin-Phonon Dynamics.

Combines the full chain into one automated, non-destructive workflow:
  1. Ingests raw simulation data (VASP OUTCARs) or pre-parsed .npz files.
  2. Calculates ZFS derivatives and spin-phonon couplings (V_00, V_pm, V_0pm).
  3. Calculates 1-phonon, Raman, and 2-phonon transition rates vs temperature.
  4. Solves population relaxation dynamics and extracts T_1 times vs temperature.
  5. Saves all results and metadata bunched together in a dedicated unique run folder.
  6. (Optional) Generates publication-quality figures directly in the run folder.

Usage examples:
  # From VASP simulation folder:
  python scripts/run_analysis.py --sim_folder ../../ClV_128/first_order/pert_0.025 --all

  # From pre-parsed raw ZFS .npz:
  python scripts/run_analysis.py --raw_zfs_file data/NV_64_raw_zfs.npz --phonon_file data/phonon.npz

  # From pre-computed spin-phonon coupling:
  python scripts/run_analysis.py --coupling_file derivatives/NV_64_zfs_coefficients.npz
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from beyblade.pipeline import run_full_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end spin-phonon relaxation analysis from simulation to T_1 times."
    )

    # Input modes
    parser.add_argument("--sim_folder", type=str, nargs="+", help="Path to VASP simulation folder.")
    parser.add_argument("--raw_zfs_file", type=str, nargs="+", help="Path to raw ZFS .npz dataset file(s).")
    parser.add_argument(
        "--raw_zfs_file_1d",
        type=str,
        help="Path to raw ZFS 1st-order .npz dataset (combined 1d+2d mode).",
    )
    parser.add_argument(
        "--raw_zfs_file_2d",
        type=str,
        help="Path to raw ZFS 2nd-order .npz dataset (combined 1d+2d mode).",
    )
    parser.add_argument("--coupling_file", type=str, help="Path to pre-computed spin-phonon coupling .npz file.")
    parser.add_argument("-ph", "--phonon_file", type=str, help="Path to phonopy.yaml or phonon_data.npz.")
    parser.add_argument("--two_phonon", type=str, help="Path to two-phonon Raman .npz file (optional).")

    # Calculation method / options
    parser.add_argument("--all", action="store_true", help="Use all_bands method (ZFS_hyp).")
    parser.add_argument("--approx", action="store_true", help="Use defect_band_approx method (ZFS_occup).")
    parser.add_argument("--order", type=int, choices=[1, 2], help="Perturbation order (1 or 2).")
    parser.add_argument("--pert_scale", type=float, help="Override perturbation scale (e.g. 0.025).")
    parser.add_argument("--defect", type=str, help="Override defect name (e.g. NV, ClV).")
    parser.add_argument("--cell_size", type=int, help="Override supercell size (e.g. 64, 128).")

    # Temperature parameters
    parser.add_argument("--t_start", type=float, default=0.0, help="Start temperature in K (default: 0.0).")
    parser.add_argument("--t_end", type=float, default=300.0, help="End temperature in K (default: 300.0).")
    parser.add_argument("--t_step", type=float, default=10.0, help="Temperature step in K (default: 10.0).")

    # Relaxation dynamics parameters
    parser.add_argument(
        "--init_state",
        type=str,
        default="ms_0",
        choices=["ms_0", "ms_1", "ms_-1"],
        help="Initial spin state for population relaxation (default: ms_0).",
    )

    # Output run configuration
    parser.add_argument(
        "-o",
        "--output_root",
        type=str,
        default="runs",
        help="Root folder for output run directories (default: runs).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Custom run folder name. If it exists, an increment index is appended.",
    )

    # Plotting & debugging
    parser.add_argument("--plot", action="store_true", help="Generate plots inside <run_dir>/figures/.")
    parser.add_argument("-d", "--debug", action="store_true", help="Print debug details during derivative calculations.")

    args = parser.parse_args()

    calc_method = "defect_band_approx" if args.approx else "all_bands"
    zfs_folder = "ZFS_occup" if args.approx else "ZFS_hyp"

    # Normalize raw_zfs_file if single argument
    raw_files = args.raw_zfs_file
    if raw_files and len(raw_files) == 1:
        raw_files = raw_files[0]

    res = run_full_pipeline(
        sim_folder=args.sim_folder,
        raw_zfs_file=raw_files,
        raw_zfs_file_1d=args.raw_zfs_file_1d,
        raw_zfs_file_2d=args.raw_zfs_file_2d,
        coupling_file=args.coupling_file,
        phonon_file=args.phonon_file,
        two_phonon_file=args.two_phonon,
        calc_method=calc_method,
        zfs_folder=zfs_folder,
        order=args.order,
        pert_scale=args.pert_scale,
        defect=args.defect,
        cell_size=args.cell_size,
        t_start=args.t_start,
        t_end=args.t_end,
        t_step=args.t_step,
        init_state=args.init_state,
        output_root=args.output_root,
        run_name=args.run_name,
        save_plots=args.plot,
        show_plots=False,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
