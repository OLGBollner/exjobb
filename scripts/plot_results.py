#!/usr/bin/env python3
"""
Dedicated Plotting Script for Defect Spin-Phonon Pipeline Results.

Reads all results stored in a run directory and easily creates publication-quality plots:
  - Coupling coefficients & spectral functions F(w)
  - Transition rates vs temperature (stacked area charts & lines)
  - T_1 relaxation times vs temperature

Usage examples:
  # Plot everything from a run folder:
  python scripts/plot_results.py runs/NV_64_all_bands_1d_20260903_120000/

  # Plot only spin-phonon coupling:
  python scripts/plot_results.py runs/NV_64_all_bands_1d_20260903_120000/ --coupling

  # Plot only T_1 relaxation times:
  python scripts/plot_results.py runs/NV_64_all_bands_1d_20260903_120000/ --t1

  # Compare T_1 across multiple runs:
  python scripts/plot_results.py runs/run_NV_64 runs/run_NV_128 --t1 --show
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from beyblade.models import SpinPhononCouplingData
from beyblade.plotter import (
    plot_1d_spectral_functions,
    plot_t1_relaxation,
    plot_transition_rates_stacked,
)


def plot_run_coupling(run_dir: Path, out_dir: Path, fmt: str, dpi: int, show: bool):
    """Plots 1D and diagonal 2D spin-phonon couplings and spectral functions."""
    coupling_file = run_dir / "spin_phonon_coupling.npz"
    if not coupling_file.exists():
        print(f"Skipping coupling plot: {coupling_file} not found.")
        return

    data = SpinPhononCouplingData.load(coupling_file)
    data_display = data.to_unit("MHz").frequencies_to_unit("meV")

    # 1D plot
    fig1, _, _ = plot_1d_spectral_functions(
        frequencies_mev=data_display.frequencies,
        V_0_0=data_display.V_0_0,
        V_p_m=data_display.V_p_m,
        V_0_pm=data_display.V_0_pm,
        order=1,
    )
    out_file1 = out_dir / f"coupling_spectral_1d.{fmt}"
    fig1.savefig(out_file1, dpi=dpi, bbox_inches="tight")
    print(f"  [✓] Saved 1D coupling plot -> {out_file1}")

    # Backward compatible alias
    out_file_alias = out_dir / f"coupling_spectral.{fmt}"
    fig1.savefig(out_file_alias, dpi=dpi, bbox_inches="tight")

    # 2D plot (diagonal phonons i == j)
    if data_display.V2_0_0 is not None:
        v2_00 = data_display.V2_0_0
        v2_pm = data_display.V2_p_m
        v2_0pm = data_display.V2_0_pm

        v2_00_diag = np.diag(v2_00) if v2_00.ndim == 2 else v2_00
        v2_pm_diag = np.diag(v2_pm) if (v2_pm is not None and v2_pm.ndim == 2) else (v2_pm if v2_pm is not None else np.zeros_like(v2_00_diag))
        v2_0pm_diag = np.diag(v2_0pm) if (v2_0pm is not None and v2_0pm.ndim == 2) else (v2_0pm if v2_0pm is not None else np.zeros_like(v2_00_diag))

        fig2, _, _ = plot_1d_spectral_functions(
            frequencies_mev=data_display.frequencies,
            V_0_0=v2_00_diag,
            V_p_m=v2_pm_diag,
            V_0_pm=v2_0pm_diag,
            order=2,
        )
        out_file2 = out_dir / f"coupling_spectral_2d.{fmt}"
        fig2.savefig(out_file2, dpi=dpi, bbox_inches="tight")
        print(f"  [✓] Saved 2D coupling plot -> {out_file2}")


def plot_run_rates(run_dir: Path, out_dir: Path, fmt: str, dpi: int, show: bool):
    """Plots transition rates vs temperature."""
    rates_file = run_dir / "transition_rates.npz"
    if not rates_file.exists():
        print(f"Skipping transition rates plot: {rates_file} not found.")
        return

    out_base = out_dir / f"transition_rates.{fmt}"
    plot_transition_rates_stacked(rates_file, output_path=out_base)
    print(f"  [✓] Saved transition rate plots in -> {out_dir}")


def plot_run_t1(run_dir: Path, out_dir: Path, fmt: str, dpi: int, show: bool):
    """Plots T_1 relaxation times vs temperature."""
    t1_file = run_dir / "t1_relaxation.npz"
    if not t1_file.exists():
        print(f"Skipping T1 plot: {t1_file} not found.")
        return

    out_file = out_dir / f"t1_vs_temperature.{fmt}"
    plot_t1_relaxation(t1_file, output_path=out_file)
    print(f"  [✓] Saved T1 plot -> {out_file}")


def compare_runs_t1(run_dirs: list[Path], out_dir: Path, fmt: str, dpi: int, show: bool):
    """Overlays T_1 curves from multiple runs for easy comparison."""
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = plt.cm.tab10.colors
    plotted_any = False

    for idx, r_dir in enumerate(run_dirs):
        t1_file = r_dir / "t1_relaxation.npz"
        if not t1_file.exists():
            continue

        data = np.load(t1_file, allow_pickle=True)
        temps = data["temperatures"]
        t1_fit = data["t1_fit"] if "t1_fit" in data else None
        defect = data.get("defect", r_dir.name)
        cell = data.get("cell_size", "")
        method = data.get("calc_method", "")
        label = f"{defect} {cell} ({method})" if cell else r_dir.name

        color = colors[idx % len(colors)]
        if t1_fit is not None:
            valid = np.isfinite(t1_fit) & (t1_fit > 0)
            if np.any(valid):
                ax.plot(temps[valid], t1_fit[valid], "o-", color=color, linewidth=2, markersize=5, label=label)
                plotted_any = True

    if not plotted_any:
        print("No valid T1 data found across provided run directories.")
        return

    ax.set_xlabel("Temperature (K)", fontsize=14)
    ax.set_ylabel(r"$T_1$ (s)", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.set_title(r"Comparison of $T_1$ Relaxation Times", fontsize=15)
    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()

    out_file = out_dir / f"t1_comparison.{fmt}"
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    print(f"  [✓] Saved multi-run T1 comparison -> {out_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot results from spin-phonon pipeline run folders."
    )
    parser.add_argument(
        "run_dirs",
        type=str,
        nargs="+",
        help="Path to one or more run directories (e.g. runs/NV_64_all_bands_1d_...).",
    )

    # Plot selectors
    parser.add_argument("-c", "--coupling", action="store_true", help="Plot spin-phonon coupling coefficients.")
    parser.add_argument("-r", "--rates", action="store_true", help="Plot transition rates vs temperature.")
    parser.add_argument("-t", "--t1", action="store_true", help="Plot T1 relaxation times vs temperature.")
    parser.add_argument("-a", "--all", action="store_true", help="Plot all graph types (default if none selected).")

    # Output options
    parser.add_argument("-o", "--out_dir", type=str, help="Custom output directory for figures.")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Image format.")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for raster images.")
    parser.add_argument("--show", action="store_true", help="Display figures interactively.")

    args = parser.parse_args()

    # Default to all plots if no specific flag chosen
    do_all = args.all or (not args.coupling and not args.rates and not args.t1)
    plot_coupling = do_all or args.coupling
    plot_rates = do_all or args.rates
    plot_t1 = do_all or args.t1

    run_paths = [Path(p) for p in args.run_dirs]

    # If multiple run directories provided and T1 is requested, also create a comparison plot
    if len(run_paths) > 1 and plot_t1:
        comp_out = Path(args.out_dir) if args.out_dir else run_paths[0].parent
        comp_out.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Generating Comparison Plots Across {len(run_paths)} Runs ---")
        compare_runs_t1(run_paths, comp_out, args.format, args.dpi, args.show)

    for run_dir in run_paths:
        if not run_dir.is_dir():
            print(f"Warning: {run_dir} is not a valid directory – skipping.")
            continue

        fig_out = Path(args.out_dir) if args.out_dir else (run_dir / "figures")
        fig_out.mkdir(parents=True, exist_ok=True)

        print(f"\n--- Generating Plots for Run: {run_dir.name} ---")
        if plot_coupling:
            plot_run_coupling(run_dir, fig_out, args.format, args.dpi, args.show)
        if plot_rates:
            plot_run_rates(run_dir, fig_out, args.format, args.dpi, args.show)
        if plot_t1:
            plot_run_t1(run_dir, fig_out, args.format, args.dpi, args.show)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
