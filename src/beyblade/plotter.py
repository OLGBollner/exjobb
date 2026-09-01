from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

from beyblade.constants import CONSTANTS
from beyblade.utils import MathUtils


def plot_vlines_sorted_by_magnitude(
    ax: plt.Axes,
    x: np.ndarray,
    y_data_dict: dict[str, Union[np.ndarray, tuple[np.ndarray, np.ndarray]]],
    *,
    sort_metric: str = "max",
    colors: Optional[Sequence[Optional[str]]] = None,
    alphas: Union[float, Sequence[float]] = 1.0,
    **vlines_kwargs,
):
    """
    Plots vertical lines so that datasets with smaller values are rendered on top (drawn last).
    """
    labels = list(y_data_dict.keys())
    data = list(y_data_dict.values())

    metric_func = {"max": np.max, "mean": np.mean, "median": np.median}.get(sort_metric, np.max)
    sort_vals = [metric_func(d[1] if isinstance(d, tuple) else d) for d in data]

    order = np.argsort(sort_vals)[::-1]  # descending: large first, small last

    if colors is None:
        colors = [None] * len(labels)
    if isinstance(alphas, (int, float)):
        alphas = [float(alphas)] * len(labels)

    # Remove alpha from kwargs if already in alphas
    vlines_kwargs.pop("alpha", None)

    for idx in order:
        label = labels[idx]
        y_data = data[idx]
        ymin, ymax = y_data if isinstance(y_data, tuple) else ([0], y_data)
        ax.vlines(
            x,
            ymin,
            ymax,
            label=label,
            color=colors[idx] if idx < len(colors) else None,
            alpha=alphas[idx] if idx < len(alphas) else 1.0,
            **vlines_kwargs,
        )


def plot_1d_spectral_functions(
    frequencies_mev: np.ndarray,
    V_0_0: np.ndarray,
    V_p_m: np.ndarray,
    V_0_pm: np.ndarray,
    *,
    zfs_mev: Optional[float] = None,
    sigma: float = 7.5,
    res: float = 1.0,
    label_prefix: str = "",
    ax: Optional[plt.Axes] = None,
    ax2: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Plots 1D spin-phonon coupling coefficients (V_00, V_pm, V_0pm in MHz) as discrete vertical lines,
    and their Gaussian-smeared spectral functions F(w) on a twin y-axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax2 = ax.twinx()
    else:
        fig = ax.get_figure()
        if ax2 is None:
            ax2 = ax.twinx()

    y_data = {
        label_prefix + r"$V_{+-}^l$": V_p_m,
        label_prefix + r"$V_{0\pm}^l$": V_0_pm,
        label_prefix + r"$V_{00}^l$": V_0_0,
    }

    plot_vlines_sorted_by_magnitude(
        ax,
        frequencies_mev,
        y_data,
        sort_metric="mean",
        colors=["red", "blue", "black"],
        alphas=[0.6, 0.6, 0.6],
        linewidth=1,
    )

    for V, col, lbl in [(V_p_m, "red", "+-"), (V_0_pm, "blue", r"0\pm"), (V_0_0, "black", "00")]:
        smooth_x, smooth_y = MathUtils.smear_data(frequencies_mev, V**2, res, sigma)
        ax2.plot(smooth_x, smooth_y, color=col, linewidth=2, label=label_prefix + f"$F_{{{lbl}}}^{(1)}$")

    ax.set_ylabel("Coupling coefficient (MHz)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(25))
    ax.set_ylim(bottom=0)

    ax.set_xlim(0, max(200.0, float(np.max(frequencies_mev)) * 1.05 if len(frequencies_mev) > 0 else 200.0))
    ax.set_xlabel("Vibration frequency (meV)")
    ax2.set_ylabel(r"Spectral function (MHz$^2$/meV)")
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax2.set_ylim(bottom=0)

    if zfs_mev is not None:
        ax.axvline(x=zfs_mev, color="gray", linewidth=1, linestyle="--", label=r"$\hbar\omega = D$")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    return fig, ax, ax2


def plot_ipr_spectrum(
    frequencies_mev: np.ndarray,
    ipr: np.ndarray,
    *,
    sigma: float = 7.5,
    as_bar: bool = False,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots Inverse Participation Ratio (IPR) across phonon frequencies or mode indices."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    if as_bar:
        ax.bar(range(len(ipr)), ipr, color="blue", alpha=0.6, label="IPR")
        ax.set_xlabel("Mode index")
    else:
        smooth_x, smooth_y = MathUtils.smear_data(frequencies_mev, ipr, 1.0, sigma)
        ax.plot(smooth_x, smooth_y, color="blue", alpha=0.8, linewidth=2, label="IPR")
        ax.set_xlabel("Vibration frequency (meV)")

    ax.set_ylabel("Phonon IPR")
    ax.legend(loc="upper right", frameon=True)
    return fig, ax


def plot_2d_spectral_density_map(
    frequencies_mev: np.ndarray,
    zfs_2nd_derivs_mhz: np.ndarray,
    *,
    sigma: float = 7.5,
    res: float = 1.0,
    ax: Optional[plt.Axes] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a 2D Raman phonon coupling intensity heatmap."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    Z = np.linalg.norm(zfs_2nd_derivs_mhz, axis=(2, 3)) if zfs_2nd_derivs_mhz.ndim == 4 else zfs_2nd_derivs_mhz
    X, Y, spectral_density = MathUtils.get_2d_spectral_density(frequencies_mev, Z, sigma, res)
    allowed_transitions = MathUtils.broad_delta(X, Y, sigma)

    mesh = ax.pcolormesh(X, Y, spectral_density * allowed_transitions, cmap="viridis", shading="auto")
    ax.set_xlabel("Vibration frequency (meV)")
    ax.set_ylabel("Vibration frequency (meV)")

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Spectral function intensity (MHz)")
    return fig, ax


class ZFSPlotter:
    """
    High-level plotter that coordinates matplotlib rendering without mutating underlying data arrays.
    """

    def __init__(self, plot_config: Optional[dict[str, Any]] = None):
        self.config = plot_config or {}
        plt.rcParams.update({
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 14,
        })

    def plot_data(self, data_files: Sequence[Union[str, Path]], args: Any):
        """
        Processes and plots one or more derivative dataset files.
        Maintains backward compatibility with CLI arguments from zfs_analysis.py.
        """
        zfs_data_list = [np.load(file, allow_pickle=True) for file in data_files]

        for data in zfs_data_list:
            cell_size = data["cell_size"]
            defect = data["defect"]
            pert_scale = data["pert_scale"]
            is_second_order = bool(data.get("second_order", False))

            freqs_mev = data["freqs"] / CONSTANTS["meV2J"]
            zfs_mev = (data["zfs"] / CONSTANTS["meV2J"]) if "zfs" in data else None

            if is_second_order:
                # Extract diagonal for 1D representation safely without mutating original data
                V_0_0 = data["V_0_0"].diagonal() / CONSTANTS["MHz2J"]
                V_p_m = data["V_p_m"].diagonal() / CONSTANTS["MHz2J"]
                V_0_pm = data["V_0_pm"].diagonal() / CONSTANTS["MHz2J"]
                calc_suffix = "_2ph"
            else:
                V_0_0 = data["V_0_0"] / CONSTANTS["MHz2J"]
                V_p_m = data["V_p_m"] / CONSTANTS["MHz2J"]
                V_0_pm = data["V_0_pm"] / CONSTANTS["MHz2J"]
                calc_suffix = ""

            if getattr(args, "ipr", False):
                fig, ax = plot_ipr_spectrum(freqs_mev, data["ipr"], as_bar=getattr(args, "bar", False))
            else:
                fig, ax, _ = plot_1d_spectral_functions(
                    freqs_mev,
                    V_0_0=V_0_0,
                    V_p_m=V_p_m,
                    V_0_pm=V_0_pm,
                    zfs_mev=zfs_mev,
                )

            plt.tight_layout()
            if getattr(args, "plot", False):
                plt.show()
            else:
                out_name = (
                    f"{args.output}{args.format}"
                    if getattr(args, "output", None)
                    else f"{defect}_{cell_size}_zfs_plot{calc_suffix}_{pert_scale}.png"
                )
                Path("figures").mkdir(exist_ok=True)
                fig.savefig(f"figures/{out_name}")
                print(f"Saved figure in figures/{out_name}")
                plt.close(fig)
