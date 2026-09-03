from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence, Union
import numpy as np
import matplotlib.pyplot as plt

from beyblade.constants import CONSTANTS
from beyblade.models import SpinPhononCouplingData
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
    vline_scale: str = "MHz",
) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """
    Plots 1D spin-phonon coupling coefficients (V_00, V_pm, V_0pm in MHz) as discrete vertical lines,
    and their Gaussian-smeared spectral functions F(w) on a twin y-axis.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()
    ax2 = ax.twinx() if ax2 is None else ax2

    vfs = CONSTANTS.get("MHz2J", 1.0)
    if vline_scale == "J":
        V_0_0_j = V_0_0 * vfs
        V_p_m_j = V_p_m * vfs
        V_0_pm_j = V_0_pm * vfs
    else:
        V_0_0_j = V_0_0
        V_p_m_j = V_p_m
        V_0_pm_j = V_0_pm

    y_data = {
        label_prefix + r"$V_{+-}^l$": V_p_m_j,
        label_prefix + r"$V_{0\pm}^l$": V_0_pm_j,
        label_prefix + r"$V_{00}^l$": V_0_0_j,
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

    for V, col, lbl in [(V_p_m_j, "red", "+-"), (V_0_pm_j, "blue", r"0\pm"), (V_0_0_j, "black", "00")]:
        smooth_x, smooth_y = MathUtils.smear_data(frequencies_mev, V**2, res, sigma)
        ax2.plot(smooth_x, smooth_y, color=col, linewidth=2, label=label_prefix + f"$F_{{{lbl}}}^{(1)}$")

    ax.set_ylabel(f"Coupling coefficient ({vline_scale})")
    ax.set_ylim(bottom=0)

    ax.set_xlim(0, max(200.0, float(np.max(frequencies_mev)) * 1.05 if len(frequencies_mev) > 0 else 200.0))
    ax.set_xlabel("Vibration frequency (meV)")
    ax2.set_ylabel(r"Spectral function (MHz$^2$/meV)")
    ax2.set_ylim(bottom=0)

    if zfs_mev is not None:
        ax.axvline(x=zfs_mev, color="gray", linewidth=1, linestyle="--", label=r"$\hbar\omega = D$")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True)

    return fig, ax, ax2


def plot_transition_rates_stacked(
    rates_data: Union[str, Path, dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    log_scale: bool = True,
) -> list[tuple[plt.Figure, plt.Axes]]:
    """
    Plots stacked area charts for transition rates (0_1 and 1_-1) versus temperature.
    """
    if isinstance(rates_data, (str, Path)):
        data = np.load(str(rates_data), allow_pickle=True)
    else:
        data = rates_data

    temperatures = np.asarray(data["temperatures"], dtype=float)
    defect = str(data.get("defect", ""))
    cell_size = str(data.get("cell_size", ""))

    first = data["first_order"][()] if "first_order" in data else {}
    second = data["second_order"][()] if "second_order" in data else {}
    two_ph = data["two_phonon"][()] if "two_phonon" in data else {}

    colors = ["#1b9e77", "#d95f02", "#7570b3"]  # direct, 1st-order Raman, 2nd-order Raman
    process_labels = ["Direct (1-phonon)", "1st-order Raman", "2nd-order Raman"]

    transitions = [("0_1", r"$0 \leftrightarrow \pm 1$"), ("1_-1", r"$+1 \leftrightarrow -1$")]
    figures = []

    for trans_key, trans_title in transitions:
        rates_stack = []
        labels_stack = []

        if trans_key in first and len(first[trans_key]) == len(temperatures):
            rates_stack.append(np.asarray(first[trans_key], dtype=float))
            labels_stack.append(process_labels[0])
        else:
            rates_stack.append(np.zeros_like(temperatures))
            labels_stack.append(process_labels[0])

        if trans_key in second and len(second[trans_key]) == len(temperatures):
            rates_stack.append(np.asarray(second[trans_key], dtype=float))
            labels_stack.append(process_labels[1])
        else:
            rates_stack.append(np.zeros_like(temperatures))
            labels_stack.append(process_labels[1])

        if trans_key in two_ph and len(two_ph[trans_key]) == len(temperatures):
            rates_stack.append(np.asarray(two_ph[trans_key], dtype=float))
            labels_stack.append(process_labels[2])
        else:
            rates_stack.append(np.zeros_like(temperatures))
            labels_stack.append(process_labels[2])

        fig, ax = plt.subplots(figsize=(7, 5))
        cumulative = np.zeros_like(temperatures)

        for rate, label, col in zip(rates_stack, labels_stack, colors):
            upper = cumulative + rate
            ax.fill_between(temperatures, cumulative, upper, color=col, alpha=0.55, label=label, linewidth=0)
            ax.plot(temperatures, upper, color=col, lw=0.8, alpha=0.9)
            cumulative = upper.copy()

        # Total line
        ax.plot(temperatures, cumulative, color="black", lw=1.5, label="Total")

        if log_scale:
            ax.set_xscale("log")
            ax.set_yscale("log")
            valid_cum = cumulative[cumulative > 0]
            if len(valid_cum) > 0:
                ax.set_ylim(bottom=max(1e-12, float(np.min(valid_cum)) * 0.5))

        ax.set_xlabel("Temperature (K)")
        ax.set_ylabel(r"Transition rate $\Gamma$ (s$^{-1}$)")
        title_str = f"Transition rate {trans_title}"
        if defect:
            title_str += f" ({defect} {cell_size})"
        ax.set_title(title_str)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", frameon=True)
        fig.tight_layout()

        if output_path is not None:
            p = Path(output_path)
            stem = p.stem
            out_file = p.parent / f"{stem}_{trans_key}{p.suffix}"
            fig.savefig(out_file, dpi=300)
            print(f"Saved stacked rate figure to: {out_file}")

        figures.append((fig, ax))

    if show:
        plt.show()

    return figures


def plot_t1_relaxation(
    t1_data: Union[str, Path, dict[str, Any]],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots T_1 relaxation times versus temperature.
    """
    if isinstance(t1_data, (str, Path)):
        data = np.load(str(t1_data), allow_pickle=True)
    else:
        data = t1_data

    temperatures = np.asarray(data["temperatures"], dtype=float)
    defect = str(data.get("defect", ""))
    cell_size = str(data.get("cell_size", ""))
    calc_method = str(data.get("calc_method", ""))
    init_state = str(data.get("init_state", "ms_0"))

    t1_fit = np.asarray(data["t1_fit"], dtype=float) if "t1_fit" in data else None
    t1_eigenval = np.asarray(data["t1_eigenval"], dtype=float) if "t1_eigenval" in data else None

    # Fallback for legacy key
    if t1_fit is None and "T1_times" in data:
        t1_fit = np.asarray(data["T1_times"], dtype=float)
    if t1_fit is None and "T1_range" in data:
        t1_fit = np.asarray(data["T1_range"], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 6))

    label_suffix = f" ({defect} {cell_size})" if defect else ""
    if t1_fit is not None:
        valid = np.isfinite(t1_fit) & (t1_fit > 0)
        ax.plot(temperatures[valid], t1_fit[valid], "o-", color="#1f77b4", linewidth=2, markersize=5, label=f"$T_1$ ODE fit{label_suffix}")

    if t1_eigenval is not None:
        valid_eig = np.isfinite(t1_eigenval) & (t1_eigenval > 0)
        ax.plot(temperatures[valid_eig], t1_eigenval[valid_eig], "--", color="#d62728", linewidth=1.8, label=f"$T_1$ Eigenvalue{label_suffix}")

    ax.set_xlabel("Temperature (K)", fontsize=14)
    ax.set_ylabel(r"$T_1$ (s)", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, which="both", linestyle=":", alpha=0.6)
    ax.tick_params(axis="both", which="both", direction="in")
    ax.set_title(r"$T_1$ Spin Relaxation Time vs Temperature", fontsize=15)
    ax.legend(frameon=True, fontsize=12)
    fig.tight_layout()

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)
        print(f"Saved T1 figure to: {p}")

    if show:
        plt.show()

    return fig, ax


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
    zfs_2nd_derivs_unit: str = "MHz",
) -> tuple[plt.Figure, plt.Axes]:
    """Plots a 2D Raman phonon coupling intensity heatmap."""
    if zfs_2nd_derivs_unit == "J":
        zfs_2nd_derivs_mhz = zfs_2nd_derivs_mhz / CONSTANTS["MHz2J"]
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

    def plot_data(self, data_files: Sequence[Union[str, Path, SpinPhononCouplingData]], args: Any):
        """
        Processes and plots one or more derivative dataset files or SpinPhononCouplingData objects.
        Uses the internal unit conversion of SpinPhononCouplingData instead of manual conversions.
        """
        for item in data_files:
            if isinstance(item, SpinPhononCouplingData):
                raw_data = item
            else:
                raw_data = SpinPhononCouplingData.load(item)

            # Convert to display units (MHz for coupling, meV for frequencies) using dataclass methods!
            display_data = raw_data.to_unit("MHz").frequencies_to_unit("meV")

            cell_size = display_data.cell_size
            defect = display_data.defect
            pert_scale = display_data.pert_scale
            is_second_order = display_data.order == 2

            freqs_mev = display_data.frequencies
            zfs_mev = (
                display_data.ground_state_zfs.to_unit("meV").D
                if display_data.ground_state_zfs is not None
                else None
            )

            if is_second_order and display_data.V_0_0.ndim == 2:
                # Extract diagonal for 1D representation safely without mutating original data
                V_0_0 = display_data.V_0_0.diagonal()
                V_p_m = display_data.V_p_m.diagonal()
                V_0_pm = display_data.V_0_pm.diagonal()
                calc_suffix = "_2ph"
            else:
                V_0_0 = display_data.V_0_0
                V_p_m = display_data.V_p_m
                V_0_pm = display_data.V_0_pm
                calc_suffix = ""

            if getattr(args, "ipr", False) and display_data.iprs is not None:
                fig, ax = plot_ipr_spectrum(freqs_mev, display_data.iprs, as_bar=getattr(args, "bar", False))
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
