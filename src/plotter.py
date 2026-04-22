import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from constants import CONSTANTS
from utils import MathUtils


class ZFSPlotter:
    def __init__(self, plot_config=None):
        self.config = plot_config or {}
        plt.rcParams.update({
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 10
        })

    def plot_data(self, data_files, args):

        zfs_data = [np.load(file) for file in data_files]

        cell_size = zfs_data[0]["cell_size"]
        sim_type = zfs_data[0]["sub_folder"]

        if zfs_data[0].get("second_order"):
            fig_2d, ax_2d = self._process_2d_plots(zfs_data, args)
            diag_data = {key: value for key, value in zfs_data[0].items() if key not in ["zfs_derivs", "V_p_m", "V_0_pm", "V_0_0"]}

            print("Extracting diagonal elements...")
            diag_data["zfs_derivs"] = np.zeros(zfs_data[0]["zfs_derivs"].shape[0]) # Dummy data
            diag_data["V_p_m"] = np.diag(zfs_data[0]["V_p_m"])
            diag_data["V_0_pm"] = np.diag(zfs_data[0]["V_0_pm"])
            diag_data["V_0_0"] = np.diag(zfs_data[0]["V_0_0"])
            print("Done!")

            fig, ax = self._process_1d_plots([diag_data], args)
            sim_type += "_2ph"
        else:
            fig, ax = self._process_1d_plots(zfs_data, args)

        plt.tight_layout()
        if args.plot:
            plt.show()
        else:
            out_file = f"{args.output}{args.format}" if args.output else f"zfs_plot_{cell_size}_{sim_type}.png"
            Path("figures").mkdir(exist_ok=True)
            fig.savefig(f"figures/{out_file}")
            print(f"Saved figure in figures/{out_file}")
            if zfs_data[0].get("second_order"):
                fig_2d.savefig(f"figures/{out_file.replace('.png', '_heatmap.png')}")
                print(f"Saved heatmap in figures/{out_file.replace('.png', '_heatmap.png')}")

    def _process_1d_plots(self, zfs_data, args):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax2 = ax.twinx()
        colors = iter(["red", "black", "blue", "orange", "green"])
        
        plot_name = ""

        if args.difference and len(zfs_data) > 1:
            diff_data = {key: np.abs(zfs_data[1][key] - zfs_data[0][key])
                         for key in zfs_data[0].keys() if key not in ["ipr", "freqs"]}
            diff_data["ipr"] = zfs_data[0]["ipr"]
            diff_data["freqs"] = zfs_data[0]["freqs"]
            zfs_data = [diff_data]
            plot_name = r"$\Delta$"

        for i, data in enumerate(zfs_data):
            self._render_single_dataset(ax, ax2, data, args, plot_name, colors)
        
        return fig, ax

    def _process_2d_plots(self, zfs_data, args):
        fig, ax = plt.subplots(figsize=(6, 5))
        for i, data in enumerate(zfs_data):
            self._render_heatmap(ax, fig, data)
        return fig, ax

    def _render_single_dataset(self, ax, ax2, data, args, plot_name, colors):
        freqs = data["freqs"]
        pert_scale = data["pert_scale"]
        sym = data["sym"]
        sigma = 7.5
        res = 0.5
        marker_map = {
            "A1": "o",
            "A2": "s",
            "Ex": "^",
            "Ey": "v"
        }

        if args.norm:
            color = next(colors)
            coupling_strength = np.linalg.norm(data["zfs_derivs"], axis=(1, 2)) / CONSTANTS["MHz2meV"]

            if args.bar:
                ax2.bar(range(len(coupling_strength)), coupling_strength, color=color, alpha=0.6, label=plot_name + r"$|\partial D^{(1)}|$")
            else:
                if not args.ipr:
                    ax.vlines(freqs, [0], coupling_strength, color=color, alpha=0.6, label=plot_name + r"$|\partial D^{(1)}|$" + f" {pert_scale}")
                smooth_x, smooth_y = MathUtils.smear_data(freqs, coupling_strength, 1, sigma)
                ax2.plot(smooth_x, smooth_y, color=color, linewidth=2, label=plot_name + r"$F^{(1)}$" + f" {pert_scale}")
        else:
            V_0_pm = data["V_0_pm"] / CONSTANTS["MHz2meV"]
            V_p_m = data["V_p_m"] / CONSTANTS["MHz2meV"]
            V_0_0 = data["V_0_0"] / CONSTANTS["MHz2meV"]

            if not args.ipr:
                ax.vlines(freqs, [0], V_p_m, label=plot_name + r"$V_{+-}^l$", color="red", alpha=0.6)
                ax.vlines(freqs, [0], V_0_pm, label=plot_name + r"$V_{0\pm}^l$", color="blue", alpha=0.6)
                ax.vlines(freqs, [0], V_0_0, label=plot_name + r"$V_{00}^l$", color="black", alpha=0.6)
                # for sym_type, marker_shape in marker_map.items():
                #     mask = sym == sym_type
                #     ax.scatter(freqs[mask], V_p_m[mask], marker=marker_shape, edgecolors="none", color="red")
                #     ax.scatter(freqs[mask], V_0_pm[mask], marker=marker_shape, edgecolors="none", color="blue")
                #     ax.scatter(freqs[mask], V_0_0[mask], marker=marker_shape, edgecolors="none", color="black")

            for V, col, lbl in [(V_p_m, "red", "+-"), (V_0_pm, "blue", r"0\pm"), (V_0_0, "black", "00")]:
                smooth_x, smooth_y = MathUtils.smear_data(freqs, V * CONSTANTS["MHz2meV"], res, sigma)
                ax2.plot(smooth_x, smooth_y / CONSTANTS["MHz2meV"]**2, color=col, linewidth=2, label=plot_name + f"$F_{{{lbl}}}^{(1)}$")

        if args.ipr:
            if args.bar:
                ax.bar(range(len(data["ipr"])), data["ipr"], color="blue", alpha=0.6, label="IPR")
            else:
                smooth_x, smooth_y = MathUtils.smear_data(freqs, data["ipr"], 1, sigma)
                ax.plot(smooth_x, smooth_y, color="blue", alpha=0.6, label="IPR")
            ax.set_ylabel("Phonon IPR")
        else:
            ax.set_ylabel("Coupling coefficient (MHz)")
            ax.set_ylim(0)

        if args.bar:
            ax.set_xlabel("Mode index")
            ax2.set_ylabel("Coupling coefficient (MHz)")
            legend_pos = "upper right"
        else:
            ax.set_xlim(0, 200)
            ax.set_xlabel("Vibration frequency (meV)")
            ax2.set_ylabel(r"Spectral function (MHz$^2$/meV)")
            ax2.set_ylim(0)
            legend_pos = "upper left"

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc=legend_pos, frameon=True)

    def _render_heatmap(self, ax, fig, data):
        Z = np.linalg.norm(data["zfs_derivs"], axis=(2, 3))
        freqs = data["freqs"]
        allowed_transitions = Z * MathUtils.broad_delta(freqs[:, None], freqs[None, :], 7.5)
        sigma = 7.5
        res = 1

        X, Y, spectral_density = MathUtils.get_2d_spectral_density(data["freqs"], allowed_transitions, sigma, res)

        mesh = ax.pcolormesh(X, Y, spectral_density/CONSTANTS["MHz2meV"]**2, cmap="viridis", shading="auto")
        ax.set_xlabel("Vibration frequency (meV)")
        ax.set_ylabel("Vibration frequency (meV)")

        cbar = fig.colorbar(mesh, ax=ax)
        cbar.set_label("Spectral function intensity (meV)")