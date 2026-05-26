import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize
from matplotlib import ticker
from argparse import ArgumentParser as Parser
from beyblade.phonon_manager import PhononManager
from beyblade.constants import CONSTANTS

if __name__ == "__main__":
    parser = Parser("Plots IPR.")
    parser.add_argument("--data", metavar="data", help="File containing coupling coefficients")
    parser.add_argument("--phonon_data", metavar="phonon_data", help="File containing phonon data")
    parser.add_argument("-f", "--format", help="File format of plot")
    parser.add_argument("-o", "--output", help="Specify the output filename for the plot or data")
    parser.add_argument("-s", "--smooth", action="store_true", help="Plot smooth IPR")
    parser.add_argument("-p", "--plot", action="store_true", help="Plot the data directly instead of saving")

    args = parser.parse_args()

    phonon_mgr = PhononManager(args.phonon_data)
    ipr = phonon_mgr.get_ipr()
    
    freqs_mev = phonon_mgr.get_freqs()

    coupling_data = np.load(args.data)

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14
    })

    fig, ax = plt.subplots(figsize=(6, 5))

    # Coupling coefficients (already divided for convenient units)
    V_0_0  = coupling_data["V_0_0"]  / CONSTANTS["MHz2J"]
    V_0_pm = coupling_data["V_0_pm"] / CONSTANTS["MHz2J"]
    V_p_m  = coupling_data["V_p_m"]  / CONSTANTS["MHz2J"]

    # If matrices are stored, take diagonal elements
    if V_0_0.ndim > 1:
        V_0_0  = np.diag(V_0_0)
        V_0_pm = np.diag(V_0_pm)
        V_p_m  = np.diag(V_p_m)

    # Colour mapping: spin-phonon coupling strength
    cmap = plt.cm.plasma
    vmin = min(V_0_0.min(), V_0_pm.min(), V_p_m.min())
    vmax = max(V_0_0.max(), V_0_pm.max(), V_p_m.max())
    norm = Normalize(vmin=vmin, vmax=vmax)


    size = 100

    # Three different marker shapes plotted against the converted meV frequencies
    sc00 = ax.scatter(freqs_mev, ipr,
                      c=V_0_0, cmap=cmap, norm=norm,
                      marker='o', s=size, alpha=norm(V_0_0))
    sc0pm = ax.scatter(freqs_mev, ipr,
                       c=V_0_pm, cmap=cmap, norm=norm,
                       marker='s', s=size, alpha=norm(V_0_pm))
    scpm = ax.scatter(freqs_mev, ipr,
                      c=V_p_m, cmap=cmap, norm=norm,
                      marker='^', s=size, alpha=norm(V_p_m))

    # Colour bar based on the first scatter (all share the same norm/cmap)
    cbar = fig.colorbar(scpm, ax=ax, label='Spin-phonon coupling')

    ax.set_xlabel('Phonon energy (meV)')
    ax.set_ylabel('IPR')
    ax.set_title('IPR and coupling vs phonon energy')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 200)

    # Legend: only coupling types, symmetry information removed
    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='none', markeredgecolor='black',
               markersize=8, label=r"$V_{00}^l$"),
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor='none', markeredgecolor='black',
               markersize=8, label=r"$V_{0\pm}^l$"),
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor='none', markeredgecolor='black',
               markersize=8, label=r"$V_{+-}^l$")
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))

    plt.tight_layout()

    if args.plot:
        plt.show()
    else:
        filename = ""
        if args.output:
            filename = args.output
        else:
            order = "2d" if "2d" in args.data else "1d"
            base_name = f"{coupling_data['defect']}_{coupling_data['cell_size']}_{coupling_data['sub_folder']}_{coupling_data['pert_scale']}_{order}_ipr"
            
            ext = args.format if args.format else "png"
            if not ext.startswith('.'):
                ext = f".{ext}"
                
            filename = f"{base_name}{ext}"

        print(f"Saving figure in: figures/{filename}")
        plt.savefig(f"figures/{filename}")
