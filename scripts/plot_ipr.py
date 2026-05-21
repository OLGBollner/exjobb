import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
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
    freqs = phonon_mgr.get_freqs()

    coupling_data = np.load(args.data)

    plt.rcParams.update({
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14
    })

    fig, ax = plt.subplots(figsize=(6, 5))

    V_0_0 =  coupling_data["V_0_0"]  / CONSTANTS["MHz2J"]
    V_0_pm = coupling_data["V_0_pm"] / CONSTANTS["MHz2J"]
    V_p_m =  coupling_data["V_p_m"]  / CONSTANTS["MHz2J"]

    if V_0_0.ndim > 1:
        V_0_0 =  np.diag(V_0_0)
        V_0_pm = np.diag(V_0_pm)
        V_p_m =  np.diag(V_p_m)


    sym_data = phonon_mgr.symmetry_data['sym']

    mask_A1 = (sym_data == 'A1')
    mask_A2 = (sym_data == 'A2')
    mask_E = (sym_data == 'Ex') | (sym_data == 'Ey')
    mask_mixed = ~(mask_A1 | mask_A2 | mask_E)

    size = 32
    ax.scatter(ipr[mask_A1], V_0_0[mask_A1], marker='^', color='black', s=size, alpha=0.6)
    ax.scatter(ipr[mask_A2], V_0_0[mask_A2], marker='s', color='black', s=size, alpha=0.6)
    ax.scatter(ipr[mask_E], V_0_0[mask_E], marker='o', color='black', s=size, alpha=0.6)
    ax.scatter(ipr[mask_mixed], V_0_0[mask_mixed], marker='|', color='black', s=size, alpha=0.6)

    ax.scatter(ipr[mask_A1], V_0_pm[mask_A1], marker='^', color='blue', s=size, alpha=0.6)
    ax.scatter(ipr[mask_A2], V_0_pm[mask_A2], marker='s', color='blue', s=size, alpha=0.6)
    ax.scatter(ipr[mask_E], V_0_pm[mask_E], marker='o', color='blue', s=size, alpha=0.6)
    ax.scatter(ipr[mask_mixed], V_0_pm[mask_mixed], marker='|', color='blue', s=size, alpha=0.6)

    ax.scatter(ipr[mask_A1], V_p_m[mask_A1], marker='^', color='red', s=size, alpha=0.6)
    ax.scatter(ipr[mask_A2], V_p_m[mask_A2], marker='s', color='red', s=size, alpha=0.6)
    ax.scatter(ipr[mask_E], V_p_m[mask_E], marker='o', color='red', s=size, alpha=0.6)
    ax.scatter(ipr[mask_mixed], V_p_m[mask_mixed], marker='|', color='red', s=size, alpha=0.6)

    ax.set_xlabel('IPR')
    ax.set_ylabel('Coupling coefficient')
    ax.set_title('Coupling vs IPR')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label=r"$V_{00}^l$"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label=r"$V_{0\pm}^l$"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label=r"$V_{+-}^l$"),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=8, label='Pure A1'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=8, label='Pure A2'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Pure E'),
        Line2D([0], [0], marker='|', color='w', markeredgecolor='gray', markersize=8, label='Mixed')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    # plt.yscale("log")
    # plt.xscale("log")
    plt.tight_layout()

    if args.plot:
        plt.show()
    else:
        filename = ""
        if args.output:
            filename = args.output
        else:
            order = "2d" if "2d" in args.data else "1d"
            filename += f"{coupling_data['defect']}_{coupling_data['cell_size']}_{coupling_data['sub_folder']}_{coupling_data['pert_scale']}_{order}_ipr"

        filename += args.format if args.format else ".png"
        print("Saving figure in: ", "figures/" + filename)
        plt.savefig("figures/" + filename)
