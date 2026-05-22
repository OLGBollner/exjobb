#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

from beyblade.constants import CONSTANTS

# Ensure we can import from the current directory
sys.path.append(str(Path(__file__).parent))

try:
    from beyblade.transition_rate import TransitionRate
except ImportError:
    print("Error: Could not import TransitionRate class. Ensure transition_rate.py is in the same directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate and plot spin transition rates vs Temperature using ZFS data."
    )
    parser.add_argument("data_file", type=str, nargs="+", help="Path to the data file (.npz)")

    # Calculating
    parser.add_argument("--calc", action="store_true", help="Calculate transition rates")
    parser.add_argument("--two-phonon", type=str, help="Path to the two-phonon ZFS data file (.npz)")
    parser.add_argument("--t-start", type=float, default=0.0, help="Start temperature in Kelvin (default: 0)")
    parser.add_argument("--t-end", type=float, default=300.0, help="End temperature in Kelvin (default: 300)")
    parser.add_argument("--t-step", type=float, default=10.0, help="Temperature step in Kelvin (default: 10)")

    # Plotting
    parser.add_argument("--save", action="store_true", help="Save the plot")
    parser.add_argument("-o", "--output", type=str, help="Output filename to save the plot (e.g., rates.png)")
    parser.add_argument("--log", action="store_true", help="Use logarithmic scale for the axes")
    parser.add_argument("-p", "--plot", action="store_true", help="Show the plot interactively instead of saving")
    parser.add_argument("--first-order", action="store_true")
    parser.add_argument("--second-order", action="store_true")
    parser.add_argument("--second-phonon", action="store_true")

    args = parser.parse_args()

    if args.calc:
        assert len(args.data_file) == 1
        data_path = Path(args.data_file[0])
        if not data_path.exists():
            print(f"Error: Data file '{str(data_path)}' not found.")
            return

        if args.two_phonon:
            two_phonon_path = Path(args.two_phonon)
            if not two_phonon_path.exists():
                print(f"Error: Two-phonon data file '{str(two_phonon_path)}' not found.")
                return
        else:
            two_phonon_path = None

        # Initialize calculator
        try:
            calculator = TransitionRate(str(data_path), two_phonon_path)
        except Exception as e:
            print(f"Error loading data file: {e}")
            return

        t_start = max(0.0, args.t_start)
        low_range = np.arange(0, 1, args.t_step/100)
        high_range = np.arange(1 + args.t_step, args.t_end + args.t_step, args.t_step)
        temperatures = np.concatenate([low_range, high_range])
        
        results = {
            "first_order": {
              "0_1": [],
              "1_-1": [],
              },
            "second_order": {
              "0_1": [],
              "1_-1": [],
              },
            "two_phonon": {
              "0_1": [],
              "1_-1": [],
              },
            }
        valid_temps = []

        print(f"Computing transition rates for {len(temperatures)} temperature points...")
        
        omega, J_0_pm, J_p_m, J_0_0 = calculator.get_spectral_density(res=0.01, sigma=7.5)
        if args.two_phonon:
            omega_x, omega_y, J2_0_pm, J2_p_m, J2_0_0 = calculator.get_2d_spectral_density(res=0.5, sigma=7.5)

        zfs = calculator.data["zfs"] / CONSTANTS["meV2J"]
        print(zfs / CONSTANTS["GHz2meV"])

        for T in temperatures:
            calculator.compute_transition_rates(T, omega, J_0_pm, J_p_m, J_0_0, zfs)
            if args.two_phonon:
                calculator.compute_two_phonon_rates(T, omega_x, omega_y, J2_0_pm, J2_p_m, J2_0_0, zfs)
            rates = calculator.get_total_rates()
            if rates:
                for order in ["first_order", "second_order", "two_phonon"]:
                    for transition, rate in rates[order].items():
                        results[order][transition].append(rate)
                valid_temps.append(T)

        meta_data = {
            "cell_size": calculator.data["cell_size"],
            "defect": calculator.data["defect"],
            "sub_folder": calculator.data["sub_folder"],
            "pert_scale": calculator.data["pert_scale"],
            }

        save_name = f"{meta_data['defect']}_{meta_data['cell_size']}_rates_{meta_data['sub_folder']}_{meta_data['pert_scale']}"

        Path("rates").mkdir(exist_ok=True)
        save_name = str(Path("rates") / save_name)

        print(f"Saving transition rates in file: {save_name}.npz")
        np.savez(save_name, **meta_data, **results, valid_temps=valid_temps)

    if args.plot:
        plt.rcParams.update({
            "axes.titlesize": 16,
            "axes.labelsize": 16,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 14
        })
        # Line styles for multiple files (solid, dashed, dash‑dot, dotted, …)
        linestyles = ['-', '--', '-.', ':']
        color_for_pair = {}            # (key, order) -> matplotlib colour
        first_file = True

        labels = {
            "first_order": {
              "0_1": r"$\Gamma_{1, 0\pm}^{(1)}$",
              "1_-1": r"$\Gamma_{1, +-}^{(1)}$",
              "0_0": r"$\Gamma_{1, 00}^{(1)}$",
              },
            "second_order": {
              "0_1": r"$\Gamma_{2, 0\pm}^{(1)}$",
              "1_-1": r"$\Gamma_{2, +-}^{(1)}$",
              "0_0": r"$\Gamma_{2, 00}^{(1)}$",
              },
            "two_phonon": {
              "0_1": r"$\Gamma_{1, 0\pm}^{(2)}$",
              "1_-1": r"$\Gamma_{1, +-}^{(2)}$",
              "0_0": r"$\Gamma_{1, 00}^{(2)}$",
              }
            }

        # Determine which orders to plot based on arguments
        orders_to_plot = []
        if args.first_order:
            orders_to_plot.append("first_order")
        if args.second_order:
            orders_to_plot.append("second_order")
        if args.second_phonon:
            orders_to_plot.append("two_phonon")
        if orders_to_plot == []:
            raise ValueError("No rate specified.")

        plt.figure(figsize=(6, 5))

        # Set log scales if requested (must be done before plotting)
        if args.log:
            plt.yscale("log")
            plt.xscale("log")
        data_files = [np.load(file, allow_pickle=True) for file in args.data_file]
        # Process each data file
        for i, data in enumerate(data_files):

            # Extract temperatures and rates
            valid_temps = data["valid_temps"]
            # The original results were saved as a dict of dicts: data[key] is a 0‑d array of object
            # keys = [k for k in data.files if k != "valid_temps"]   # all transition keys

            results = {k: item[()] for k, item in data.items() if "order" in k or "phonon" in k}

            # Replace zeros with a tiny number for log scale
            if args.log:
                for k, item in results.items():
                    if k not in orders_to_plot:
                        continue
                    for transition, rate in item.items():
                        arr = np.array(rate)
                        arr[arr == 0] = 1e-12
                        item[transition] = arr

            # Plot each transition from this file
            for k, item in results.items():
                if k not in orders_to_plot:
                    continue
                for transition, rate in item.items():
                    # Build label: transition label only for the first file
                    label = None
                    if first_file:
                        label = labels.get(k, {}).get(transition,
                                  f"{k} {transition}")   # fallback if key/order not in 'labels'

                    # Line style based on file index
                    ls = linestyles[i % len(linestyles)]

                    # Get consistent colour for this (k, order) pair
                    pair = (k, transition)
                    if pair in color_for_pair:
                        # Reuse colour from the first file
                        plt.plot(valid_temps, rate,
                                 color=color_for_pair[pair], linestyle=ls,
                                 linewidth=2, label=label)
                    else:
                        line, = plt.plot(valid_temps, rate,
                                         linestyle=ls, linewidth=2, label=label)
                        color_for_pair[pair] = line.get_color()

            first_file = False

        # Add a small legend for line styles (one entry per file)
        from matplotlib.lines import Line2D
        style_handles = []
        spin_formalism_labels = {
            "all_bands":          r"$D_{cont}$",
            "defect_band_approx": r"$D_{corr}$"
        }

        all_defects    = [str(d['defect'])    for d in data_files]
        all_sizes      = [int(d['cell_size']) for d in data_files]
        all_formalisms = [str(d['sub_folder']) for d in data_files]

        show_defect    = len(set(all_defects)) > 1
        show_size      = len(set(all_sizes)) > 1
        show_formalism = len(set(all_formalisms)) > 1

        for i, data in enumerate(data_files):
            defect = str(data["defect"])
            cell_size = str(data["cell_size"])
            formalism = str(data["sub_folder"])
            parts = []
            if show_defect:
                parts.append(defect)
            if show_size:
                parts.append(cell_size)
            if show_formalism:
                formalism_str = spin_formalism_labels.get(formalism,
                                                          formalism)
                parts.append(f"({formalism_str})")

            if not parts:   # fallback when everything is identical
                parts.append(f"{defect} {cell_size} "
                             f"({spin_formalism_labels.get(formalism, formalism)})")

            label = " ".join(parts)

            style_handles.append(
                Line2D([0], [0], color='black',
                       linestyle=linestyles[i % len(linestyles)],
                       label=label))

        # Combine with the existing legend (transition labels)
        leg1 = plt.legend(loc='best')
        leg2 = plt.legend(handles=style_handles, loc='upper center')

        plt.gca().add_artist(leg1)   # keep both legends visible

        plt.xlabel("Temperature (K)")
        plt.ylabel(r"Transition Rate (s$^{-1}$)")
        plt.title("Spin Transition Rates vs Temperature")
        #plt.grid(True, linestyle='--', alpha=0.7)
        plt.axvline(x=125, color="gray", linestyle="-", linewidth=1)
        #plt.ylim(1e-6,1e6)
        #plt.xlim(1e-1)
        plt.tight_layout()

        if args.save:
            plt.tight_layout()
            if args.output:
                filename = args.output
            else:
                # Construct descriptive filename
                defect = data["defect"]
                sub_folder = data["sub_folder"]
                pert_scale = data["pert_scale"]
                t_start = np.min(data["valid_temps"])
                t_end = np.max(data["valid_temps"])

                scale_str = "_log" if args.log else ""
                filename = f"{defect}_{'_'.join(orders_to_plot)}_rates_{sub_folder}_{int(t_start)}-{int(t_end)}K{scale_str}_{pert_scale}.png"
                
                # Save to figures directory by default
                Path("figures").mkdir(exist_ok=True)
                filename = str(Path("figures") / filename)
                
            plt.savefig(filename, dpi=300)
            print(f"Plot saved to {filename}")
        else:
            plt.show()

if __name__ == "__main__":
    main()
