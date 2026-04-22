#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Ensure we can import from the current directory
sys.path.append(str(Path(__file__).parent))

try:
    from transition_rate import TransitionRate
except ImportError:
    print("Error: Could not import TransitionRate class. Ensure transition_rate.py is in the same directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="Calculate and plot spin transition rates vs Temperature using ZFS data."
    )
    parser.add_argument("data_file", type=str, help="Path to the ZFS coefficients data file (.npz)")
    parser.add_argument("--two-phonon", type=str, help="Path to the two-phonon ZFS data file (.npz)")
    parser.add_argument("--t-start", type=float, default=0.0, help="Start temperature in Kelvin (default: 0)")
    parser.add_argument("--t-end", type=float, default=300.0, help="End temperature in Kelvin (default: 300)")
    parser.add_argument("--t-step", type=float, default=10.0, help="Temperature step in Kelvin (default: 10)")
    parser.add_argument("-o", "--output", type=str, help="Output filename to save the plot (e.g., rates.png)")
    parser.add_argument("--log", action="store_true", help="Use logarithmic scale for the Y-axis")
    parser.add_argument("-p", "--plot", action="store_true", help="Show the plot interactively instead of saving")
    parser.add_argument("--second-order", action="store_true", help="Include second-order transition rates in the plot")

    args = parser.parse_args()

    data_path = Path(args.data_file)
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

    # Prepare temperature range
    t_start = max(0.0, args.t_start)
    temperatures = np.arange(t_start, args.t_end + args.t_step, args.t_step)
    
    # Store results
    # Keys match those in TransitionRate.compute_transition_rate
    keys = ["V_0_pm", "V_p_m"]
    results = {k: {"first_order": [], "second_order": [], "two_phonon": []} for k in keys}
    valid_temps = []

    print(f"Computing transition rates for {len(temperatures)} temperature points...")

    for T in temperatures:
        rates = calculator.compute_transition_rates(T)
        if rates:
            for k in keys:
                for order in ["first_order", "second_order", "two_phonon"]:
                    transition = "0_to_1" if k == "V_0_pm" else "1_to_-1"
                    results[k][order].append(rates[order].get(transition, 0))
            valid_temps.append(T)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    labels = {
        "V_0_pm": {
            "first_order": r"$\Gamma_{1, 0\pm}^{(1)}$",
            "second_order": r"$\Gamma_{2, 0\pm}^{(1)}$",
            "two_phonon": r"$\Gamma_{1, 0\pm}^{(2)}$",
        },
        "V_p_m": {
            "first_order": r"$\Gamma_{1, +-}^{(1)}$",
            "second_order": r"$\Gamma_{2, +-}^{(1)}$",
            "two_phonon": r"$\Gamma_{1, +-}^{(2)}$"
        },
        "V_0_0": {
            "first_order": r"$\Gamma_{1, 00}^{(1)}$",
            "second_order": r"$\Gamma_{2, 00}^{(1)}$",
            "two_phonon": r"$\Gamma_{1, 00}^{(2)}$"
        }
    }

    if args.log:
        plt.yscale("log")
        for k in keys:
            for order in ["first_order", "second_order", "two_phonon"]:
                first_elem = results[k][order][0]
                if first_elem == 0:
                    first_elem = 1e-10
    for k in keys:
        plt.plot(valid_temps, results[k]["first_order"], label=labels.get(k).get("first_order", labels[k]["first_order"]), linewidth=2)
        if args.two_phonon:
            plt.plot(valid_temps, results[k]["two_phonon"], label=labels.get(k).get("two_phonon", labels[k]["two_phonon"]), linewidth=2)
        if args.second_order:
            plt.plot(valid_temps, results[k]["second_order"], label=labels.get(k).get("second_order", labels[k]["second_order"]), linewidth=2)

    plt.xlabel("Temperature (K)")
    plt.ylabel(r"Transition Rate ($s^{-1}$)")
    plt.title(f"Spin Transition Rates vs Temperature\nData: {data_path.name}")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0)

    if args.plot:
        plt.show()
    else:
        if args.output:
            filename = args.output
        else:
            # Construct descriptive filename
            stem = data_path.stem.replace("zfs_coefficients_", "").strip("_")
            scale_str = "_log" if args.log else ""
            filename = f"rates_{stem}_{int(t_start)}-{int(args.t_end)}K{scale_str}.png"
            
            # Save to figures directory by default
            Path("figures").mkdir(exist_ok=True)
            filename = str(Path("figures") / filename)
            
        plt.savefig(filename, dpi=300)
        print(f"Plot saved to {filename}")

if __name__ == "__main__":
    main()