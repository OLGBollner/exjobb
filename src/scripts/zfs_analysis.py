import argparse

from zfs_calculator import ZFSCalculator
from phonon_manager import PhononManager
from pathlib import Path
from plotter import ZFSPlotter

def main():
    parser = argparse.ArgumentParser(description="Manage and analyze ZFS phonon derivatives and plot the results.")

    # Commands for ZFS calculations
    parser.add_argument("--calc", action="store_true", help="Run calculation of ZFS derivatives.")
    parser.add_argument("--sim_folder", type=str, help="Folder containing simulation results for calculation.")
    parser.add_argument("-ph", "--phonon_file", type=str, help="Phonon data file to use.")
    parser.add_argument("--all", action="store_true", help="Use results from all_bands (for calculation).")
    parser.add_argument("--approx", action="store_true", help="Use results from defect_band_approx (for calculation).")
    parser.add_argument("-d", "--debug", action="store_true", help="Print debug information for derivatives.")

    # Commands for plotting
    parser.add_argument("--plot", action="store_true", help="Display the plot directly on the screen.")
    parser.add_argument("--data_files", nargs='+', help="Files with pre-calculated ZFS data to plot.")
    parser.add_argument("-f", "--format", default=".png", help="File format for saved plot (e.g., .svg, .png).")
    parser.add_argument("-i", "--ipr", action="store_true", help="Include IPR in the plot.")
    parser.add_argument("--difference", action="store_true", help="Plot the difference between two datasets.")
    parser.add_argument("-n", "--norm", action="store_true", help="Plot the norm of the derivatives.")
    parser.add_argument("-b", "--bar", action="store_true", help="Use bar chart based on mode index.")

    # Common
    parser.add_argument("-o", "--output", help="Name of output file (for both data and image depending on context).")

    args = parser.parse_args()

    generated_files = []

    if args.calc:
        if not args.sim_folder:
            print("Error: --sim_folder must be specified when --calc is used.")
            return

        sub_folder, zfs_folder = ("all_bands", "ZFS_hyp") if args.all else ("defect_band_approx", "ZFS_occup")

        sim_folder = Path(args.sim_folder)

        if args.phonon_file:
            path_to_phonon = Path(args.phonon_file)
            if not path_to_phonon.exists():
                print(f"Error: Specified phonon file {args.phonon_file} does not exist.")
                return
        else:
            path_to_phonon = sim_folder.parent.parent / "data" / "phonon_data.npz"
            print("Using default phonon file: ", path_to_phonon)
            if not path_to_phonon.exists():
                raise FileNotFoundError(f"Default phonon file not found at {path_to_phonon}. Please provide a valid phonon data file.")
        
        phonon_manager = PhononManager(data_path=path_to_phonon)
        calculator = ZFSCalculator(
            sim_folder=args.sim_folder,
            sub_folder=sub_folder,
            zfs_folder=zfs_folder,
            phonon_manager=phonon_manager,
            debug=args.debug
        )

        order = 1 if "first_order" in args.sim_folder else 2

        print("Starting calculation of ZFS derivatives...")
        if order == 1:
            print("Processing first-order perturbations...")
            generated_files = calculator.process_first_order_perturbations(args.output)
        elif order == 2:
            print("Processing second-order perturbations...")
            generated_files = calculator.process_second_order_perturbations(args.data_files[0], args.output)

    if args.data_files and not args.sim_folder:
        files_to_plot = args.data_files
    elif args.sim_folder:
        if order == 1 and args.data_files:
            files_to_plot = args.data_files
        elif order == 2 and not generated_files:
            files_to_plot = args.data_files
        else:
            files_to_plot = generated_files

    if files_to_plot and (args.plot or args.format):
        print("Beginning visualization...")
        plotter = ZFSPlotter()
        plotter.plot_data(files_to_plot, args)
    elif not args.calc:
        parser.print_help()

if __name__ == "__main__":
    main()
