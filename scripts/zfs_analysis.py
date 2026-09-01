import argparse
from pathlib import Path

from beyblade.zfs_manager import ZFSManager
from beyblade.phonon_manager import PhononManager
from beyblade.plotter import ZFSPlotter

def main():
    parser = argparse.ArgumentParser(description="Manage and analyze ZFS phonon derivatives and plot the results.")

    # Commands for ZFS derivative calculation
    parser.add_argument("--calc", action="store_true", help="Run calculation of ZFS derivatives.")
    parser.add_argument("--order", type=int, help="Derivative order")
    parser.add_argument("-d", "--debug", action="store_true", help="Print debug information for derivatives.")

    # Commands for ZFS management
    parser.add_argument("--raw_zfs_file", type=str, nargs="+", help="Path to raw ZFS data in .npz file")
    parser.add_argument("-ph", "--phonon_file", type=str, help="Phonon data file to use.")
    parser.add_argument("--all", action="store_true", help="Use results from all_bands (for calculation).")
    parser.add_argument("--approx", action="store_true", help="Use results from defect_band_approx (for calculation).")
    
    parser.add_argument("--sim_folder", type=str, help="Folder containing simulation results for calculation.")

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
    
    zfs_manager = None

    if args.sim_folder:

        sim_folder = Path(args.sim_folder)

        calc_method, zfs_folder = ("all_bands", "ZFS_hyp") if args.all else ("defect_band_approx", "ZFS_occup")

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
        zfs_manager = ZFSManager(
            phonon_manager=phonon_manager,
            debug=args.debug
        )

        order = 1 if "first_order" in sim_folder.parent.name else 2
        zfs_manager.load_outcar_zfs_data(sim_folder=sim_folder, calc_method=calc_method, zfs_folder=zfs_folder)

        save_name = f"{zfs_manager.defect}_{zfs_manager.cell_size}_raw_zfs_data_{zfs_manager.calc_method}_{order}d.npz"

        raw_zfs_data = zfs_manager.save_data(save_name,
                                             order=order,
                                             eigen_rotation=zfs_manager.eigen_rotation,
                                             zfs_relaxed=zfs_manager.zfs_relaxed,
                                             zfs_tensors=zfs_manager.zfs_tensors,
                                             zfs_tensors_2d=zfs_manager.zfs_tensors_2d
                                             )

    elif args.raw_zfs_file:
        try:
            phonon_manager = PhononManager(data_path=args.phonon_file)
            zfs_manager = ZFSManager(phonon_manager=phonon_manager, debug=args.debug)

            zfs_manager.load_outcar_zfs_data(raw_data_path=args.raw_zfs_file)
        except Exception as e:
            raise e


    if args.calc:
        print("Starting calculation of ZFS derivatives...")
        if args.order == 3:
            generated_files = zfs_manager.process_second_order_perturbations(args.output)
        elif args.order == 1:
            print("Processing first-order perturbations...")
            generated_files = zfs_manager.process_first_order_perturbations(args.output)
        elif args.order == 2:
            print("Processing second-order perturbations...")
            generated_files = zfs_manager.process_second_order_perturbations(args.output)

    if args.data_files and zfs_manager is None:
        files_to_plot = args.data_files
    elif zfs_manager is not None:
        if args.order == 1 and args.data_files:
            files_to_plot = args.data_files
        elif args.order == 2 and not generated_files:
            files_to_plot = args.data_files
        else:
            files_to_plot = generated_files

    if files_to_plot and (args.plot or args.format):
        print("Beginning visualization...")
        plotter = ZFSPlotter()
        plotter.plot_data(files_to_plot, args)
    elif not args.calc:
        print("Finnished loading raw ZFS data.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
