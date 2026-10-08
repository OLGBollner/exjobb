"""CLI subcommand for the end-to-end analysis pipeline."""

import argparse

from beyblade.pipeline import run_full_pipeline


def build_run_parser(parser: argparse.ArgumentParser) -> None:
    """Configure ``parser`` as the ``run`` subcommand."""

    inputs = parser.add_argument_group("input mode (exactly one)")
    inputs.add_argument("--sim-folder", nargs="+", metavar="DIR", help="Path to VASP simulation folder(s)")
    inputs.add_argument("--raw-zfs-file", nargs="+", metavar="FILE", help="Path to raw ZFS .npz dataset file(s)")
    inputs.add_argument("--raw-zfs-file-1d", metavar="FILE", help="Raw ZFS 1st-order .npz (combined 1d+2d mode)")
    inputs.add_argument("--raw-zfs-file-2d", metavar="FILE", help="Raw ZFS 2nd-order .npz (combined 1d+2d mode)")
    inputs.add_argument("--coupling-file", metavar="FILE", help="Path to pre-computed spin-phonon coupling .npz")

    opt = parser.add_argument_group("options")
    opt.add_argument("-ph", "--phonon-file", metavar="FILE", help="phonopy.yaml or phonon_data.npz")
    opt.add_argument("--two-phonon", metavar="FILE", help="Two-phonon Raman .npz file (optional)")
    opt.add_argument(
        "--method",
        choices=["all", "approx"],
        help="Calculation method: 'all' for all_bands (ZFS_hyp), 'approx' for defect_band_approx (ZFS_occup). "
        "Inferred from the input file name when omitted.",
    )
    opt.add_argument("--order", type=int, choices=[1, 2], help="Perturbation order (1 or 2)")
    opt.add_argument("--pert-scale", type=float, help="Override perturbation scale (e.g. 0.025)")
    opt.add_argument("--defect", help="Override defect name (e.g. NV, ClV)")
    opt.add_argument("--cell-size", type=int, help="Override supercell size (e.g. 64, 128)")
    opt.add_argument("--t-start", type=float, default=0.0, help="Start temperature in K (default: 0.0)")
    opt.add_argument("--t-end", type=float, default=300.0, help="End temperature in K (default: 300.0)")
    opt.add_argument("--t-step", type=float, default=10.0, help="Temperature step in K (default: 10.0)")
    opt.add_argument(
        "--temperatures", type=float, nargs="+", metavar="T", help="Explicit temperature grid in K (overrides --t-*)"
    )
    opt.add_argument(
        "--init-state", choices=["ms_0", "ms_1", "ms_-1"], default="ms_0", help="Initial spin state (default: ms_0)"
    )
    opt.add_argument(
        "-o", "--output-root", default="runs", help="Root folder for output run directories (default: runs)"
    )
    opt.add_argument("--run-name", help="Custom run folder name; an increment index is appended if it exists")
    opt.add_argument("--plot", action="store_true", help="Generate plots inside <run_dir>/figures/")
    opt.add_argument("-d", "--debug", action="store_true", help="Print debug details during derivative calculations")

    parser.set_defaults(func=_run)


_METHOD_MAP = {"all": ("all_bands", "ZFS_hyp"), "approx": ("defect_band_approx", "ZFS_occup")}


def _infer_method(args: argparse.Namespace) -> str:
    """Return 'all' or 'approx', inferring from input file/folder names when --method is omitted."""
    if args.method:
        return args.method
    names = " ".join(
        str(p)
        for p in (
            *(args.sim_folder or []),
            *(args.raw_zfs_file or []),
            args.raw_zfs_file_1d,
            args.raw_zfs_file_2d,
            args.coupling_file,
        )
        if p
    ).lower()
    has_all = "all_bands" in names or "zfs_hyp" in names
    has_approx = "defect_band_approx" in names or "approx" in names or "zfs_occup" in names
    if has_all == has_approx:
        raise SystemExit(
            "could not infer --method from input names; pass --method all or --method approx"
            if not has_all
            else "input names match both methods; pass --method all or --method approx explicitly"
        )
    return "all" if has_all else "approx"


def _validate(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Enforce exactly one primary input mode."""
    modes = [
        ("--sim-folder", args.sim_folder),
        ("--raw-zfs-file", args.raw_zfs_file),
        (
            "--raw-zfs-file-1d/2d",
            [args.raw_zfs_file_1d, args.raw_zfs_file_2d] if (args.raw_zfs_file_1d or args.raw_zfs_file_2d) else None,
        ),
        ("--coupling-file", args.coupling_file),
    ]
    given = [name for name, val in modes if val]
    if not given:
        parser.error("one of --sim-folder, --raw-zfs-file, --raw-zfs-file-1d/2d, --coupling-file is required")
    if len(given) > 1:
        parser.error(f"input modes are mutually exclusive, got: {', '.join(given)}")
    if (args.raw_zfs_file_1d is None) != (args.raw_zfs_file_2d is None):
        parser.error("--raw-zfs-file-1d and --raw-zfs-file-2d must be given together")


def _run(args: argparse.Namespace) -> None:
    _validate(argparse.ArgumentParser(prog="beyblade run"), args)
    run_full_pipeline(
        sim_folder=args.sim_folder[0] if args.sim_folder and len(args.sim_folder) == 1 else args.sim_folder,
        raw_zfs_file=args.raw_zfs_file[0] if args.raw_zfs_file and len(args.raw_zfs_file) == 1 else args.raw_zfs_file,
        raw_zfs_file_1d=args.raw_zfs_file_1d,
        raw_zfs_file_2d=args.raw_zfs_file_2d,
        coupling_file=args.coupling_file,
        phonon_file=args.phonon_file,
        two_phonon_file=args.two_phonon,
        calc_method=_METHOD_MAP[_infer_method(args)][0],
        zfs_folder=_METHOD_MAP[_infer_method(args)][1],
        order=args.order,
        pert_scale=args.pert_scale,
        defect=args.defect,
        cell_size=args.cell_size,
        t_start=args.t_start,
        t_end=args.t_end,
        t_step=args.t_step,
        temperatures=args.temperatures,
        init_state=args.init_state,
        output_root=args.output_root,
        run_name=args.run_name,
        save_plots=args.plot,
        show_plots=False,
        debug=args.debug,
    )
