"""CLI command for packaging raw ZFS simulation data into a single .npz."""

import argparse
from datetime import date
from pathlib import Path

from beyblade.parsers import parse_zfs_simulation_dataset


def build_package_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--sim-folder", type=str, nargs="+", required=True, help="Path(s) to VASP simulation folder(s).")
    parser.add_argument("--method", type=str, default="all", help="ZFS calculation method (all or approx).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root folder for packaged data (default: ./data).",
    )
    parser.set_defaults(func=_run)


def _run(args: argparse.Namespace) -> None:
    datasets = [
        parse_zfs_simulation_dataset(sim_folder=sf, calc_method=args.method)
        for sf in args.sim_folder
    ]

    defects = {d.defect for d in datasets}
    sizes = {d.cell_size for d in datasets}
    if len(defects) != 1 or len(sizes) != 1:
        raise SystemExit(
            "Cannot package simulations spanning multiple defects or cell sizes: "
            f"defects={sorted(str(d) for d in defects)}, cell_sizes={sorted(str(s) for s in sizes)}"
        )

    raw_data = datasets[0]
    for other in datasets[1:]:
        raw_data = raw_data.combine(other)

    folder = args.output_root / f"{raw_data.defect}_{raw_data.cell_size}_{date.today():%Y%m%d}"
    folder.mkdir(parents=True, exist_ok=True)

    save_path = raw_data.save(folder / raw_data._default_name())
    print(f"Saved raw data in: {save_path}")
