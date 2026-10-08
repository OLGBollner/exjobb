"""CLI command for packaging raw ZFS simulation data into a single .npz."""

import argparse
import warnings
from datetime import date
from pathlib import Path

from beyblade.parsers import parse_zfs_simulation_dataset


def build_package_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--defect-folder", type=Path, required=True, help="Root folder for one defect, e.g. ../NV_512.")
    parser.add_argument(
        "--orders",
        type=str,
        default="first,second",
        help="Comma-separated orders to package: first, second (default: both).",
    )
    parser.add_argument(
        "--pert",
        type=float,
        nargs="+",
        default=None,
        help="Perturbation scale(s) to package. Default: every pert_* folder found.",
    )
    parser.add_argument("--method", type=str, default="all", help="ZFS calculation method (all or approx).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root folder for packaged data (default: ./data).",
    )
    parser.set_defaults(func=_run)


_ORDER_DIRS = {"first": "first_order", "second": "second_order"}
_ORDER_NUM = {"first": 1, "second": 2}


def _run(args: argparse.Namespace) -> None:
    orders = [o.strip().lower() for o in args.orders.split(",") if o.strip()]
    unknown = [o for o in orders if o not in _ORDER_DIRS]
    if unknown:
        raise SystemExit(f"Unknown order(s): {', '.join(unknown)} (expected 'first' and/or 'second')")

    defect_folder = args.defect_folder.resolve()
    if not defect_folder.is_dir():
        raise SystemExit(f"Defect folder not found: {defect_folder}")

    defect = defect_folder.name.split("_")[0]
    cell_size = int(defect_folder.name.split("_")[-1])

    folder = args.output_root / f"{defect}_{cell_size}_{date.today():%Y%m%d}"
    folder.mkdir(parents=True, exist_ok=True)

    def _collect(order: str) -> dict[float, Path]:
        """Map pert_scale -> pert folder for one order."""
        order_root = defect_folder / _ORDER_DIRS[order]
        if not order_root.is_dir():
            warnings.warn(f"Order folder not found, skipping: {order_root}")
            return {}
        perts = {}
        for pert_dir in sorted(order_root.glob("pert_*")):
            pert_scale = float(pert_dir.name.split("_")[1])
            if args.pert is not None and pert_scale not in args.pert:
                continue
            perts[pert_scale] = pert_dir
        return perts

    datasets = {o: _collect(o) for o in orders}
    pert_scales = sorted(set().union(*datasets.values()))

    for pert_scale in pert_scales:
        raw = None
        for o in orders:
            if pert_scale not in datasets[o]:
                continue
            data = parse_zfs_simulation_dataset(
                sim_folder=datasets[o][pert_scale], order=_ORDER_NUM[o], calc_method=args.method
            )
            raw = data if raw is None else raw.combine(data)
        pert_folder = folder / f"pert_{pert_scale:g}"
        pert_folder.mkdir(parents=True, exist_ok=True)
        save_path = raw.save(pert_folder / raw._default_name())
        print(f"Saved raw data in: {save_path}")
