"""CLI command for packaging raw ZFS simulation data into a single .npz."""

import argparse
import warnings
from datetime import date
from pathlib import Path

from beyblade.pack_convergence import pack_convergence
from beyblade.parsers import parse_zfs_simulation_dataset


def build_package_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--defect-folder",
        type=Path,
        required=True,
        help="Root folder for one defect, e.g. ../NV_512. With --conv: the folder containing runs/.",
    )
    parser.add_argument(
        "--conv",
        action="store_true",
        help="Package convergence runs (runs/<mode>_pert_<p>/OUTCAR) into one npz instead of raw simulation folders.",
    )
    parser.add_argument(
        "--ground-state",
        type=Path,
        default=None,
        metavar="OUTCAR",
        help="With --conv: ground-state OUTCAR; its ZFS tensor is stored as 'd0' (the Q=0 point) in the npz.",
    )
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
    defect_folder = args.defect_folder.resolve()
    if not defect_folder.is_dir():
        raise SystemExit(f"Defect folder not found: {defect_folder}")

    if getattr(args, "conv", False):
        _run_conv(args, defect_folder)
        return

    orders = [o.strip().lower() for o in args.orders.split(",") if o.strip()]
    unknown = [o for o in orders if o not in _ORDER_DIRS]
    if unknown:
        raise SystemExit(f"Unknown order(s): {', '.join(unknown)} (expected 'first' and/or 'second')")

    print(f"Defect folder: {defect_folder}")
    print(f"Orders: {', '.join(orders)} | method: {args.method}")

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
    for o in orders:
        print(f"  {o} order: {len(datasets[o])} perturbation(s) found in {defect_folder / _ORDER_DIRS[o]}")
    pert_scales = sorted(set().union(*datasets.values()))
    print(f"Perturbation scales to package: {pert_scales}")
    for pert_scale in pert_scales:
        raw = None
        for o in orders:
            if pert_scale not in datasets[o]:
                continue
            data = parse_zfs_simulation_dataset(
                sim_folder=datasets[o][pert_scale], order=_ORDER_NUM[o], calc_method=args.method, verbose=True
            )
            raw = data if raw is None else raw.combine(data)
        pert_folder = folder / f"pert_{pert_scale:g}"
        pert_folder.mkdir(parents=True, exist_ok=True)
        save_path = raw.save(pert_folder / raw._default_name())
        print(f"Saved raw data in: {save_path}")


def _run_conv(args: argparse.Namespace, defect_folder: Path) -> None:
    for opt in ("orders", "pert", "method"):
        if getattr(args, opt) != {"orders": "first,second", "pert": None, "method": "all"}[opt]:
            print(f"warning: --{opt.replace('_', '-')} is ignored with --conv")
    conv_roots = sorted(p for p in defect_folder.glob("convergence*") if p.is_dir())
    if not conv_roots:
        raise SystemExit(f"No convergence* folders found under {defect_folder}")
    for root in conv_roots:
        print(f"Convergence folder: {root}")
    defect = defect_folder.name.split("_")[0]
    cell = defect_folder.name.split("_")[-1]
    output = args.output_root / f"{defect}_{cell}_{date.today():%Y%m%d}_conv.npz"
    args.output_root.mkdir(parents=True, exist_ok=True)
    pack_convergence(conv_roots, output, ground_state=args.ground_state, defect=defect, cell=cell)
