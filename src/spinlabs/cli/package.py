"""CLI command for packaging raw ZFS simulation data into a single .npz."""

import argparse
import re
import sys
import warnings
from datetime import date
from pathlib import Path

from spinlabs.data_index import append_row as append_data_row
from spinlabs.models import _normalize_calc_method
from spinlabs.registry import ensure_db, register_file
from spinlabs.pack_convergence import pack_convergence
from spinlabs.parsers import parse_zfs_simulation_dataset, split_defect_folder_name


def build_package_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--defect-folder",
        type=Path,
        nargs="+",
        metavar="FOLDER",
        help="Root folder(s) for one defect each, e.g. ../NV_512 ../ClV_128. With --conv: exactly one folder containing runs/.",
    )
    parser.add_argument(
        "--register-only",
        type=Path,
        metavar="DATA_PATH",
        default=None,
        help="Don't package anything: scan DATA_PATH for packaged .npz files and add them to the registry.",
    )
    parser.add_argument(
        "--conv",
        action="store_true",
        help="Package convergence runs (runs/<mode>_pert_<p>/OUTCAR) into one npz instead of raw simulation folders.",
    )
    parser.add_argument(
        "--ground-state",
        type=Path,
        nargs="?",
        const=_GS_AUTO,
        default=None,
        metavar="OUTCAR",
        help=(
            "With --conv: ground-state OUTCAR; its ZFS tensor is stored as 'd0' (the Q=0 point) in the npz. "
            "Flag without a path: auto-discover <defect-folder>/ZFS_hyp/OUTCAR (all) or ZFS_occup/OUTCAR (occup)."
        ),
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
    parser.add_argument(
        "--method",
        type=str,
        nargs="+",
        default=["all"],
        metavar="METHOD",
        help="ZFS calculation method(s) to package: all or approx (default: all).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data"),
        help="Root folder for packaged data (default: ./data).",
    )
    parser.set_defaults(func=_run)


_ORDER_DIRS = {"first": "first_order", "second": "second_order"}
_ORDER_NUM = {"first": 1, "second": 2}
_GS_AUTO = Path("__auto__")  # --ground-state given without a path: auto-discover


def _run(args: argparse.Namespace) -> None:
    if getattr(args, "register_only", None) is not None:
        _register_only(args.register_only)
        return

    if not args.defect_folder:
        raise SystemExit("--defect-folder is required (unless --register-only is given)")
    defect_folders = [f.resolve() for f in args.defect_folder]
    for f in defect_folders:
        if not f.is_dir():
            raise SystemExit(f"Defect folder not found: {f}")

    if getattr(args, "conv", False):
        if len(defect_folders) > 1:
            raise SystemExit("--conv takes exactly one --defect-folder; got " + str(len(defect_folders)))
        _run_conv(args, defect_folders[0])
        return

    methods = args.method
    for defect_folder in defect_folders:
        _run_raw(args, defect_folder, methods)


def _run_raw(args: argparse.Namespace, defect_folder: Path, methods: list[str]) -> None:
    orders = [o.strip().lower() for o in args.orders.split(",") if o.strip()]
    unknown = [o for o in orders if o not in _ORDER_DIRS]
    if unknown:
        raise SystemExit(f"Unknown order(s): {', '.join(unknown)} (expected 'first' and/or 'second')")

    defect, cell_size = split_defect_folder_name(defect_folder.name)
    if cell_size is None:
        raise SystemExit(f"Cell size not found in defect folder name: {defect_folder.name} (expected e.g. NV_512)")

    print(f"Defect folder: {defect_folder}")
    date_dir = date.today().strftime("%Y%m%d")

    def _collect(order: str) -> dict[float, Path]:
        """Map pert_scale -> pert folder for one order, filtered to the requested scales."""
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

    for method in methods:
        calc_method = _normalize_calc_method(method)
        folder = args.output_root / f"{defect}_{cell_size}" / calc_method / date_dir
        folder.mkdir(parents=True, exist_ok=True)

        datasets = {o: _collect(o) for o in orders}
        for o in orders:
            print(f"  {o} order: {len(datasets[o])} perturbation(s) found in {defect_folder / _ORDER_DIRS[o]}")
        pert_scales = sorted(set().union(*datasets.values()))
        if args.pert is not None:
            missing = [p for p in args.pert if p not in pert_scales]
            for p in missing:
                warnings.warn(f"Perturbation scale {p:g} not found in {defect_folder.name}; skipping")
        if not pert_scales:
            warnings.warn(f"Nothing to package for {defect_folder.name} ({calc_method}); skipping")
            continue
        print(f"Perturbation scales to package: {pert_scales}")
        for pert_scale in pert_scales:
            raw = None
            for o in orders:
                if pert_scale not in datasets[o]:
                    continue
                data = parse_zfs_simulation_dataset(
                    sim_folder=datasets[o][pert_scale], order=_ORDER_NUM[o], calc_method=method, verbose=True
                )
                raw = data if raw is None else raw.combine(data)
            pert_folder = folder / f"pert_{pert_scale:g}"
            pert_folder.mkdir(parents=True, exist_ok=True)
            save_path = raw.save(pert_folder / raw._default_name())
            register_file(
                ensure_db(),
                save_path,
                "packaged_zfs",
                defect=defect,
                cell_size=cell_size,
                calc_method=calc_method,
                pert_order=([1, 2] if raw.has_first_order and raw.has_second_order else [_ORDER_NUM[orders[0]]]),
                pert_scale=pert_scale,
            )
            append_data_row(
                args.output_root,
                defect=defect,
                cell=cell_size,
                method=calc_method,
                order=None if raw.combined_order == 2 else _ORDER_NUM[orders[0]],
                pert=pert_scale,
                n_dims=(raw.combined_order, ),
                npz_files=(Path(save_path).name, ),
                data_dir=pert_folder,
            )
            print(f"Saved raw data in: {save_path}")


def _register_only(data_path: Path) -> None:
    """Register already-packaged .npz files under data_path without re-packaging."""
    import numpy as np

    from spinlabs.models import RawZFSData

    data_path = data_path.resolve()
    if not data_path.is_dir():
        raise SystemExit(f"Data path not found: {data_path}")

    conn = ensure_db()
    registered, skipped = 0, 0
    for npz_path in sorted(data_path.rglob("*.npz")):
        try:
            with np.load(str(npz_path), allow_pickle=True) as data:
                keys = set(data.files)
                required = {f.name for f in RawZFSData.SCHEMA if not f.optional} | {
                    "defect",
                    "cell_size",
                    "order",
                }
                if not required.issubset(keys):
                    warnings.warn(f"Not a packaged raw ZFS file, skipping: {npz_path.name}")
                    skipped += 1
                    continue
                defect = str(data["defect"])
                cell_size = int(data["cell_size"])
                pert_scale = float(data["pert_scale"])
                calc_method = _normalize_calc_method(str(data["calc_method"]))
                order = int(data["order"])
                orders = [1, 2] if order == 12 else [order]
        except Exception as exc:  # unreadable npz: report and move on
            warnings.warn(f"Could not read {npz_path.name}: {exc}")
            skipped += 1
            continue
        register_file(
            conn,
            npz_path,
            "packaged_zfs",
            defect=defect,
            cell_size=cell_size,
            calc_method=calc_method,
            pert_order=orders,
            pert_scale=pert_scale,
        )
        print(f"Registered: {npz_path} ({defect}_{cell_size}, {calc_method}, order {orders}, pert {pert_scale:g})")
        registered += 1
    print(f"\nDone: {registered} registered, {skipped} skipped")


def _auto_ground_state(defect_folder: Path, method: str | None) -> Path | None:
    """Flag-only --ground-state: pick ZFS_hyp (all) or ZFS_occup (occup) OUTCAR."""
    if method == "all":
        zfs_folder = "ZFS_hyp"
    elif method in ("occup", "approx"):
        zfs_folder = "ZFS_occup"
    else:
        print(
            "warning: cannot infer ground-state method from folder suffix "
            "(neither 'all' nor 'occup'); skipping ground state",
            file=sys.stderr,
        )
        return None
    candidate = defect_folder / zfs_folder / "OUTCAR"
    if candidate.exists():
        print(f"ground state: {candidate}")
        return candidate
    print(f"warning: no ground-state OUTCAR at {candidate}; skipping ground state", file=sys.stderr)
    return None


def _run_conv(args: argparse.Namespace, defect_folder: Path) -> None:
    # _run_conv warns when a --conv run passes options that only apply to raw
    # packaging. These defaults MUST mirror the argparse defaults below; if
    # one changes, change both, or a matching user value silently skips the
    # warning.
    defaults = {"orders": "first,second", "pert": None, "method": ["all"]}
    for opt in ("orders", "pert", "method"):
        if getattr(args, opt, defaults[opt]) != defaults[opt]:
            print(f"warning: --{opt.replace('_', '-')} is ignored with --conv")
    conv_roots = sorted(p for p in defect_folder.glob("convergence*") if p.is_dir())
    if not conv_roots:
        raise SystemExit(f"No convergence* folders found under {defect_folder}")
    method_map = {"all": "all", "occup": "approx"}
    for root in conv_roots:
        print(f"Convergence folder: {root}")
        suffix = re.sub(r"^convergence[._-]*", "", root.name).lstrip("_-") or "conv"
        method = next((v for k, v in method_map.items() if k in suffix), None)
        defect, cell = split_defect_folder_name(defect_folder.name)
        stem = f"{defect}_{cell}_{date.today():%Y%m%d}_conv_{suffix}"
        calc_method = _normalize_calc_method(method) if method else method
        date_dir = date.today().strftime("%Y%m%d")
        leaf = args.output_root / f"{defect}_{cell}" / (calc_method or "conv") / date_dir
        leaf.mkdir(parents=True, exist_ok=True)
        output = leaf / f"{stem}.npz"
        ground_state = args.ground_state
        if ground_state == _GS_AUTO:
            ground_state = _auto_ground_state(defect_folder, method)
        pack_convergence(
            root, output, ground_state=ground_state, defect=defect, cell=cell, method=method, method_key=suffix
        )
        register_file(
            ensure_db(), output, "convergence", defect=defect, cell_size=cell, calc_method=calc_method
        )
        append_data_row(
            args.output_root,
            defect=defect,
            cell=cell,
            method=calc_method or "conv",
            n_dims=(1, 2),
            npz_files=(output.name, ),
            data_dir=leaf,
        )
