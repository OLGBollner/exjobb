"""CLI subcommands for the plotting functions."""

import argparse
from pathlib import Path

from spinlabs.cli import add_verbosity_args
from spinlabs.convergence import plot_pert_convergence, read_conv_metadata
from spinlabs.plotter import plot_run_coupling, plot_run_rates, plot_t1
from spinlabs.registry import ensure_db, find_runs, parse_orders
from spinlabs.relaxation_dynamics import find_t1_files


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file (or directory for run plots); defaults to figures/<kind>/"
    )
    parser.add_argument("--fmt", default="png", help="Image format, e.g. png or pdf")
    parser.add_argument("--dpi", type=int, default=300, help="Image resolution")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")
    add_verbosity_args(parser)


def build_plot_parser(subparsers: argparse._SubParsersAction) -> None:  # noqa: SLF001
    """Add the ``plot`` subcommands to a subparsers object."""
    coupling = subparsers.add_parser("coupling", help="Plot run coupling from a run directory")
    coupling.add_argument("run_dir", type=Path, nargs="?")
    _add_query_args(coupling)
    _add_output_args(coupling)
    coupling.set_defaults(func=_run_coupling)

    rates = subparsers.add_parser("rates", help="Plot transition rates from a run directory")
    rates.add_argument("run_dir", type=Path, nargs="?")
    _add_query_args(rates)
    _add_output_args(rates)
    rates.set_defaults(func=_run_rates)

    t1 = subparsers.add_parser("t1", help="Plot T1 curves from one or more npz files or run directories in one figure")
    t1.add_argument("inputs", nargs="*", type=Path)
    _add_query_args(t1)
    t1.add_argument("--plain-name", action="store_true", help="Do not append metadata to the output filename")
    _add_output_args(t1)
    t1.set_defaults(func=_run_t1)

    conv = subparsers.add_parser(
        "convergence", help="Plot ZFS principal components vs perturbation scale from a packed npz"
    )
    conv.add_argument("npz", type=Path, help="npz from pack_perturbation_runs.py")
    conv.add_argument("-m", "--mode", type=int, action="append", help="plot only this mode (repeatable; default: all)")
    conv.add_argument(
        "--quad", action="store_true", help="also fit y = a + b x + c x^2 and report the second-order derivative 2c"
    )
    conv.add_argument(
        "--deviation",
        action="store_true",
        help="plot D(Q) - D(Q=0) instead of raw D, so the response fills the y-range",
    )
    conv.add_argument(
        "--csv", type=Path, default=None, help="write slopes (and quad coefficients with --quad) to this csv"
    )
    conv.add_argument(
        "--drop-anchor",
        action="store_true",
        help="exclude the Q=0 anchor from fits and residuals (curves still extrapolate to Q=0)",
    )
    _add_output_args(conv)
    conv.set_defaults(func=_run_convergence)


def _run_convergence(args: argparse.Namespace) -> None:
    meta = read_conv_metadata(args.npz)
    stem = "_".join(str(meta[k]) for k in ("defect", "cell") if meta.get(k) is not None)
    method = meta.get("method") or meta.get("method_key")
    if method:
        stem = f"{stem}_{method}" if stem else str(method)
    if not stem:
        if not args.output:
            raise SystemExit(
                f"error: {args.npz} has no defect/cell metadata; "
                "repack it with 'spinlabs package --conv' or pass -o explicitly"
            )
        stem = args.npz.stem
    out = args.output or _default_output(args, "convergence", stem)
    plot_pert_convergence(
        args.npz,
        output_path=out,
        mode=args.mode,
        quad=args.quad,
        deviation=args.deviation,
        csv_path=args.csv,
        show=args.show,
        drop_anchor=args.drop_anchor,
    )


def _figures_dir(kind: str) -> Path:
    return Path("figures") / kind


def _default_output(args: argparse.Namespace, kind: str, stem: str) -> Path:
    """Default output file under figures/<kind>/ when -o is omitted."""
    p = _figures_dir(kind) / f"{stem}.{args.fmt}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _explicit_inputs(args: argparse.Namespace) -> list[Path]:
    """Explicit run paths given on the command line, if any."""
    if hasattr(args, "inputs"):
        return list(args.inputs)
    run_dir = getattr(args, "run_dir", None)
    return [run_dir] if run_dir else []


def _resolve_query_runs(args: argparse.Namespace) -> list[Path]:
    """Completed run dirs in the registry matching the --query filters."""
    if _explicit_inputs(args):
        raise SystemExit("error: pass either explicit run paths or --query, not both")
    conn = ensure_db()
    dirs = find_runs(conn, _query_filters(args), latest=args.latest, output_root=args.output_root)
    if not dirs:
        raise SystemExit("error: no completed runs in the registry match the query")
    return dirs


def _resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    """Run directories to plot: explicit paths, or registry matches via --query."""
    if args.query:
        return _resolve_query_runs(args)
    provided = _explicit_inputs(args)
    if not provided:
        raise SystemExit("error: provide a run directory, or use --query to select registry runs")
    return [p if p.is_dir() else p.parent for p in provided]


def _run_coupling(args: argparse.Namespace) -> None:
    for run_dir in _resolve_run_dirs(args):
        out_dir = args.output / run_dir.name if args.output else _figures_dir("coupling") / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_run_coupling(run_dir, out_dir, args.fmt, args.dpi, args.show)


def _run_rates(args: argparse.Namespace) -> None:
    for run_dir in _resolve_run_dirs(args):
        out_dir = args.output / run_dir.name if args.output else _figures_dir("rates") / run_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_run_rates(run_dir, out_dir, args.fmt, args.dpi, args.show)


def _add_query_args(parser: argparse.ArgumentParser) -> None:
    """Registry-query filters shared by all plot subcommands (same as 'run --query')."""
    parser.add_argument(
        "--query",
        action="store_true",
        help="Plot the completed runs in the registry matching the filters "
        "(--defect/--cell-size/--method/--order/--pert-scale). Same syntax as 'run --query'",
    )
    parser.add_argument("--defect", help="Filter by defect name (e.g. NV, ClV)")
    parser.add_argument("--cell-size", type=int, help="Filter by supercell size (e.g. 64, 128)")
    parser.add_argument(
        "--method", choices=["all", "approx"], help="Filter by calculation method: 'all' for all_bands, 'approx' for defect_band_approx"
    )
    parser.add_argument(
        "--order",
        nargs="+",
        type=int,
        choices=[1, 2, 12],
        help="Filter by perturbation order(s), e.g. --order 1 2 (12 is the legacy alias for 1 2)",
    )
    parser.add_argument("--pert-scale", type=float, help="Filter by perturbation scale (e.g. 0.025)")
    parser.add_argument("--latest", action="store_true", help="With --query, use only the most recent matching run")
    parser.add_argument(
        "-r", "--output-root", default="runs",
        help="Root folder holding run directories (default: runs); used to resolve runs queried from the registry",
    )


def _query_filters(args: argparse.Namespace) -> list[tuple[str, str]]:
    """--query flag values as (column, value) filters for registry.find_runs."""
    out = []
    if args.defect is not None:
        out.append(("defect", args.defect))
    if args.cell_size is not None:
        out.append(("cell", args.cell_size))
    if args.method is not None:
        out.append(("method", {"all": "all_bands", "approx": "defect_band_approx"}[args.method]))
    if args.order is not None:
        out.append(("order", parse_orders(args.order)))
    if args.pert_scale is not None:
        out.append(("pert", args.pert_scale))
    return out


def _run_t1(args: argparse.Namespace) -> None:
    if args.query:
        dirs = _resolve_query_runs(args)
        missing = [d for d in dirs if not find_t1_files(d)]
        if missing:
            raise SystemExit(f"error: missing t1_relaxation*.npz in: {', '.join(str(m) for m in missing)}")
        inputs = [f for d in dirs for f in find_t1_files(d)]
    else:
        if not args.inputs:
            raise SystemExit("error: provide npz/run paths, or use --query to select registry runs")
        inputs = args.inputs
    out = args.output or _default_output(args, "t1", "t1")
    plot_t1(inputs, output_path=out, show=args.show, plain_name=args.plain_name)



