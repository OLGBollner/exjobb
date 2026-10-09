"""CLI subcommands for the plotting functions."""

import argparse
from pathlib import Path

from beyblade.convergence import plot_pert_convergence, read_conv_metadata
from beyblade.plotter import plot_run_coupling, plot_run_rates, plot_t1
from beyblade.runs_index import filter_rows, read_rows, run_dirs


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", type=Path, help="Output file (or directory for run plots); defaults to figures/<kind>/")
    parser.add_argument("--fmt", default="png", help="Image format, e.g. png or pdf")
    parser.add_argument("--dpi", type=int, default=300, help="Image resolution")
    parser.add_argument("--show", action="store_true", help="Show the figure interactively")


def build_plot_parser(subparsers: argparse._SubParsersAction) -> None:  # noqa: SLF001
    """Add the ``plot`` subcommands to a subparsers object."""
    coupling = subparsers.add_parser("coupling", help="Plot run coupling from a run directory")
    coupling.add_argument("run_dir", type=Path)
    _add_output_args(coupling)
    coupling.set_defaults(func=_run_coupling)

    rates = subparsers.add_parser("rates", help="Plot transition rates from a run directory")
    rates.add_argument("run_dir", type=Path)
    _add_output_args(rates)
    rates.set_defaults(func=_run_rates)

    t1 = subparsers.add_parser("t1", help="Plot T1 curves from one or more npz files or run directories in one figure")
    t1.add_argument("inputs", nargs="*", type=Path)
    t1.add_argument("-r", "--output-root", default="runs", help="Root folder containing runs_index.csv (default: runs)")
    t1.add_argument("--filter", action="append", metavar="FIELD=VALUE",
                    help="Resolve runs from the index, e.g. --filter method=all_bands,pert=1.0. Repeatable; comma-separated values allowed.")
    t1.add_argument("--latest", action="store_true", help="With --filter, use only the most recent matching run")
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
                f"error: {args.npz} has no defect/cell metadata; repack it with 'beyblade package --conv' or pass -o explicitly"
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


def _run_coupling(args: argparse.Namespace) -> None:
    out_dir = args.output or _figures_dir("coupling") / args.run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_run_coupling(args.run_dir, out_dir, args.fmt, args.dpi, args.show)


def _run_rates(args: argparse.Namespace) -> None:
    out_dir = args.output or _figures_dir("rates") / args.run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_run_rates(args.run_dir, out_dir, args.fmt, args.dpi, args.show)


def _run_t1(args: argparse.Namespace) -> None:
    if args.filter:
        if args.inputs:
            raise SystemExit("error: pass either explicit run paths or --filter, not both")
        rows = filter_rows(read_rows(args.output_root), _parse_filters(args.filter), latest=args.latest)
        if not rows:
            raise SystemExit(f"error: no indexed runs match in {args.output_root}/runs_index.csv")
        run_dirs_list = run_dirs(rows, args.output_root)
        inputs = [d / "t1_relaxation.npz" for d in run_dirs_list]
        missing = [i for i in inputs if not i.exists()]
        if missing:
            raise SystemExit(f"error: missing t1_relaxation.npz in: {', '.join(str(m) for m in missing)}")
    else:
        if not args.inputs:
            raise SystemExit("error: provide npz/run paths, or use --filter to select indexed runs")
        inputs = args.inputs
    out = args.output or _default_output(args, "t1", "t1")
    plot_t1(inputs, output_path=out, show=args.show, plain_name=args.plain_name)


def _parse_filters(pairs: list[str]) -> list[tuple[str, str]]:
    out = []
    for pair in pairs:
        for part in pair.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise SystemExit(f"error: --filter expects FIELD=VALUE, got '{part}'")
            field, _, value = part.partition("=")
            out.append((field, value))
    return out
