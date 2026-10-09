"""CLI subcommands for the plotting functions."""

import argparse
from pathlib import Path

from beyblade.plotter import plot_run_coupling, plot_run_rates, plot_t1
from beyblade.runs_index import filter_rows, read_rows, run_dirs


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-o", "--output", type=Path, help="Output file (or directory for run plots)")
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


def _require_output(args: argparse.Namespace) -> Path:
    if args.output is None:
        raise SystemExit("error: -o/--output is required")
    return args.output


def _run_coupling(args: argparse.Namespace) -> None:
    out_dir = _require_output(args)
    plot_run_coupling(args.run_dir, out_dir, args.fmt, args.dpi, args.show)


def _run_rates(args: argparse.Namespace) -> None:
    out_dir = _require_output(args)
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
    plot_t1(inputs, output_path=_require_output(args), show=args.show, plain_name=args.plain_name)


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
