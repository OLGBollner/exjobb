"""CLI subcommands for the plotting functions."""

import argparse
from pathlib import Path

from beyblade.plotter import plot_run_coupling, plot_run_rates, plot_t1


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

    t1 = subparsers.add_parser(
        "t1", help="Plot T1 curves from one or more npz files or run directories in one figure"
    )
    t1.add_argument("inputs", nargs="+", type=Path)
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
    plot_t1(args.inputs, output_path=_require_output(args), show=args.show, plain_name=args.plain_name)
