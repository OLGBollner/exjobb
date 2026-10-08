"""Command line interface for beyblade.

Each module in this package defines one CLI domain, e.g. plotting.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    from beyblade.cli.plotting import build_plot_parser
    from beyblade.cli.run import build_run_parser

    parser = argparse.ArgumentParser(prog="beyblade", description="Beyblade command line tools")
    subparsers = parser.add_subparsers(dest="domain", required=True)

    plot_parser = subparsers.add_parser("plot", help="Plotting commands")
    build_plot_parser(plot_parser.add_subparsers(dest="command", required=True))

    build_run_parser(subparsers.add_parser("run", help="Run the end-to-end analysis pipeline"))

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
