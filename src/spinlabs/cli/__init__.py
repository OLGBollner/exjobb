"""Command line interface for spinlabs.

Each module in this package defines one CLI domain, e.g. plotting.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    from spinlabs.cli.package import build_package_parser
    from spinlabs.cli.plotting import build_plot_parser
    from spinlabs.cli.run import build_run_parser

    parser = argparse.ArgumentParser(prog="spinlabs", description="SpinLabs command line tools")
    subparsers = parser.add_subparsers(dest="domain", required=True)

    plot_parser = subparsers.add_parser("plot", help="Plotting commands")
    build_plot_parser(plot_parser.add_subparsers(dest="command", required=True))

    build_run_parser(subparsers.add_parser("run", help="Run the end-to-end analysis pipeline"))

    build_package_parser(subparsers.add_parser("scan", help="Scan/package raw ZFS simulation data into a single .npz"))

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
