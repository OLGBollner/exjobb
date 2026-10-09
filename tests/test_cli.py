"""Tests for the beyblade CLI."""

import numpy as np
import pytest

from beyblade.cli import build_parser
import argparse
import io


def _make_t1_npz(path, defect="NV", cell=2, method="pbe", state="ms0"):
    t = np.linspace(0, 10, 50)
    pop = np.exp(-t / 5)
    np.savez(
        path,
        temperatures=t,
        t1_fit=pop,
        t1_eigenval=pop * 0.5,
        defect=defect,
        cell_size=cell,
        calc_method=method,
        init_state=state,
    )


def test_plot_t1_single_file(tmp_path, capsys):
    npz = tmp_path / "t1_relaxation.npz"
    _make_t1_npz(npz)
    out = tmp_path / "out.png"
    parser = build_parser()
    args = parser.parse_args(["plot", "t1", str(npz), "-o", str(out)])
    args.func(args)
    assert list(tmp_path.glob("out*.png"))
    assert "Saved T1 figure to:" in capsys.readouterr().out


def test_plot_t1_multiple_files_same_figure(tmp_path):
    p1 = tmp_path / "a.npz"
    p2 = tmp_path / "b.npz"
    _make_t1_npz(p1, defect="NV", cell=2)
    _make_t1_npz(p2, defect="ClV", cell=4)
    out = tmp_path / "out.png"
    parser = build_parser()
    args = parser.parse_args(["plot", "t1", str(p1), str(p2), "-o", str(out)])
    args.func(args)
    assert list(tmp_path.glob("out*.png"))


def test_plot_t1_requires_output(tmp_path):
    npz = tmp_path / "t1.npz"
    _make_t1_npz(npz)
    parser = build_parser()
    with pytest.raises(SystemExit):
        args = parser.parse_args(["plot", "t1", str(npz)])
        args.func(args)


def test_plot_t1_accepts_run_dir(tmp_path):
    run = tmp_path / "run1"
    run.mkdir()
    _make_t1_npz(run / "t1_relaxation.npz")
    out = tmp_path / "out.png"
    parser = build_parser()
    args = parser.parse_args(["plot", "t1", str(run), "-o", str(out)])
    args.func(args)
    assert list(tmp_path.glob("out*.png"))


def test_plot_t1_missing_file_errors(tmp_path):
    parser = build_parser()
    args = parser.parse_args(["plot", "t1", str(tmp_path / "missing.npz"), "-o", str(tmp_path / "o.png")])
    with pytest.raises(FileNotFoundError):
        args.func(args)


class TestRunValidation:
    """Mutually exclusive input modes for beyblade run."""

    @staticmethod
    def _parse(*argv: str) -> argparse.Namespace:
        from beyblade.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", *argv])
        return args

    def _expect_error(self, *argv: str, match: str) -> None:
        from contextlib import redirect_stderr
        from beyblade.cli import build_parser

        parser = build_parser()
        args = parser.parse_args(["run", *argv])
        err = io.StringIO()
        with pytest.raises(SystemExit), redirect_stderr(err):
            args.func(args)
        assert match in err.getvalue()

    def test_no_input_mode_errors(self) -> None:
        self._expect_error(match="one of --sim-folder")

    def test_mixed_modes_error(self) -> None:
        self._expect_error("--sim-folder", "sim", "--coupling-file", "c.npz", match="mutually exclusive")

    def test_1d_without_2d_errors(self) -> None:
        self._expect_error("--raw-zfs-file-1d", "a.npz", match="given together")

    def test_single_sim_folder_parsed(self) -> None:
        args = self._parse("--sim-folder", "sim", "--t-end", "100")
        assert args.sim_folder == ["sim"]
        assert args.t_end == 100.0
        assert args.init_state == "ms_0"

    def test_defaults(self) -> None:
        args = self._parse("--coupling-file", "c.npz")
        assert args.output_root == "runs"
        assert args.t_start == 0.0
        assert args.t_step == 10.0
        assert args.method is None
