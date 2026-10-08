"""Tests for ``beyblade package``."""

from __future__ import annotations

import argparse
from unittest.mock import patch

import pytest

from beyblade.cli import package as cli_package


def _make_args(sim_folders: list[str], method: str = "all", output_root="data") -> argparse.Namespace:
    return argparse.Namespace(sim_folder=sim_folders, method=method, output_root=__import__("pathlib").Path(output_root))


class _FakeRaw:
    def __init__(self, defect, cell_size):
        self.defect = defect
        self.cell_size = cell_size
        self.combined_with: list = []

    def combine(self, other):
        self.combined_with.append(other)
        return self

    def _default_name(self):
        return f"{self.defect}_{self.cell_size}_raw_zfs_data_all_1d.npz"

    def save(self, out_path):
        out_path.write_text("fake")
        return str(out_path)


def test_package_writes_to_dated_defect_folder(tmp_path):
    raw = _FakeRaw("NV", 512)
    with patch.object(cli_package, "parse_zfs_simulation_dataset", return_value=raw):
        cli_package._run(_make_args(["/sim"], output_root=str(tmp_path)))
    assert (tmp_path / f"NV_512_{cli_package.date.today():%Y%m%d}" / "NV_512_raw_zfs_data_all_1d.npz").exists()


def test_package_hard_error_on_mixed_defects():
    with patch.object(cli_package, "parse_zfs_simulation_dataset", side_effect=[_FakeRaw("NV", 512), _FakeRaw("ClV", 128)]):
        with pytest.raises(SystemExit, match="multiple defects"):
            cli_package._run(_make_args(["/a", "/b"]))


def test_package_combines_multiple_folders():
    raws = [_FakeRaw("NV", 512), _FakeRaw("NV", 512)]
    with patch.object(cli_package, "parse_zfs_simulation_dataset", side_effect=raws):
        with patch.object(_FakeRaw, "save", return_value="p"):
            cli_package._run(_make_args(["/a", "/b"], output_root="/tmp/x"))
    assert raws[0].combined_with == [raws[1]]
