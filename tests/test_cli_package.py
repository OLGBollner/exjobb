import argparse
from unittest.mock import patch

import pytest

from beyblade.cli.package import _run
from beyblade.models import RawZFSData


def _raw(order=1, pert=0.025):
    return RawZFSData(
        defect="NV",
        cell_size=512,
        pert_scale=pert,
        calc_method="all_bands",
        order=order,
    )


def _make_args(tmp_path, orders="first,second", pert=None):
    return argparse.Namespace(
        defect_folder=tmp_path / "NV_512",
        orders=orders,
        pert=pert,
        method="all",
        output_root=tmp_path / "data",
    )


def _tree(tmp_path, orders=("first", "second"), perts=(0.025,)):
    for o in orders:
        for p in perts:
            (tmp_path / "NV_512" / f"{o}_order" / f"pert_{p:g}").mkdir(parents=True)


def test_saves_per_order_and_pert(tmp_path):
    _tree(tmp_path)
    with patch(
        "beyblade.cli.package.parse_zfs_simulation_dataset",
        side_effect=lambda **kw: _raw(order=kw["order"], pert=float(kw["sim_folder"].name.split("_")[1])),
    ):
        _run(_make_args(tmp_path))
    base = tmp_path / "data" / "NV_512_20261008" / "pert_0.025"
    assert (base / "NV_512_raw_zfs_data_all_bands_1d.npz").exists()
    assert (base / "NV_512_raw_zfs_data_all_bands_2d.npz").exists()


def test_orders_filter(tmp_path):
    _tree(tmp_path, orders=("first",))
    with patch("beyblade.cli.package.parse_zfs_simulation_dataset", return_value=_raw()):
        _run(_make_args(tmp_path, orders="first"))
    assert len(list((tmp_path / "data").rglob("*.npz"))) == 1


def test_pert_filter(tmp_path):
    _tree(tmp_path, perts=(0.01, 0.025))
    with patch("beyblade.cli.package.parse_zfs_simulation_dataset", return_value=_raw()):
        _run(_make_args(tmp_path, pert=[0.025]))
    assert len(list((tmp_path / "data").rglob("*.npz"))) == 1


def test_missing_order_folder_warns(tmp_path):
    _tree(tmp_path, orders=("first",))
    with patch("beyblade.cli.package.parse_zfs_simulation_dataset", return_value=_raw()):
        with pytest.warns(UserWarning):
            _run(_make_args(tmp_path))
    assert len(list((tmp_path / "data").rglob("*.npz"))) == 1
