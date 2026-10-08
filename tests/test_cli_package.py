import argparse
import warnings
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


def _patch(side_effect=_raw):
    return patch("beyblade.cli.package.parse_zfs_simulation_dataset", side_effect=side_effect)


def test_default_combines_orders_into_one_file(tmp_path):
    _tree(tmp_path)
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path))
    files = list((tmp_path / "data").rglob("*.npz"))
    assert files == [tmp_path / "data" / "NV_512_20261008" / "pert_0.025" / "NV_512_raw_zfs_data_all_bands_2d.npz"]


def test_single_order_saves_separate_file(tmp_path):
    _tree(tmp_path, orders=("first",))
    with _patch():
        _run(_make_args(tmp_path, orders="first"))
    files = list((tmp_path / "data").rglob("*.npz"))
    assert files == [tmp_path / "data" / "NV_512_20261008" / "pert_0.025" / "NV_512_raw_zfs_data_all_bands_1d.npz"]


def test_pert_filter(tmp_path):
    _tree(tmp_path, perts=(0.01, 0.025))
    with _patch():
        _run(_make_args(tmp_path, orders="first", pert=[0.025]))
    assert len(list((tmp_path / "data").rglob("*.npz"))) == 1


def test_missing_order_folder_warns(tmp_path):
    _tree(tmp_path, orders=("first",))
    with _patch():
        with pytest.warns(UserWarning):
            _run(_make_args(tmp_path))
    assert len(list((tmp_path / "data").rglob("*.npz"))) == 1
