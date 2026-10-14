import argparse
from datetime import date
from pathlib import Path

import numpy as np
from unittest.mock import patch

import pytest

from spinlabs.cli.package import _run, _run_conv
from spinlabs.models import PerturbationEntry, RawZFSData, ZFSTensor


def _raw(order=1, pert=0.025, calc_method=None, **_kw):
    kwargs = dict(
        defect="NV",
        cell_size=512,
        pert_scale=pert,
        calc_method={"all": "all_bands", "approx": "defect_band_approx"}.get(calc_method, calc_method or "all_bands"),
        order=order,
    )
    if order == 2:
        # Non-empty second order so combined_order() resolves to 2,
        # matching real second-order data.
        kwargs["second_order"] = {
            (0, 0): PerturbationEntry(
                order=2,
                mode_indices=(0, 0),
                amplitude=(0.025, 0.025),
                zfs_tensor=ZFSTensor(matrix=np.zeros((3, 3))),
            )
        }
    return RawZFSData(**kwargs)


def _make_args(tmp_path, orders="first,second", pert=None):
    return argparse.Namespace(
        defect_folder=[tmp_path / "NV_512"],
        orders=orders,
        pert=pert,
        method=["all"],
        output_root=tmp_path / "data",
    )


def _tree(tmp_path, orders=("first", "second"), perts=(0.025,)):
    for o in orders:
        for p in perts:
            (tmp_path / "NV_512" / f"{o}_order" / f"pert_{p:g}").mkdir(parents=True)


def _patch(side_effect=_raw):
    return patch("spinlabs.cli.package.parse_zfs_simulation_dataset", side_effect=side_effect)


def test_default_combines_orders_into_one_file(tmp_path):
    _tree(tmp_path)
    with _patch(lambda **kw: _raw(order=kw["order"])):
        _run(_make_args(tmp_path))
    files = list((tmp_path / "data").rglob("*.npz"))
    run_dir = (
        tmp_path / "data" / "NV_512" / "all_bands" / date.today().strftime("%Y%m%d")
    )
    assert files == [run_dir / "pert_0.025" / "NV_512_raw_zfs_data_all_bands_2d.npz"]
    index = (tmp_path / "data" / "data_index.csv").read_text()
    assert "NV,512,all_bands,-1,0.025,2" in index


def test_single_order_saves_separate_file(tmp_path):
    _tree(tmp_path, orders=("first",))
    with _patch():
        _run(_make_args(tmp_path, orders="first"))
    files = list((tmp_path / "data").rglob("*.npz"))
    run_dir = (
        tmp_path / "data" / "NV_512" / "all_bands" / date.today().strftime("%Y%m%d")
    )
    assert files == [run_dir / "pert_0.025" / "NV_512_raw_zfs_data_all_bands_1d.npz"]


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


# ── --conv --ground-state (flag-only auto-discovery) ─────────────────────────

def _outcar(path, d_diag, energy=-10.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f" free  energy   TOTEN  =       {energy:.8f} eV\n\n"
        " Spin-spin contribution to zero-field splitting tensor (MHz)\n"
        " -------------------------------------------------------------\n"
        "      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz\n"
        " -------------------------------------------------------------\n"
        f"   {d_diag[0]:.2f}   {d_diag[1]:.2f}   {d_diag[2]:.2f}      0.00      0.00      0.00\n"
        " -------------------------------------------------------------\n",
        encoding="utf-8",
    )


def _conv_args(tmp_path, ground_state):
    return argparse.Namespace(
        defect_folder=tmp_path / "NV_512",
        output_root=tmp_path / "data",
        ground_state=ground_state,
    )


def _conv_tree(tmp_path):
    root = tmp_path / "NV_512" / "convergence_test_all"
    for p in (0.025, 0.05):
        _outcar(root / "runs" / f"0_pert_{p:g}" / "OUTCAR", (1.0, 1.0, 2.0))
    return root


def test_conv_ground_state_flag_auto_all(tmp_path, capsys):
    _conv_tree(tmp_path)
    _outcar(tmp_path / "NV_512" / "ZFS_hyp" / "OUTCAR", (1.0, 1.0, 4.0))
    _run_conv(args=_conv_args(tmp_path, Path("__auto__")), defect_folder=tmp_path / "NV_512")
    out = next((tmp_path / "data").rglob("NV_512_*_conv_test_all.npz"))
    data = np.load(out, allow_pickle=True)
    assert data["d0"].shape == (3, 3)
    assert abs(data["d0"][2, 2] - 4.0) < 1e-9
    assert "ground state:" in capsys.readouterr().out
    index = (tmp_path / "data" / "data_index.csv").read_text()
    assert "NV,512,all_bands,-1,-1,1+2" in index


def test_conv_ground_state_flag_auto_occup(tmp_path):
    root = tmp_path / "NV_512" / "convergence_test_occup"
    for p in (0.025,):
        _outcar(root / "runs" / f"0_pert_{p:g}" / "OUTCAR", (1.0, 1.0, 2.0))
    _outcar(tmp_path / "NV_512" / "ZFS_occup" / "OUTCAR", (1.0, 1.0, 3.0))
    _run_conv(args=_conv_args(tmp_path, Path("__auto__")), defect_folder=tmp_path / "NV_512")
    out = next((tmp_path / "data").rglob("NV_512_*_conv_test_occup.npz"))
    data = np.load(out, allow_pickle=True)
    assert abs(data["d0"][2, 2] - 3.0) < 1e-9


def test_conv_ground_state_missing_warns(tmp_path, capsys):
    _conv_tree(tmp_path)
    _run_conv(args=_conv_args(tmp_path, Path("__auto__")), defect_folder=tmp_path / "NV_512")
    out = next((tmp_path / "data").rglob("NV_512_*_conv_*.npz"))
    data = np.load(out, allow_pickle=True)
    assert not any(k == "d0" for k in data.files)
    assert "no ground-state OUTCAR" in capsys.readouterr().err


def test_conv_ground_state_explicit_path(tmp_path):
    _conv_tree(tmp_path)
    _outcar(tmp_path / "elsewhere" / "OUTCAR", (1.0, 1.0, 5.0))
    _run_conv(args=_conv_args(tmp_path, tmp_path / "elsewhere" / "OUTCAR"), defect_folder=tmp_path / "NV_512")
    out = next((tmp_path / "data").rglob("NV_512_*_conv_*.npz"))
    data = np.load(out, allow_pickle=True)
    assert abs(data["d0"][2, 2] - 5.0) < 1e-9


def test_multiple_folders_and_methods(tmp_path):
    _tree(tmp_path)
    clv = tmp_path / "ClV_128"
    clv.mkdir()
    (clv / "first_order" / "pert_0.025").mkdir(parents=True)
    args = _make_args(tmp_path)
    args.defect_folder = [tmp_path / "NV_512", clv]
    args.method = ["all", "approx"]
    with _patch(lambda **kw: _raw(order=kw["order"], calc_method=kw["calc_method"])):
        _run(args)
    day = date.today().strftime("%Y%m%d")
    assert (tmp_path / "data" / "NV_512" / "all_bands" / day / "pert_0.025" / "NV_512_raw_zfs_data_all_bands_2d.npz").exists()
    assert (tmp_path / "data" / "NV_512" / "defect_band_approx" / day / "pert_0.025" / "NV_512_raw_zfs_data_defect_band_approx_2d.npz").exists()
    assert (tmp_path / "data" / "ClV_128" / "all_bands" / day / "pert_0.025" / "NV_512_raw_zfs_data_all_bands_1d.npz").exists()
    index = (tmp_path / "data" / "data_index.csv").read_text().strip().splitlines()[1:]
    assert len(index) == 4
    assert "ClV,128,all_bands,1,0.025,1" in index[2]


def test_missing_pert_warns_and_skips(tmp_path):
    _tree(tmp_path, perts=(0.025,))
    args = _make_args(tmp_path)
    args.pert = [0.025, 0.1]
    with _patch():
        with pytest.warns(UserWarning, match="0.1"):
            _run(args)
    files = list((tmp_path / "data").rglob("*.npz"))
    assert len(files) == 1


def test_register_only(tmp_path):
    from spinlabs.cli.package import _register_only
    from spinlabs.registry import ensure_db

    root = tmp_path / "data"
    day = date.today().strftime("%Y%m%d")
    d1 = root / "NV_512" / "all_bands" / day / "pert_0.1"
    d1.mkdir(parents=True)
    np.savez(
        d1 / "NV_512_raw_zfs_data_all_bands_12d.npz",
        defect="NV", cell_size=512, pert_scale=0.1, calc_method="all_bands", order=12,
        first_order=np.array({}, dtype=object),
    )
    (root / "stray.npz").write_bytes(b"")  # junk with no metadata

    with pytest.warns(UserWarning, match="stray.npz"):
        _register_only(root)

    conn = ensure_db()
    rows = conn.execute(
        "SELECT path, kind, defect, cell_size, calc_method, pert_order, pert_scale FROM files"
    ).fetchall()
    assert len(rows) == 1
    path, kind, defect, cell, method, order, pert = rows[0]
    assert kind == "packaged_zfs"
    assert (defect, cell, method, order, pert) == ("NV", 512, "all_bands", "[1, 2]", 0.1)
    assert path.endswith("NV_512_raw_zfs_data_all_bands_12d.npz")
