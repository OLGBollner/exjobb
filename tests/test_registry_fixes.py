"""Regression tests for PR #50 review fixes.

Covers:
- find_packaged combined-order matching (order=12 must match [1, 2] rows).
- calc_method filtering in find_packaged.
- CLI `run --method all` normalization to 'all_bands'.
- Pipeline must not downgrade a packaged_zfs row to raw_zfs after a run.
- --order 2/12 without second-order data raises instead of silently degrading.
"""
import argparse
import numpy as np
import pytest

from spinlabs.models import PhononSpectrum, RawZFSData, ZFSTensor
from spinlabs.pipeline import run_full_pipeline
from spinlabs.registry import ensure_db, find_packaged, register_file


def _register(conn, path, *, kind="packaged_zfs", orders, method="all_bands", pert=0.025):
    path.write_bytes(b"placeholder")  # file must exist for find_packaged
    return register_file(
        conn,
        path,
        kind,
        defect="NV",
        cell_size=512,
        calc_method=method,
        pert_order=orders,
        pert_scale=pert,
    )


def test_find_packaged_combined_order_matches_both_rows(tmp_path):
    """order=12 (combined) must match rows storing [1] and [1, 2]; order=1 both too."""
    conn = ensure_db()
    f1d = tmp_path / "NV_512_raw_zfs_data_all_bands_1d.npz"
    f2d = tmp_path / "NV_512_raw_zfs_data_all_bands_2d.npz"
    fcomb = tmp_path / "NV_512_raw_zfs_data_all_bands_12d.npz"
    _register(conn, f1d, orders=[1])
    _register(conn, f2d, orders=[2])
    _register(conn, fcomb, orders=[1, 2])

    combined = find_packaged(conn, pert_order=12)
    assert {p.name for p in combined} == {fcomb.name}

    only_first = find_packaged(conn, pert_order=1)
    assert {p.name for p in only_first} == {f1d.name, fcomb.name}

    only_second = find_packaged(conn, pert_order=2)
    assert {p.name for p in only_second} == {f2d.name, fcomb.name}


def test_find_packaged_filters_by_calc_method(tmp_path):
    conn = ensure_db()
    f_all = tmp_path / "NV_512_raw_zfs_data_all_bands_1d.npz"
    f_approx = tmp_path / "NV_512_raw_zfs_data_defect_band_approx_1d.npz"
    _register(conn, f_all, orders=[1], method="all_bands")
    _register(conn, f_approx, orders=[1], method="defect_band_approx")

    hits = find_packaged(conn, calc_method="all_bands")
    assert [p.name for p in hits] == [f_all.name]


def test_run_query_normalizes_method(tmp_path, monkeypatch):
    """`spinlabs run --query --method all` must query calc_method='all_bands'."""
    from spinlabs.cli import run as run_mod

    captured = {}

    def fake_find(conn, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(run_mod, "find_packaged", fake_find, raising=False)
    monkeypatch.setattr(
        "spinlabs.registry.find_packaged", fake_find, raising=False
    )

    args = argparse.Namespace(
        defect="NV",
        cell_size=512,
        method="all",
        pert_scale=None,
        order=None,
        output_root=str(tmp_path / "runs"),
        debug=False,
    )
    run_mod._run_query(args)
    assert captured["calc_method"] == "all_bands"


def _make_raw_zfs(tmp_path, second=True, name="NV_512_raw_zfs_data_all_bands_12d.npz"):
    n_modes = 4
    pert_si = 0.025 * 1.60218e-22
    gs = ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz")
    first_order = {
        i: {
            "tensor": np.diag([-1000.0, -1000.0, 2000.0]) * 6.626e-28,
            "pert": pert_si,
            "symmetry": "A1" if i % 2 == 0 else "Ex",
            "ipr": 0.5,
        }
        for i in range(n_modes)
    }
    second_order = (
        {
            (i, j): {
                "tensor": np.diag([-1000.0, -1000.0, 2000.0]) * 6.626e-28,
                "pert": (pert_si, pert_si),
                "symmetry": ("A1", "A1") if i == j else ("A1", "Ex"),
                "ipr": (0.5, 0.5),
            }
            for i in range(n_modes)
            for j in range(i, n_modes)
        }
        if second
        else {}
    )
    raw = RawZFSData(
        defect="NV",
        cell_size=512,
        pert_scale=0.025,
        calc_method="all_bands",
        order=1,
        ground_state_zfs=gs,
        first_order=first_order,
        second_order=second_order,
    )
    f = tmp_path / name
    raw.save(f)
    return f


def _make_phonon(tmp_path):
    n = 4
    freqs = np.linspace(10.0, 40.0, n)
    freqs[1] = freqs[2] = 30.0  # degenerate pair for e-pair completeness
    spec = PhononSpectrum(
        frequencies_mev=freqs,
        symmetries=["A1", "Ex", "Ey", "A1"],
        iprs=np.ones(n) * 0.5,
        eigenvectors=np.ones((n, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3) * 5.0,
    )
    f = tmp_path / "phonon_data.npz"
    spec.save(f)
    return f


def test_pipeline_does_not_downgrade_packaged_row(tmp_path):
    """After a run over a packaged file, the registry row must stay packaged_zfs
    with its pert metadata intact, and the run must be registered."""
    conn = ensure_db()
    raw_file = _make_raw_zfs(tmp_path)
    ph_file = _make_phonon(tmp_path)
    register_file(
        conn,
        raw_file,
        "packaged_zfs",
        defect="NV",
        cell_size=512,
        calc_method="all_bands",
        pert_order=[1, 2],
        pert_scale=0.025,
    )

    run_full_pipeline(
        raw_zfs_file=raw_file,
        phonon_file=ph_file,
        calc_method="all_bands",
        save_plots=False,
        output_root=tmp_path / "runs",
        temperatures=[10.0],
    )

    row = conn.execute(
        "SELECT kind, defect, cell_size, calc_method, pert_order, pert_scale "
        "FROM files WHERE path = ?",
        (str(raw_file.resolve()),),
    ).fetchone()
    assert row is not None
    kind, defect, cell, method, orders, pert = row
    assert kind == "packaged_zfs"
    assert (defect, cell, method, pert) == ("NV", 512, "all_bands", 0.025)
    import json as _json

    assert _json.loads(orders) == [1, 2]


def test_pipeline_still_registers_unregistered_raw_file(tmp_path):
    """A raw file not already in the registry gets a raw_zfs row as before."""
    conn = ensure_db()
    raw_file = _make_raw_zfs(tmp_path, second=False, name="NV_512_raw_zfs_data_all_bands_1d.npz")
    ph_file = _make_phonon(tmp_path)

    run_full_pipeline(
        raw_zfs_file=raw_file,
        phonon_file=ph_file,
        calc_method="all_bands",
        save_plots=False,
        output_root=tmp_path / "runs",
        temperatures=[10.0],
    )

    row = conn.execute(
        "SELECT kind FROM files WHERE path = ?", (str(raw_file.resolve()),)
    ).fetchone()
    assert row is not None and row[0] == "raw_zfs"


def test_order_2_without_second_order_raises(tmp_path):
    """--order 2 on 1d-only input must fail loudly, not silently degrade to order 1."""
    raw_file = _make_raw_zfs(tmp_path, second=False, name="NV_512_raw_zfs_data_all_bands_1d.npz")
    ph_file = _make_phonon(tmp_path)

    with pytest.raises(ValueError, match="no second-order"):
        run_full_pipeline(
            raw_zfs_file=raw_file,
            phonon_file=ph_file,
            order=2,
            calc_method="all_bands",
            save_plots=False,
            output_root=tmp_path / "runs",
            temperatures=[10.0],
        )
