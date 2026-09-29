"""Tests for phonon resolution in create_perturbation_tree."""
import numpy as np
import pytest

from beyblade.vasp import (build_zfs_tree, default_phonon, resolve_sym_phonon, sbatch_time)


@pytest.fixture
def defect(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    return tmp_path, data


def _save(spec, path):
    from beyblade.parsers import save_phonon_npz
    save_phonon_npz(spec, path)


def _tiny(freqs, syms):
    """Minimal PhononSpectrum with preset labels (bypasses classification)."""
    from beyblade.models import PhononSpectrum
    n = len(freqs)
    return PhononSpectrum(
        frequencies_mev=np.asarray(freqs, dtype=float),
        eigenvectors=np.zeros((n, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3),
        symmetries=list(syms),
        original_indices=np.arange(n),
    )


def test_finds_existing_sym_file(defect):
    out, data = defect
    spec = _tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"])
    _save(spec, data / "phonon_data.npz")
    _save(spec, data / "phonon_data_sym3.npz")
    phonon = resolve_sym_phonon(out, data / "phonon_data.npz")
    assert phonon == data / "phonon_data_sym3.npz"


def test_creates_sym_file_when_missing(defect):
    out, data = defect
    spec = _tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"])
    _save(spec, data / "phonon_data.npz")
    phonon = resolve_sym_phonon(out, data / "phonon_data.npz")
    assert phonon.is_file()
    assert phonon.name == "phonon_data_sym3.npz"
    assert phonon.parent == data


def test_default_phonon_in_data_folder(defect):
    out, data = defect
    _save(_tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"]), data / "phonon_data.npz")

    phonon = default_phonon(out)
    assert phonon == data / "phonon_data.npz"


def test_sbatch_time_rounds_up():
    assert sbatch_time(0.1) == "0:01:00"
    assert sbatch_time(3661) == "1:02:00"


def test_build_zfs_tree(tmp_path):
    inp = tmp_path / "NV_512"
    (inp / "data").mkdir(parents=True)
    (inp / "template" / "relax").mkdir(parents=True)
    (inp / "data" / "POSCAR").write_text("poscar\n")
    (inp / "template" / "relax" / "INCAR").write_text("INCAR\n")

    out = build_zfs_tree(inp, out=tmp_path / "tree")
    assert (out / "data" / "POSCAR").is_file()
    for stage in ("relaxation_data", "ZFS_hyp", "ZFS_occup"):
        assert (out / stage / "POSCAR").is_file()
        assert (out / stage / "INCAR").is_file()


def test_build_zfs_tree_refuses_overwrite(tmp_path):
    inp = tmp_path / "NV_512"
    (inp / "data").mkdir(parents=True)
    (inp / "template" / "relax").mkdir(parents=True)
    (inp / "data" / "POSCAR").write_text("poscar\n")
    (inp / "template" / "relax" / "INCAR").write_text("INCAR\n")
    build_zfs_tree(inp, out=tmp_path / "tree")
    with pytest.raises(FileExistsError):
        build_zfs_tree(inp, out=tmp_path / "tree")
