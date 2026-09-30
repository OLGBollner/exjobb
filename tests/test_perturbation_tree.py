"""Tests for phonon resolution in create_perturbation_tree."""
import re
import subprocess

from pathlib import Path

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


def test_patch_restart_incar_rewrites_flags():
    from beyblade.vasp.trees import patch_restart_incar
    def tag(text, name):
        vals = re.findall(rf"^\s*{name}\s*=\s*(.+?)\s*(?:!|#|$)", text, re.M | re.I)
        return vals[-1] if vals else None
    incar = "ISYM = 3\nISTART = 2\nICHARG = 2\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n"
    out = patch_restart_incar(incar)
    assert tag(out, "ISTART") == "0"
    assert tag(out, "ICHARG") == "1"
    assert tag(out, "LWAVE") == ".TRUE."
    assert tag(out, "LCHARG") == ".TRUE."
    assert out.count("ISTART") == 1  # old value replaced, not duplicated


def test_patch_restart_incar_appends_missing():
    from beyblade.vasp.trees import patch_restart_incar
    out = patch_restart_incar("ENCUT = 520\n")
    for name in ("ISTART", "ICHARG", "LWAVE", "LCHARG"):
        assert re.search(rf"^\s*{name}\s*=", out, re.M | re.I), name
    assert re.search(r"^\s*ENCUT\s*=", out, re.M)


def test_perturbation_input_incar_is_restart_ready(tmp_path):
    from beyblade.vasp import build_perturbation_tree
    defect = tmp_path / "NV_512"
    (defect / "data").mkdir(parents=True)
    _save(_tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"]),
          defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text("ENCUT = 520\nISTART = 2\nICHARG = 2\n")
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    failures = build_perturbation_tree(
        out=defect, pert=[0.0001], phonon=defect / "data" / "phonon_data.npz",
        scripts_dir=Path("scripts/cluster"), array_jobs=1, force=True,
        vasp_binary=Path("/bin/true"))
    assert failures == []
    incar = (defect / "first_order" / "pert_0.0001" / "all_bands"
             / "input" / "INCAR").read_text()
    for name, value in (("ISTART", "0"), ("ICHARG", "1"),
                        ("LWAVE", ".TRUE."), ("LCHARG", ".TRUE.")):
        vals = re.findall(rf"^\s*{name}\s*=\s*(.+?)\s*(?:!|#|$)", incar, re.M | re.I)
        assert vals, name
        assert vals[-1] == value, name


def test_cli_expands_tilde_binary(tmp_path, monkeypatch):
    """--vasp-binary ../... must resolve to an absolute path baked into the
    sbatch scripts, like --phonon."""
    defect = tmp_path / "NV_512"
    (defect / "data").mkdir(parents=True)
    _save(_tiny([10.0, 20.0, 30.0], ["A1", "Ex", "A2"]),
          defect / "data" / "phonon_data.npz")
    for stage in ("ZFS_hyp", "ZFS_occup"):
        d = defect / stage
        d.mkdir()
        (d / "INCAR").write_text("ENCUT = 520\n")
        for name in ("POSCAR", "KPOINTS", "POTCAR"):
            (d / name).write_text(name + "\n")
        (d / "OUTCAR").write_text("Elapsed time (sec):   100.000\n")
    args = [str(defect), "--pert", "0.0001", "--array-jobs", "1",
            "--vasp-binary", "../me/vasp_std"]
    r = subprocess.run(
        ["python", str(Path("scripts/cluster/create_perturbation_tree.py").resolve()), *args],
        capture_output=True, text=True, cwd=tmp_path,
        env={"PYTHONPATH": str(Path("src").resolve()), "PATH": __import__("os").environ["PATH"]})
    assert r.returncode == 0, r.stdout + r.stderr
    sbatch = next(defect.rglob("run_perturbation_first_order.sh"))
    assert f"binary={tmp_path.parent}/me/vasp_std" in sbatch.read_text()
