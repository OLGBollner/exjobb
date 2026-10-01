"""Tests for beyblade.vasp.structures.compare_displacement."""

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar

from beyblade.vasp import compare_displacement, compare_displacement_cli


@pytest.fixture
def ref_poscar(tmp_path):
    lattice = Lattice.cubic(3.57)
    species = ["C"] * 4
    coords = [[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0], [0.75, 0.75, 0.75]]
    struct = Structure(lattice, species, coords)
    path = tmp_path / "POSCAR_ref"
    Poscar(struct).write_file(path)
    return path, struct


def _write(path, struct):
    Poscar(struct).write_file(path)
    return path


def test_identical_structures(ref_poscar, tmp_path):
    ref_path, ref = ref_poscar
    pert = _write(tmp_path / "POSCAR_same", ref.copy())
    stats = compare_displacement(ref_path, pert)
    assert stats["max"] == pytest.approx(0.0, abs=1e-12)
    assert stats["rms"] == pytest.approx(0.0, abs=1e-12)
    assert stats["n_atoms"] == 4


def test_max_and_rms(ref_poscar, tmp_path):
    ref_path, ref = ref_poscar
    pert = ref.copy()
    pert.sites[0].coords = ref.sites[0].coords + [0.3, 0, 0]
    pert.sites[1].coords = ref.sites[1].coords + [0, 0, 0.1]
    pert_path = _write(tmp_path / "POSCAR_pert", pert)
    stats = compare_displacement(ref_path, pert_path)
    assert stats["max"] == pytest.approx(0.3)
    assert stats["max_index"] == 0
    assert stats["rms"] == pytest.approx(np.sqrt((0.3**2 + 0.1**2) / 4))
    assert stats["disp"].shape == (4, 3)


def test_atom_count_mismatch(ref_poscar, tmp_path):
    ref_path, ref = ref_poscar
    pert = Structure(ref.lattice, ["C"] * 3, [[0, 0, 0]] * 3)
    pert_path = _write(tmp_path / "POSCAR_bad", pert)
    with pytest.raises(ValueError, match="Atom count"):
        compare_displacement(ref_path, pert_path)


def test_composition_mismatch(ref_poscar, tmp_path):
    ref_path, ref = ref_poscar
    pert = Structure(ref.lattice, ["Si"] * 4, ref.frac_coords)
    pert_path = _write(tmp_path / "POSCAR_si", pert)
    with pytest.raises(ValueError, match="Composition"):
        compare_displacement(ref_path, pert_path)


def test_cli_table(ref_poscar, tmp_path, capsys):
    ref_path, ref = ref_poscar
    small = ref.copy()
    small.sites[0].coords = ref.sites[0].coords + [0.1, 0, 0]
    big = ref.copy()
    big.sites[0].coords = ref.sites[0].coords + [0.2, 0, 0]
    p1 = _write(tmp_path / "POSCAR_s", small)
    p2 = _write(tmp_path / "POSCAR_b", big)
    compare_displacement_cli(ref_path, [p1, p2])
    out = capsys.readouterr().out
    assert "max |disp|" in out
    assert "Relative sizes" in out
    assert "2.000" in out  # second entry is 2x the first


def test_cli_reports_missing_gracefully(ref_poscar, tmp_path, capsys):
    ref_path, _ = ref_poscar
    compare_displacement_cli(ref_path, [tmp_path / "nope"])
    out = capsys.readouterr().out
    assert "not found" in out or "No such file" in out


def test_wrapped_structure_matches(ref_poscar, tmp_path):
    """A structure shifted by a full lattice vector must compare as
    identical (nearest-periodic-image wrapping)."""
    ref_path, ref = ref_poscar
    pert = ref.copy()
    pert.sites[0].coords = ref.sites[0].coords + [3.57, 3.57, 0]
    pert_path = _write(tmp_path / "POSCAR_wrapped", pert)
    stats = compare_displacement(ref_path, pert_path)
    assert stats["max"] == pytest.approx(0.0, abs=1e-10)
    assert stats["rms"] == pytest.approx(0.0, abs=1e-10)
