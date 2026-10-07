import numpy as np
import pytest

from beyblade.models import PhononSpectrum
from beyblade.parsers import (
    parse_phonon_data,
    parse_phonon_npz,
    parse_phonopy_yaml,
    save_phonon_npz,
)


def _spectrum() -> PhononSpectrum:
    return PhononSpectrum(
        frequencies_mev=np.array([1.0, 2.0, 3.0]),
        eigenvectors=np.ones((3, 2, 3)),
        atom_frac_coords=np.zeros((2, 3)),
        atom_symbols=["C", "C"],
        atomic_masses=np.array([12.0, 12.0]),
        lattice=np.eye(3),
    )


def test_dispatch_npz(tmp_path):
    path = tmp_path / "phonon_data.npz"
    save_phonon_npz(_spectrum(), path)
    result = parse_phonon_data(path)
    expected = parse_phonon_npz(path)
    np.testing.assert_allclose(result.frequencies_mev, expected.frequencies_mev)
    assert result.atom_symbols == expected.atom_symbols


def test_dispatch_yaml(tmp_path):
    path = tmp_path / "phonopy.yaml"
    path.write_text(
        "points:\n"
        "  - symbol: C\n"
        "    coordinates: [0.0, 0.0, 0.0]\n"
        "    mass: 12.0\n"
        "  - symbol: N\n"
        "    coordinates: [0.5, 0.5, 0.5]\n"
        "    mass: 14.0\n"
        "lattice: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n"
        "phonon:\n"
        "  - q-position: [0.0, 0.0, 0.0]\n"
        "    band:\n"
        + "".join(
            f"      - frequency: {f}\n"
            "        eigenvector:\n" + "".join(f"        - [[{a}, 0.0], [0.0, {a}], [0.0, 0.0]]\n" for a in (1.0, 2.0))
            for f in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        )
    )
    result = parse_phonon_data(path)
    expected = parse_phonopy_yaml(path)
    np.testing.assert_allclose(result.frequencies_mev, expected.frequencies_mev)
    assert result.atom_symbols == ["C", "N"]


def test_yaml_forward_poscar(tmp_path, monkeypatch):
    called = {}

    def fake_yaml(path, poscar_path=None):
        called["poscar"] = poscar_path
        return _spectrum()

    monkeypatch.setattr("beyblade.parsers.parse_phonopy_yaml", fake_yaml)
    y = tmp_path / "phonopy.yaml"
    y.write_text("phonon: []")
    parse_phonon_data(y, poscar_path=tmp_path / "POSCAR")
    assert called["poscar"] == tmp_path / "POSCAR"


def test_dispatch_case_insensitive_suffix(tmp_path):
    with pytest.raises(ValueError, match="Unsupported phonon file format"):
        parse_phonon_data(tmp_path / "data.txt")


def test_missing_file_npz(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_phonon_data(tmp_path / "missing.npz")


# ---------------------------------------------------------------------------
# Structure extraction from phonopy yaml layouts
#
# Modern phonopy (2.x) writes the crystal structure nested under
# ``supercell``/``primitive_cell``/``unit_cell`` sections instead of the old
# top-level ``lattice``/``points`` keys. When the structure could not be
# found, the parser used to fall back to a degenerate structure (identity
# lattice, all atoms at the origin) which spglib later chokes on with the
# cryptic ``SymmetryUndeterminedError`` (seen on the DiV_128 rerun).
# ---------------------------------------------------------------------------


def _modern_yaml(n_atoms: int = 2) -> str:
    """phonopy 2.x-style yaml: structure nested under ``supercell``."""
    bands = "".join(
        f"      - frequency: {f}\n"
        "        eigenvector:\n" + "".join(f"        - [[{a}, 0.0], [0.0, {a}], [0.0, 0.0]]\n" for a in (1.0, 2.0))
        for f in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    )
    return (
        "phonopy:\n"
        '  version: "2.16.0"\n'
        "physical_unit:\n"
        '  length: "angstrom"\n'
        "supercell:\n"
        "  lattice: [3.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 3.0]\n"
        "  points:\n"
        "    - symbol: C\n"
        "      coordinates: [0.0, 0.0, 0.0]\n"
        "      mass: 12.0\n"
        "    - symbol: N\n"
        "      coordinates: [0.5, 0.5, 0.5]\n"
        "      mass: 14.0\n"
        "phonon:\n"
        "  - q-position: [0.0, 0.0, 0.0]\n"
        "    band:\n" + bands
    )


def test_yaml_modern_layout_extracts_supercell_structure(tmp_path):
    path = tmp_path / "phonopy.yaml"
    path.write_text(_modern_yaml())
    result = parse_phonopy_yaml(path)
    np.testing.assert_allclose(result.lattice, 3.0 * np.eye(3))
    assert result.atom_symbols == ["C", "N"]
    np.testing.assert_allclose(result.atom_frac_coords, [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])


def test_yaml_missing_structure_raises_instead_of_degenerate(tmp_path):
    path = tmp_path / "phonopy.yaml"
    # Modern layout without any cell sections: the parser must fail loudly,
    # not silently build a degenerate structure.
    text = _modern_yaml()
    text = "\n".join(
        line
        for line in text.splitlines()
        if not line.strip().startswith(("lattice:", "points:", "symbol:", "coordinates:", "mass:"))
    )
    path.write_text(text)
    with pytest.raises(ValueError, match="POSCAR|structure"):
        parse_phonopy_yaml(path)
