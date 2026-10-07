import numpy as np
import yaml
from pymatgen.core import Structure

from beyblade.parsers import parse_phonopy_yaml


def _write_yaml(path, lattice):
    n = 8
    bands = []
    for i in range(3 * n):
        bands.append(
            {
                "frequency": 1.0 + i * 0.1,
                "eigenvector": [[[1.0, 0.0, 0.0]] for _ in range(n)],
            }
        )
    path.write_text(yaml.safe_dump({"phonon": [{"q-position": [0, 0, 0], "band": bands}]}))


def test_falls_back_to_sibling_poscar(tmp_path):
    struct = Structure(
        lattice=np.eye(3) * 5.0,
        species=["Si"] * 8,
        coords=np.random.RandomState(0).random((8, 3)),
        coords_are_cartesian=False,
    )
    struct.to(fmt="poscar", filename=str(tmp_path / "POSCAR"))
    yaml_path = tmp_path / "phonons.yaml"
    _write_yaml(yaml_path, np.eye(3))
    spectrum = parse_phonopy_yaml(yaml_path)
    assert spectrum.atom_symbols == ["Si"] * 8
    assert not np.allclose(spectrum.lattice, np.eye(3))


def test_no_poscar_still_degrades_to_defaults(tmp_path, capsys):
    yaml_path = tmp_path / "phonons.yaml"
    _write_yaml(yaml_path, np.eye(3))
    spectrum = parse_phonopy_yaml(yaml_path)
    assert spectrum.atom_symbols == ["X"] * 8
