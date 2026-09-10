import numpy as np
import pytest
from beyblade.models import PhononSpectrum, RawZFSData, PerturbationEntry, ZFSTensor
from beyblade.zfs_manager import ZFSManager
from beyblade.parsers import parse_phonon_npz


def test_e_pair_completeness_flagging():
    # 4 modes: A1 (10 meV), Ex (20.0 meV), Ey (20.02 meV), Ex (30.0 meV - lonely)
    freqs = np.array([10.0, 20.0, 20.02, 30.0])
    syms = ["A1", "Ex", "Ey", "Ex"]
    eigs = np.zeros((4, 2, 3))
    atoms = np.zeros((2, 3))
    masses = np.array([12.0, 12.0])
    lat = np.eye(3)

    spec = PhononSpectrum(
        frequencies_mev=freqs,
        eigenvectors=eigs,
        atom_frac_coords=atoms,
        atom_symbols=["C", "C"],
        atomic_masses=masses,
        lattice=lat,
        symmetries=syms,
    )

    # __post_init__ should have computed e_pair_complete
    assert spec.e_pair_complete is not None
    assert spec.e_pair_complete == [False, True, True, False]

    # Expand missing E pairs
    expanded = spec.expand_missing_e_pairs()
    assert expanded.n_modes == 5
    assert expanded.symmetries == ["A1", "Ex", "Ey", "Ex", "Ey"]
    assert expanded.frequencies_mev[-1] == pytest.approx(30.0)


def test_clv_128_lonely_e_modes_expansion():
    path = "/rool-drive/exjobb/ClV_128/phonon_data_sym_n254.npz"
    spec = parse_phonon_npz(path)

    assert spec.n_modes == 254
    assert spec.n_atoms == 127
    # In this file, all Ex modes are lonely (no Ey present)
    assert len(spec.e_pair_complete) == 254
    assert all(c is False for c in spec.e_pair_complete)

    # When passed to ZFSManager, spectrum should expand to 3 * 127 = 381 modes
    mgr = ZFSManager(spectrum=spec)
    assert mgr.nmodes == 381
    assert mgr.spectrum.n_modes == 381
    # Original spectrum object remains unexpanded with 254 modes!
    assert spec.n_modes == 254


def test_save_and_load_preserves_e_pair_complete(tmp_path):
    out_file = tmp_path / "phonon_test.npz"
    freqs = np.array([15.0, 25.0])
    spec = PhononSpectrum(
        frequencies_mev=freqs,
        eigenvectors=np.zeros((2, 1, 3)),
        atom_frac_coords=np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=np.array([12.0]),
        lattice=np.eye(3),
        symmetries=["A1", "Ex"],
    )
    assert spec.e_pair_complete == [False, False]

    spec.save(out_file)
    loaded = PhononSpectrum.load(out_file)
    assert loaded.n_modes == 2
    assert loaded.e_pair_complete == [False, False]
