import pytest
import numpy as np
from pathlib import Path
import tempfile

from beyblade.parsers import (
    parse_outcar_zfs,
    parse_outcar_energy,
    parse_phonon_npz,
    save_phonon_npz,
    parse_zfs_dataset_npz,
)
from beyblade.models import PhononSpectrum, ZFSTensor


class TestParsers:
    def test_parse_real_outcar_if_exists(self):
        outcar_path = Path("NV_64/OUTCAR")
        if outcar_path.is_file():
            zfs = parse_outcar_zfs(outcar_path)
            assert isinstance(zfs, ZFSTensor)
            assert zfs.matrix.shape == (3, 3)
            assert np.isclose(zfs.D, 3143.464, atol=1.0)

            energy = parse_outcar_energy(outcar_path)
            assert energy is not None
            assert isinstance(energy, float)

    def test_parse_outcar_nonexistent_returns_none(self):
        assert parse_outcar_zfs("nonexistent_path/OUTCAR") is None
        assert parse_outcar_energy("nonexistent_path/OUTCAR") is None

    def test_phonon_npz_roundtrip(self):
        n_modes = 4
        n_atoms = 2
        spectrum = PhononSpectrum(
            frequencies_mev=np.array([15.0, 25.0, 35.0, 45.0]),
            eigenvectors=np.random.randn(n_modes, n_atoms, 3),
            atom_frac_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
            atom_symbols=["Si", "C"],
            atomic_masses=np.array([28.085, 12.011]),
            lattice=np.eye(3) * 4.36,
            symmetries=["A1", "Ex", "Ey", "A1"],
            iprs=np.array([0.5, 0.5, 0.5, 0.8]),
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            save_phonon_npz(spectrum, tmp_path)
            loaded = parse_phonon_npz(tmp_path)

            assert loaded.n_modes == n_modes
            assert loaded.n_atoms == n_atoms
            assert np.allclose(loaded.frequencies_mev, spectrum.frequencies_mev)
            assert np.allclose(loaded.eigenvectors, spectrum.eigenvectors)
            assert loaded.atom_symbols == spectrum.atom_symbols
            assert loaded.symmetries == spectrum.symmetries
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_raw_zfs_npz_roundtrip_units_in_joule(self):
        """Verifies that saved npz tensors are in Joules and correctly converted."""
        from beyblade.constants import CONSTANTS

        # Create dummy 1D and 2D npz archives matching ZFSManager.save_data
        gs_mhz = np.diag([-1000.0, -1000.0, 2000.0])
        gs_joule = gs_mhz * CONSTANTS["MHz2J"]
        eigen_rot = np.eye(3)

        tensors_1d = {
            0: {
                "tensor": gs_joule + np.diag([0, 0, 10.0]) * CONSTANTS["MHz2J"],
                "pert": 1e-12,
                "symmetry": "A1",
                "ipr": 0.5,
            }
        }
        tensors_2d = {
            (0, 0): {
                "tensor": gs_joule + np.diag([0, 0, 20.0]) * CONSTANTS["MHz2J"],
                "pert": (1e-12, 1e-12),
                "symmetry": ("A1", "A1"),
                "ipr": (0.5, 0.5),
            }
        }

        with tempfile.NamedTemporaryFile(suffix="_1d.npz", delete=False) as f1, \
             tempfile.NamedTemporaryFile(suffix="_2d.npz", delete=False) as f2:
            path_1d = Path(f1.name)
            path_2d = Path(f2.name)

        try:
            np.savez(
                str(path_1d),
                order=1,
                defect="NV",
                cell_size=64,
                pert_scale=0.025,
                calc_method="all_bands",
                zfs_relaxed=gs_joule,
                eigen_rotation=eigen_rot,
                zfs_tensors=tensors_1d,
            )
            np.savez(
                str(path_2d),
                order=2,
                defect="NV",
                cell_size=64,
                pert_scale=0.025,
                calc_method="all_bands",
                zfs_relaxed=gs_joule,
                eigen_rotation=eigen_rot,
                zfs_tensors_2d=tensors_2d,
            )

            raw = parse_zfs_dataset_npz([path_1d, path_2d])
            assert raw.defect == "NV"
            # As loaded from legacy file, stored unit is Joules
            assert raw.ground_state_zfs.unit == "J"
            assert np.allclose(raw.ground_state_zfs.matrix, gs_joule)

            # Converting via dataclass internal converter
            raw_mhz = raw.to_unit("MHz")
            assert raw_mhz.ground_state_zfs.unit == "MHz"
            assert np.allclose(raw_mhz.ground_state_zfs.matrix, gs_mhz)

            # The tensor in first_order dict is preserved and converted
            assert np.allclose(raw.first_order[0]["tensor"], tensors_1d[0]["tensor"])
            assert np.allclose(raw_mhz.first_order[0]["tensor"], tensors_1d[0]["tensor"] / CONSTANTS["MHz2J"])
        finally:
            if path_1d.exists():
                path_1d.unlink()
            if path_2d.exists():
                path_2d.unlink()

    def test_raw_zfs_saves_with_latest_naming_convention(self):
        """Verifies that saving RawZFSData uses latest keys (first_order/second_order), not legacy keys."""
        from beyblade.models import RawZFSData, ZFSTensor

        gs = ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz")
        first = {0: {"tensor": np.eye(3), "pert": 0.01}}
        second = {(0, 0): {"tensor": np.eye(3), "pert": (0.01, 0.01)}}

        raw = RawZFSData(
            defect="NV",
            cell_size=64,
            pert_scale=0.025,
            calc_method="all_bands",
            order=2,
            ground_state_zfs=gs,
            first_order=first,
            second_order=second,
        )

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            save_path = Path(f.name)

        try:
            raw.save(save_path)

            # Inspect raw keys in the npz file
            with np.load(str(save_path), allow_pickle=True) as npz:
                keys = set(npz.files)
                # Must use latest naming conventions
                assert "first_order" in keys
                assert "second_order" in keys
                assert "ground_state_zfs_matrix" in keys
                assert "ground_state_zfs_unit" in keys

                # Must NOT contain duplicate legacy keys when saving new files
                assert "zfs_tensors" not in keys
                assert "zfs_tensors_2d" not in keys
                assert "zfs_relaxed" not in keys

            # Re-load to verify full roundtrip
            reloaded = RawZFSData.load(save_path)
            assert reloaded.defect == "NV"
            assert reloaded.ground_state_zfs.unit == "MHz"
            assert 0 in reloaded.first_order
            assert (0, 0) in reloaded.second_order
        finally:
            if save_path.exists():
                save_path.unlink()
