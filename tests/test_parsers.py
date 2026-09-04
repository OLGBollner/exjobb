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
    parse_zfs_simulation_dataset,
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

            # Re-load to verify full roundtrip (load() converts to Joules)
            reloaded = RawZFSData.load(save_path)
            assert reloaded.defect == "NV"
            assert reloaded.ground_state_zfs.unit in ("J", "MHz")
            assert 0 in reloaded.first_order
            assert (0, 0) in reloaded.second_order
        finally:
            if save_path.exists():
                save_path.unlink()

    @staticmethod
    def _create_mock_outcar(path: Path, d_diag=(-950.0, -950.0, 1900.0), energy=-500.12345):
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"""
 free  energy   TOTEN  =       {energy:.8f} eV

 Spin-spin contribution to zero-field splitting tensor (MHz)
 -------------------------------------------------------------
      D_xx      D_yy      D_zz      D_xy      D_xz      D_yz
 -------------------------------------------------------------
   {d_diag[0]:.2f}   {d_diag[1]:.2f}   {d_diag[2]:.2f}      0.00      0.00      0.00
 -------------------------------------------------------------
"""
        path.write_text(content, encoding="utf-8")

    def test_parse_zfs_simulation_dataset_1d(self, tmp_path):
        """Tests parsing a 1D simulation directory structure (e.g. NV_512/first_order/pert_0.1)."""
        root = tmp_path / "NV_512"

        # Ground-state unperturbed OUTCAR in ZFS_occup
        relaxed_outcar = root / "ZFS_occup" / "OUTCAR"
        self._create_mock_outcar(relaxed_outcar, d_diag=(-1000.0, -1000.0, 2000.0))

        # 1D perturbation runs
        sim_dir = root / "first_order" / "pert_0.1"
        run_1060 = sim_dir / "defect_band_approx" / "runs" / "1060" / "OUTCAR"
        run_1307 = sim_dir / "defect_band_approx" / "runs" / "1307" / "OUTCAR"
        self._create_mock_outcar(run_1060, d_diag=(-990.0, -990.0, 1980.0), energy=-510.5)
        self._create_mock_outcar(run_1307, d_diag=(-980.0, -980.0, 1960.0), energy=-511.2)

        raw_data = parse_zfs_simulation_dataset(sim_dir, method="approx")

        assert raw_data.defect == "NV"
        assert raw_data.cell_size == 512
        assert raw_data.pert_scale == pytest.approx(0.1)
        assert raw_data.calc_method == "defect_band_approx"
        assert raw_data.order == 1
        assert raw_data.ground_state_zfs is not None
        assert np.isclose(raw_data.ground_state_zfs.matrix[2, 2], 2000.0)

        assert 1059 in raw_data.first_order
        assert 1306 in raw_data.first_order
        entry_1060 = raw_data.first_order[1059]
        assert entry_1060.order == 1
        assert entry_1060.mode_indices == (1059,)
        assert entry_1060.amplitude == pytest.approx(0.1)
        assert np.isclose(entry_1060.zfs_tensor.matrix[2, 2], 1980.0)
        assert entry_1060.energy == pytest.approx(-510.5, abs=1e-3)

    def test_parse_zfs_simulation_dataset_2d(self, tmp_path):
        """Tests parsing a 2D simulation directory structure (e.g. NV_512/second_order/pert_0.1)."""
        root = tmp_path / "NV_512"

        # Ground-state unperturbed OUTCAR in ZFS_hyp for all_bands
        relaxed_outcar = root / "ZFS_hyp" / "OUTCAR"
        self._create_mock_outcar(relaxed_outcar, d_diag=(-1000.0, -1000.0, 2000.0))

        # 2D perturbation runs
        sim_dir = root / "second_order" / "pert_0.1"
        run_412_412 = sim_dir / "all_bands" / "runs" / "412_412" / "OUTCAR"
        run_412_413 = sim_dir / "all_bands" / "runs" / "412_413" / "OUTCAR"
        self._create_mock_outcar(run_412_412, d_diag=(-995.0, -995.0, 1990.0), energy=-520.1)
        self._create_mock_outcar(run_412_413, d_diag=(-992.0, -992.0, 1984.0), energy=-520.3)

        raw_data = parse_zfs_simulation_dataset(sim_dir, method="all_bands")

        assert raw_data.defect == "NV"
        assert raw_data.cell_size == 512
        assert raw_data.pert_scale == pytest.approx(0.1)
        assert raw_data.calc_method == "all_bands"
        assert raw_data.order == 2
        assert (411, 411) in raw_data.second_order
        assert (411, 412) in raw_data.second_order

        entry_2d = raw_data.second_order[(411, 411)]
        assert entry_2d.order == 2
        assert entry_2d.mode_indices == (411, 411)
        assert entry_2d.amplitude == (0.1, 0.1)
        assert np.isclose(entry_2d.zfs_tensor.matrix[2, 2], 1990.0)

    def test_parse_zfs_simulation_dataset_overrides_and_errors(self, tmp_path):
        """Tests parameter overrides and error handling in parse_zfs_simulation_dataset."""
        root = tmp_path / "Custom_256"
        relaxed = root / "ZFS_occup" / "OUTCAR"
        self._create_mock_outcar(relaxed)

        sim_dir = root / "first_order" / "pert_0.02"
        run_1 = sim_dir / "defect_band_approx" / "runs" / "1" / "OUTCAR"
        self._create_mock_outcar(run_1)

        # Test overrides
        raw = parse_zfs_simulation_dataset(
            sim_dir,
            method="approx",
            defect="Divacancy",
            cell_size=128,
            pert_scale=0.05,
            order=1,
        )
        assert raw.defect == "Divacancy"
        assert raw.cell_size == 128
        assert raw.pert_scale == pytest.approx(0.05)
        assert raw.order == 1

        # Test non-existent directory
        with pytest.raises(FileNotFoundError):
            parse_zfs_simulation_dataset(tmp_path / "nonexistent", method="approx")

        # Test folder name without "pert"
        bad_dir = root / "first_order" / "invalid_name"
        bad_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="Perturbation scale not found"):
            parse_zfs_simulation_dataset(bad_dir, method="approx")

        # Test missing relaxed OUTCAR
        sim_dir_no_relaxed = tmp_path / "Missing_64" / "first_order" / "pert_0.1"
        (sim_dir_no_relaxed / "defect_band_approx" / "runs" / "1").mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="Relaxed ZFS tensor not found"):
            parse_zfs_simulation_dataset(sim_dir_no_relaxed, method="approx")
