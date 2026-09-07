import pytest
import numpy as np
from beyblade.zfs_manager import ZFSManager
from beyblade.models import ZFSTensor, PhononSpectrum, RawZFSData
from beyblade.constants import CONSTANTS


class TestZFSManager:
    @pytest.fixture
    def setup_system(self):
        n_modes = 4
        n_atoms = 2
        spectrum = PhononSpectrum(
            frequencies_mev=np.array([10.0, 20.0, 20.0, 30.0]),
            eigenvectors=np.random.randn(n_modes, n_atoms, 3),
            atom_frac_coords=np.zeros((n_atoms, 3)),
            atom_symbols=["C", "C"],
            atomic_masses=np.array([12.011, 12.011]),
            lattice=np.eye(3) * 3.56,
            symmetries=["A1", "Ex", "Ey", "A1"],
            iprs=np.ones(n_modes) * 0.5,
        )

        relaxed_mat = np.diag([-2870.0 / 3, -2870.0 / 3, 2 * 2870.0 / 3])
        ground_state_zfs = ZFSTensor(matrix=relaxed_mat, unit="MHz")

        raw_data = RawZFSData(
            defect="NV",
            cell_size=64,
            pert_scale=0.01,
            ground_state_zfs=ground_state_zfs,
        )

        manager = ZFSManager(spectrum=spectrum, raw_data=raw_data)
        return manager, spectrum, ground_state_zfs

    def test_first_order_derivative_calculation(self, setup_system):
        manager, spectrum, gs_zfs = setup_system
        pert_SI = 0.01 * CONSTANTS["ang_amu2SI"]

        # Synthesize perturbation for A1 mode (idx 0) and Ex mode (idx 1)
        # A1: modulates D_zz
        # Ex: modulates D_xx - D_yy and D_xz
        d_a1 = gs_zfs.matrix.copy()
        d_a1[2, 2] += 50.0  # +50 MHz shift in zz

        d_ex = gs_zfs.matrix.copy()
        d_ex[0, 2] += 20.0
        d_ex[2, 0] += 20.0

        manager.zfs_tensors = {
            0: {"tensor": d_a1 * CONSTANTS["MHz2J"], "pert": pert_SI, "symmetry": "A1", "ipr": 0.5},
            1: {"tensor": d_ex * CONSTANTS["MHz2J"], "pert": pert_SI, "symmetry": "Ex", "ipr": 0.5},
        }
        manager.treated_modes = {0, 1}

        derivs, V_00, V_pm, V_0pm = manager.calculate_first_order_derivatives()

        # Check that V_00 is non-zero for A1 and V_0pm is non-zero for Ex
        assert V_00[0] > 0
        assert V_0pm[1] > 0
        assert np.isclose(V_pm[0], 0.0)

    def test_second_order_derivative_calculation(self, setup_system):
        manager, spectrum, gs_zfs = setup_system
        pert_SI = 0.01 * CONSTANTS["ang_amu2SI"]

        # Setup 1D derivatives first
        zfs_1d_derivs = np.zeros((4, 3, 3))
        zfs_1d_derivs[0, 2, 2] = 50.0 * CONSTANTS["MHz2J"] / pert_SI

        # Synthesize 2D diagonal perturbation (0, 0)
        d_2d = gs_zfs.matrix.copy()
        d_2d[2, 2] += 120.0

        manager.zfs_tensors_2d = {
            (0, 0): {
                "tensor": d_2d * CONSTANTS["MHz2J"],
                "pert": (pert_SI, pert_SI),
                "symmetry": ("A1", "A1"),
                "ipr": (0.5, 0.5),
            }
        }
        manager.treated_modes = {0}

        derivs_2d, V_00_2nd, V_pm_2nd, V_0pm_2nd = manager.calculate_second_order_derivatives(zfs_1d_derivs)

        assert V_00_2nd[0, 0] > 0
        assert np.isclose(V_00_2nd[0, 0], V_00_2nd[0, 0])

    def test_init_from_raw_data_with_legacy_scalar_pert_updates_to_si_displacements(self, setup_system):
        manager_base, spectrum, gs_zfs = setup_system

        # Raw data containing a static scalar pert (e.g. 0.025) like saved from legacy code
        raw_data = RawZFSData(
            defect="NV",
            cell_size=64,
            pert_scale=0.025,
            calc_method="all_bands",
            ground_state_zfs=gs_zfs,
            first_order={
                0: {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": 0.025},
                3: {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": 0.025},
            },
            second_order={
                (0, 3): {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": (0.025, 0.025)},
            },
        )

        manager = ZFSManager(spectrum=spectrum, raw_data=raw_data)

        # Displacements must be updated from spectrum, not remaining 0.025
        pert_0 = manager.first_order[0]["pert"]
        pert_3 = manager.first_order[3]["pert"]
        assert pert_0 != 0.025
        assert pert_3 != 0.025

        # Displacements must vary with frequency (mode 0 is 10 meV, mode 3 is 30 meV)
        assert pert_0 != pert_3
        pert_SI = 0.025 * CONSTANTS["ang_amu2SI"]
        expected_pert = spectrum.get_phonon_pert(pert_SI)
        assert np.isclose(pert_0, expected_pert["disp"][0])
        assert np.isclose(pert_3, expected_pert["disp"][3])

        # Second order perturbations must also be updated
        pert_2d = manager.second_order[(0, 3)]["pert"]
        assert pert_2d != (0.025, 0.025)
        assert np.isclose(pert_2d[0], expected_pert["disp"][0])
        assert np.isclose(pert_2d[1], expected_pert["disp"][3])

    def test_save_raw_zfs_data_with_spectrum_enriches_si_perturbations(self, setup_system, tmp_path):
        _, spectrum, gs_zfs = setup_system

        raw_data = RawZFSData(
            defect="NV",
            cell_size=64,
            pert_scale=0.025,
            calc_method="all_bands",
            ground_state_zfs=gs_zfs,
            first_order={
                0: {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": 0.025},
                3: {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": 0.025},
            },
            second_order={
                (0, 3): {"tensor": np.eye(3) * 1e-24, "unit": "J", "pert": (0.025, 0.025)},
            },
        )

        out_path = tmp_path / "raw_enriched.npz"
        raw_data.save(out_path, spectrum=spectrum)

        reloaded = RawZFSData.load(out_path)
        pert_SI = 0.025 * CONSTANTS["ang_amu2SI"]
        expected_pert = spectrum.get_phonon_pert(pert_SI)

        # 1D displacements are SI and mode-dependent
        assert np.isclose(reloaded.first_order[0]["pert"], expected_pert["disp"][0])
        assert np.isclose(reloaded.first_order[3]["pert"], expected_pert["disp"][3])
        assert reloaded.first_order[0]["pert"] != reloaded.first_order[3]["pert"]

        # 2D displacements are SI and mode-dependent tuple
        assert np.isclose(reloaded.second_order[(0, 3)]["pert"][0], expected_pert["disp"][0])
        assert np.isclose(reloaded.second_order[(0, 3)]["pert"][1], expected_pert["disp"][3])
