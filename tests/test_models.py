import pytest
import numpy as np
from beyblade.models import ZFSTensor, PhononMode, PhononSpectrum, PerturbationEntry, RawZFSData
from beyblade.constants import CONSTANTS


class TestZFSTensor:
    def test_zfs_tensor_shape_validation(self):
        with pytest.raises(ValueError):
            ZFSTensor(matrix=np.zeros((2, 2)))

    def test_positive_D_principal_components(self):
        # Axial NV center D ~ 2870 MHz (D_xx = -2870/3, D_yy = -2870/3, D_zz = 2*2870/3)
        D_val = 2870.0
        mat = np.diag([-D_val / 3, -D_val / 3, 2 * D_val / 3])
        tensor = ZFSTensor(matrix=mat, unit="MHz")

        assert np.isclose(tensor.D, D_val)
        assert np.isclose(tensor.E, 0.0)
        assert np.isclose(tensor.zz, 2 * D_val / 3)

    def test_negative_D_principal_components(self):
        # Negative D tensor: D ~ -1200 MHz (D_zz = -800, D_xx = 400, D_yy = 400)
        D_val = -1200.0
        mat = np.diag([-D_val / 3, -D_val / 3, 2 * D_val / 3])
        tensor = ZFSTensor(matrix=mat, unit="MHz")

        assert np.isclose(tensor.D, D_val)
        assert np.isclose(tensor.E, 0.0)

    def test_rhombic_E(self):
        # D_xx = -100, D_yy = -300, D_zz = 400 -> D = 600, E = 100
        mat = np.diag([-100.0, -300.0, 400.0])
        tensor = ZFSTensor(matrix=mat, unit="MHz")

        assert np.isclose(tensor.D, 600.0)
        assert np.isclose(tensor.E, 100.0)

    def test_unit_conversion(self):
        mat = np.diag([-1000.0, -1000.0, 2000.0])
        t_mhz = ZFSTensor(matrix=mat, unit="MHz")

        t_j = t_mhz.to_unit("J")
        assert t_j.unit == "J"
        assert np.allclose(t_j.matrix, mat * CONSTANTS["MHz2J"])

        t_back = t_j.to_unit("MHz")
        assert np.allclose(t_back.matrix, mat)

    def test_rotation(self):
        # Rotate 90 degrees around z-axis
        mat = np.diag([100.0, 200.0, -300.0])
        tensor = ZFSTensor(matrix=mat)
        R_z90 = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])
        rot_tensor = tensor.rotate(R_z90)
        assert np.isclose(rot_tensor.matrix[0, 0], 200.0)
        assert np.isclose(rot_tensor.matrix[1, 1], 100.0)
        assert np.isclose(rot_tensor.matrix[2, 2], -300.0)


class TestPhononSpectrum:
    @pytest.fixture
    def sample_spectrum(self):
        n_modes = 6
        n_atoms = 4
        return PhononSpectrum(
            frequencies_mev=np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
            eigenvectors=np.random.randn(n_modes, n_atoms, 3),
            atom_frac_coords=np.zeros((n_atoms, 3)),
            atom_symbols=["C"] * n_atoms,
            atomic_masses=np.full(n_atoms, 12.011),
            lattice=np.eye(3) * 3.56,
            symmetries=["A1", "Ex", "Ey", "A1", "Ex", "Ey"],
            iprs=np.array([0.1, 0.2, 0.2, 0.5, 0.8, 0.8]),
        )

    def test_spectrum_properties(self, sample_spectrum):
        assert sample_spectrum.n_modes == 6
        assert sample_spectrum.n_atoms == 4

    def test_get_mode(self, sample_spectrum):
        mode = sample_spectrum.get_mode(1)
        assert isinstance(mode, PhononMode)
        assert mode.index == 1
        assert np.isclose(mode.frequency_mev, 20.0)
        assert mode.symmetry == "Ex"
        assert np.isclose(mode.ipr, 0.2)

    def test_filter_by_energy(self, sample_spectrum):
        filtered = sample_spectrum.filter_by_energy(min_mev=25.0, max_mev=45.0)
        assert list(filtered) == [2, 3]

    def test_filter_by_symmetry(self, sample_spectrum):
        a1_modes = sample_spectrum.filter_by_symmetry("A1")
        assert list(a1_modes) == [0, 3]
        ex_modes = sample_spectrum.filter_by_symmetry("Ex")
        assert list(ex_modes) == [1, 4]


class TestPhononPert:
    """Regression tests for PhononSpectrum.get_phonon_pert (mass-weighted displacements)."""

    def _make_spectrum(self, freqs_mev):
        n_modes = len(freqs_mev)
        n_atoms = 2
        eigs = np.zeros((n_modes, n_atoms, 3))
        eigs[:, 0, 0] = 1.0
        return PhononSpectrum(
            frequencies_mev=np.asarray(freqs_mev, dtype=float),
            eigenvectors=eigs,
            atom_frac_coords=np.zeros((n_atoms, 3)),
            atom_symbols=["C"] * n_atoms,
            atomic_masses=np.full(n_atoms, 12.011),
            lattice=np.eye(3) * 5.0,
        )

    def test_displacement_increases_with_frequency(self):
        """Heavier (higher-energy) modes give larger sqrt(2*omega/hbar) displacements."""
        spectrum = self._make_spectrum([10.0, 30.0, 90.0])
        pert = spectrum.get_phonon_pert(0.025)

        displacements = pert["disp"]
        assert displacements[0] is not None
        assert displacements[1] is not None
        assert displacements[2] is not None
        assert displacements[2] > displacements[1] > displacements[0]

    def test_displacement_values_match_old_phonon_manager(self):
        """Exact regression against the legacy PhononManager formula."""
        from scipy import constants as Cn
        from beyblade.constants import CONSTANTS

        freq_mev = 36.5
        pert_scale = 0.025
        spectrum = self._make_spectrum([freq_mev])

        expected = pert_scale * np.sqrt(2 * CONSTANTS["meV2rads"] * freq_mev / Cn.hbar)
        assert np.isclose(spectrum.get_phonon_pert(pert_scale)["disp"][0], expected)

    def test_non_positive_frequency_modes_are_none(self):
        """Modes with frequency <= 0 (acoustic at Gamma) should yield None displacement."""
        spectrum = self._make_spectrum([0.0, 25.0])
        pert = spectrum.get_phonon_pert(0.025)
        assert pert["disp"][0] is None
        assert pert["disp"][1] is not None

    def test_frequencies_converted_to_joule(self):
        """Frequencies must be returned in SI (Joule)."""
        from beyblade.constants import CONSTANTS

        spectrum = self._make_spectrum([0.0, 25.0])
        pert = spectrum.get_phonon_pert(0.025)
        assert np.isclose(pert["freqs"][1], 25.0 * CONSTANTS["meV2J"])
