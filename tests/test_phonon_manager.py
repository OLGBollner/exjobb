import pytest
import numpy as np
from beyblade.phonon_manager import PhononManager
from beyblade.models import PhononSpectrum

class TestPhononManager:
    @pytest.fixture
    def sample_manager(self):
        n_modes = 6
        n_atoms = 4
        # Create normalized eigenvectors: sum_i (e_i)^2 == 1 for each mode
        raw_eigs = np.random.randn(n_modes, n_atoms, 3)
        norms = np.linalg.norm(raw_eigs.reshape(n_modes, -1), axis=1)[:, None, None]
        normalized_eigs = raw_eigs / norms

        spectrum = PhononSpectrum(
            frequencies_mev=np.array([10.0, 20.0, 20.0, 40.0, 50.0, 50.0]),
            eigenvectors=normalized_eigs,
            atom_frac_coords=np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75]]),
            atom_symbols=["N", "C", "C", "C"],
            atomic_masses=np.array([14.007, 12.011, 12.011, 12.011]),
            lattice=np.eye(3) * 3.56,
            symmetries=["A1", "Ex", "Ey", "A1", "Ex", "Ey"],
            iprs=np.full(n_modes, 0.5),
        )
        return PhononManager(spectrum=spectrum)

    def test_manager_properties(self, sample_manager):
        assert sample_manager.nmodes == 6
        assert sample_manager.cell_size == 4
        assert len(sample_manager.get_freqs()) == 6

    def test_translate_defect_to_origin(self, sample_manager):
        shifted_frac, defect_pos = sample_manager.translate_defect_to_origin()
        # N is at index 0, so shifted position of atom 0 should be at origin (0, 0, 0)
        assert np.allclose(shifted_frac[0], [0.0, 0.0, 0.0])

    def test_get_phonon_pert(self, sample_manager):
        pert = sample_manager.get_phonon_pert(perturbation_scale=0.01)
        assert "sym" in pert
        assert "eigs" in pert
        assert "freqs" in pert
        assert len(pert["freqs"]) == 6
