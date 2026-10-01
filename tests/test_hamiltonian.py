"""Tests for Hamiltonian assembly (seam 4).

Ground truth: S=1 thesis Hamiltonian eigenvalues.
H = D(Sz^2 - S(S+1)/3) + E(Sx^2 - Sy^2)
  |0>  -> -2D/3
  |+-> -> D/3 +- E
"""
import numpy as np
import pytest

from beyblade.projection import rank2_spin_operators

from beyblade.hamiltonian import zfs_hamiltonian
from beyblade.spin import spin_x, spin_y, spin_z


def _eigvals(H):
    return np.sort(np.linalg.eigvalsh(H))


class TestZfsHamiltonian:
    def test_s1_scalar_d_and_e(self):
        S = 1
        D, E = 2.87e9, 0.1e9  # arbitrary units; rates come later
        Sz, Sx, Sy = spin_z(S), spin_x(S), spin_y(S)
        H = D * (Sz @ Sz - S * (S + 1) / 3 * np.eye(3)) + E * (Sx @ Sx - Sy @ Sy)
        expected = np.sort([-2 * D / 3, D / 3 - E, D / 3 + E])
        assert np.allclose(_eigvals(H), expected)

    def test_s1_from_d_tensor(self):
        # axis-aligned D tensor: diag(Dxx, Dyy, Dzz), traceless
        S = 1
        # convention: D = 3*Dzz/2, E = (Dxx - Dyy)/2 with traceless tensor
        Dzz = 1.0e9
        Dxx, Dyy = -0.5e9 + 0.1e9, -0.5e9 - 0.1e9  # E = (Dxx-Dyy)/2 = 0.1e9
        assert Dxx + Dyy + Dzz == 0
        D = 3 * Dzz / 2
        E = (Dxx - Dyy) / 2
        Dt = np.diag([Dxx, Dyy, Dzz])
        H = zfs_hamiltonian(S, Dt)
        expected = np.sort([-2 * D / 3, D / 3 - E, D / 3 + E])
        assert np.allclose(_eigvals(H), expected)

    def test_isotropic_part_harmless(self):
        # adding c*Identity*trace doesn't change gaps (constant shift)
        S = 1
        Dt = np.diag([-0.5e9, -0.5e9, 1.0e9])
        H1 = zfs_hamiltonian(S, Dt)
        H2 = zfs_hamiltonian(S, Dt + 5e8 * np.eye(3))
        v1, v2 = _eigvals(H1), _eigvals(H2)
        assert np.allclose(np.diff(v1), np.diff(v2))

    def test_half_integer_s(self):
        # S=1/2: no ZFS splitting -> all eigenvalues equal
        S = 0.5
        Dt = np.diag([-0.5e9, -0.5e9, 1.0e9])
        H = zfs_hamiltonian(S, Dt)
        vals = _eigvals(H)
        assert np.allclose(vals, vals[0])


class TestSpinPhononCoupling:
    def test_a1_mode_operator_is_scaled_fz(self):
        from beyblade.hamiltonian import spin_phonon_coupling
        S = 1
        V = {"V_0_0": 0.37}  # only the A1 channel
        H = spin_phonon_coupling(S, V)
        F = rank2_spin_operators(S)
        assert np.allclose(H, 0.37 * F["F_z"])

    def test_e_mode_channels(self):
        from beyblade.hamiltonian import spin_phonon_coupling
        S = 1
        V = {"V_0_pm": 0.21, "V_p_m": 0.13}
        H = spin_phonon_coupling(S, V)
        F = rank2_spin_operators(S)
        assert np.allclose(H, 0.21 * F["F_xp"] + 0.13 * F["F_x"])

    def test_zero_channels_give_zero(self):
        from beyblade.hamiltonian import spin_phonon_coupling
        H = spin_phonon_coupling(1.0, {})
        assert np.allclose(H, 0)

    def test_hermitian_for_real_channels(self):
        from beyblade.hamiltonian import spin_phonon_coupling
        H = spin_phonon_coupling(1.5, {"V_0_0": 1.0, "V_p_m": -0.4})
        assert np.allclose(H, H.conj().T)

    def test_unknown_channel_raises(self):
        from beyblade.hamiltonian import spin_phonon_coupling
        with pytest.raises(ValueError):
            spin_phonon_coupling(1.0, {"V_bogus": 1.0})
