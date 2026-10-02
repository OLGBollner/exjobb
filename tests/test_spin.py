"""General-S spin operator construction.

Basis convention: m descending, |S> first, |−S> last (matches the S=1
thesis matrices). Everything is plain numpy — no state classes.
"""
import numpy as np

from beyblade.spin import spin_z, spin_raise, spin_lower, spin_x, spin_y, spin_i


class TestHalfInteger:
    def test_s_half_z(self):
        np.testing.assert_array_equal(spin_z(0.5), 0.5 * np.diag([1.0, -1.0]))

    def test_s_half_raise(self):
        np.testing.assert_array_equal(
            spin_raise(0.5), np.array([[0.0, 1.0], [0.0, 0.0]]))

    def test_s_half_lower_is_conjugate_transpose_of_raise(self):
        np.testing.assert_array_equal(spin_lower(0.5), spin_raise(0.5).conj().T)

    def test_s_half_x(self):
        np.testing.assert_array_equal(
            spin_x(0.5), 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]]))

    def test_s_half_y(self):
        np.testing.assert_array_equal(
            spin_y(0.5), 0.5 * np.array([[0.0, -1j], [1j, 0.0]]))


class TestInteger:
    def test_s_one_z(self):
        np.testing.assert_array_equal(spin_z(1.0), np.diag([1.0, 0.0, -1.0]))

    def test_s_one_raise_coefficients(self):
        S = np.array([[0.0, np.sqrt(2.0), 0.0],
                      [0.0, 0.0, np.sqrt(2.0)],
                      [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(spin_raise(1.0), S)

    def test_s_one_ladder_product_identity(self):
        """S_+ S_- = S(S+1) - S_z^2 + S_z (m-descending: careful sign)."""
        Sp, Sm, Sz = spin_raise(1.0), spin_lower(1.0), spin_z(1.0)
        np.testing.assert_allclose(Sp @ Sm,
                                   (1.0 * (1.0 + 1)) * np.eye(3) - Sz @ Sz + Sz,
                                   atol=1e-12)


class TestGeneral:
    def test_s_three_halves_dimension_and_spectrum(self):
        Sz = spin_z(1.5)
        assert Sz.shape == (4, 4)
        np.testing.assert_allclose(np.diag(Sz), [1.5, 0.5, -0.5, -1.5])

    def test_s_two_dimension(self):
        assert spin_z(2.0).shape == (5, 5)

    def test_highest_ladder_element_is_sqrt_2S(self):
        """First element of S_+ is sqrt(S(S+1) - S(S-1)) = sqrt(2S)."""
        Sp = spin_raise(2.0)
        assert Sp[0, 1] == np.sqrt(4.0)

    def test_cartesian_components(self):
        Sx, Sy = spin_x(1.0), spin_y(1.0)
        np.testing.assert_allclose(Sx, (spin_raise(1.0) + spin_lower(1.0)) / 2)
        np.testing.assert_allclose(Sy, -1j * (spin_raise(1.0) - spin_lower(1.0)) / 2)


class TestIdentities:
    def test_commutator_sx_sy(self):
        Sx, Sy, Sz = spin_x(1.5), spin_y(1.5), spin_z(1.5)
        np.testing.assert_allclose(Sx @ Sy - Sy @ Sx, 1j * Sz, atol=1e-12)

    def test_squared_total_spin(self):
        """S_x^2 + S_y^2 + S_z^2 = S(S+1) I."""
        S = 2.5
        Sx, Sy, Sz = spin_x(S), spin_y(S), spin_z(S)
        total = Sx @ Sx + Sy @ Sy + Sz @ Sz
        np.testing.assert_allclose(total, S * (S + 1) * np.eye(int(2 * S + 1)), atol=1e-12)

    def test_spin_i_convenience(self):
        assert spin_i("z", 0.5).shape == (2, 2)
        np.testing.assert_array_equal(spin_i("x", 1.0), spin_x(1.0))
