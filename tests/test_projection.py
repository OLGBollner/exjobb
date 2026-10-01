"""Seam 3: symmetry projection of spin operators and derivative tensors.

Anchors:
- Thesis S=1 F matrices (operators.py F_x, F_y, F_z, F_xp, F_yp) reproduced
  from general-S formulas.
- Appendix-A channel formulas (zfs_manager.py V_0_0 / V_p_m / V_0_pm)
  reproduced from generic group projection of the 3x3 derivative tensor.
"""
import numpy as np

from beyblade.spin import spin_z, spin_raise, spin_lower, spin_x, spin_y
from beyblade.projection import (
    rank2_spin_operators,
    project_tensor,
    c3v_channel_components,
)


def _thesis_F(S=1.0):
    """Thesis definitions, written general-S."""
    Sp, Sm, Sz = spin_raise(S), spin_lower(S), spin_z(S)
    Sx, Sy = spin_x(S), spin_y(S)
    F_z = 3.0 * (Sz @ Sz - S * (S + 1) / 3.0 * np.eye(int(2 * S + 1)))
    F_x = 0.5 * (Sp @ Sp + Sm @ Sm)
    F_y = -0.5j * (Sp @ Sp - Sm @ Sm)
    F_xp = np.sqrt(2.0) * (Sx @ Sz + Sz @ Sx)
    F_yp = np.sqrt(2.0) * (Sy @ Sz + Sz @ Sy)
    return {"F_z": F_z, "F_x": F_x, "F_y": F_y, "F_xp": F_xp, "F_yp": F_yp}


C3V_OPS = None  # filled in TestC3vProjection via defect_frame_operations


class TestRank2SpinOperators:
    def test_s_one_matches_thesis_matrices(self):
        ops = rank2_spin_operators(1.0)
        thesis = _thesis_F(1.0)
        for name, F in thesis.items():
            np.testing.assert_allclose(ops[name], F, atol=1e-12,
                                       err_msg=name)

    def test_s_half_traceless_and_shapes(self):
        ops = rank2_spin_operators(0.5)
        assert set(ops) == {"F_z", "F_x", "F_y", "F_xp", "F_yp"}
        for name, F in ops.items():
            assert F.shape == (2, 2), name
            # traceless: no multiple of identity
            np.testing.assert_allclose(F - np.trace(F) / 2 * np.eye(2),
                                       F, atol=1e-12)

    def test_s_two_hermitian_and_traceless(self):
        ops = rank2_spin_operators(2.0)
        for name, F in ops.items():
            np.testing.assert_allclose(F, F.conj().T, atol=1e-12, err_msg=name)
            assert abs(np.trace(F)) < 1e-10, name

    def test_matches_legacy_operators_py_s1(self):
        """The old operators.py F matrices (built from SpinOperator class).

        NOTE: legacy S_y = +0.5j(S_+ - S_-) is the *negative* of the standard
        convention S_y = (S_+ - S_-)/(2i); so legacy F_yp is sign-flipped.
        The new code uses the standard convention (matches the thesis F_y).
        """
        from beyblade.operators import F_x, F_y, F_z, F_xp, F_yp
        ops = rank2_spin_operators(1.0)
        np.testing.assert_allclose(ops["F_x"], F_x.matrix, atol=1e-12)
        np.testing.assert_allclose(ops["F_y"], F_y.matrix, atol=1e-12)
        np.testing.assert_allclose(ops["F_z"], F_z.matrix, atol=1e-12)
        np.testing.assert_allclose(ops["F_xp"], F_xp.matrix, atol=1e-12)
        np.testing.assert_allclose(ops["F_yp"], -F_yp.matrix, atol=1e-12)


def _c3v_operations():
    """Standard C3v ops about z: E, C3, C3^2, and three sigma_v mirrors."""
    c = np.cos(2 * np.pi / 3)
    s = np.sin(2 * np.pi / 3)
    C3 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    mirror = np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, 1.0]])
    return [np.eye(3), C3, C3.T, mirror, C3 @ mirror, C3.T @ mirror]


def _D_xy():
    d = np.zeros((3, 3))
    d[0, 0], d[1, 1], d[2, 2] = 1.0, -2.0, 5.0
    d[0, 2] = d[2, 0] = 3.0
    d[1, 2] = d[2, 1] = -1.0
    return d


class TestC3vProjection:
    def test_appendix_A_A1(self):
        dD = _D_xy()
        comps = c3v_channel_components(dD, "A1")
        expected = abs(dD[2, 2] - 0.5 * (dD[0, 0] + dD[1, 1])) / 3.0
        assert abs(comps - expected) < 1e-12

    def test_appendix_A_E_pairs(self):
        dD = _D_xy()
        comps = c3v_channel_components(dD, "E")
        v_pm = abs(0.5 * np.sqrt((dD[0, 0] - dD[1, 1])**2 + 4 * dD[0, 1]**2))
        v_0pm = np.sqrt(dD[0, 2]**2 + dD[1, 2]**2) / np.sqrt(2)
        assert abs(comps["V_p_m"] - v_pm) < 1e-12
        assert abs(comps["V_0_pm"] - v_0pm) < 1e-12

    def test_generic_projection_reproduces_channels(self):
        """project_tensor with C3v ops gives the same numbers per irrep."""
        dD = _D_xy()
        ops = _c3v_operations()
        proj = project_tensor(dD, ops)
        a1 = abs(dD[2, 2] - 0.5 * (dD[0, 0] + dD[1, 1])) / 3.0
        assert abs(proj["A1"] - a1) < 1e-10
        e1 = np.array(proj["E"])  # two components per E irrep
        v_0pm = np.sqrt(dD[0, 2]**2 + dD[1, 2]**2) / np.sqrt(2)
        v_pm = 0.5 * np.sqrt((dD[0, 0] - dD[1, 1])**2 + 4 * dD[0, 1]**2)
        assert abs(np.linalg.norm(e1[0]) - v_0pm) < 1e-10
        assert abs(np.linalg.norm(e1[1]) - v_pm) < 1e-10

    def test_projection_invariance_under_group(self):
        """Rotating the input tensor by a group element must not change
        the projected magnitudes (character-weighted average)."""
        dD = _D_xy()
        ops = _c3v_operations()
        base = project_tensor(dD, ops)
        R = ops[1]
        rotated = R @ dD @ R.T
        proj = project_tensor(rotated, ops)
        assert abs(abs(proj["A1"]) - abs(base["A1"])) < 1e-10
        for b, p in zip(base["E"], proj["E"]):
            assert abs(np.linalg.norm(np.asarray(p)) -
                       np.linalg.norm(np.asarray(b))) < 1e-10
