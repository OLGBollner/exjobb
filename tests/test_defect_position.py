"""Tests for beyblade.vasp.structures.find_defect.

Synthetic diamond-Si structures with known defects: a vacancy is
detected as the centroid of its under-coordinated neighbours, an
interstitial as a low-coordination atom. No pristine reference is
used, matching the self-contained heuristic.
"""

import numpy as np
from pymatgen.core import Structure

from beyblade.vasp.structures import find_defect


def _diamond_si_supercell(n: int = 3) -> Structure:
    """n x n x n supercell of diamond Si (2 atoms per cell, a = 5.43 A)."""
    prim = Structure(
        [[0, 2.715, 2.715], [2.715, 0, 2.715], [2.715, 2.715, 0]],
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )
    return prim.copy().make_supercell([n, n, n])


class TestFindDefect:
    def test_no_defect_in_pristine(self):
        struct = _diamond_si_supercell()
        assert find_defect(struct) == []

    def test_vacancy_position(self):
        struct = _diamond_si_supercell()
        removed = struct[5]
        defect_struct = Structure(
            struct.lattice,
            [s.specie for s in struct if s != removed],
            [s.frac_coords for s in struct if s != removed],
        )
        defects = find_defect(defect_struct)
        assert len(defects) == 1
        loc = defects[0]
        assert loc.defect_class == "vacancy"
        # Cartesian centroid of the under-coordinated neighbours should
        # coincide with the removed site (up to a periodic image).
        disp = (loc.frac_coords - removed.frac_coords + 0.5) % 1.0 - 0.5
        assert np.allclose(disp, 0.0, atol=1e-6)

    def test_interstitial_position(self):
        struct = _diamond_si_supercell(n=4)
        # Off-lattice-site position; a tetrahedral-site interstitial is
        # geometrically indistinguishable from a substitution and is
        # deliberately out of scope for this heuristic.
        inter_frac = np.array([0.6, 0.6, 0.6])
        struct.append("Si", inter_frac)
        defects = find_defect(struct)
        assert len(defects) == 1
        loc = defects[0]
        assert loc.defect_class == "interstitial"
        assert loc.site_index is None
        # The close pair: the added atom and the squeezed lattice atom.
        assert len(loc.neighbor_indices) == 2
        assert len(struct) - 1 in loc.neighbor_indices
        cart = np.mean([struct[i].coords for i in loc.neighbor_indices], axis=0)
        expected = struct.lattice.get_fractional_coords(cart) % 1.0
        assert np.allclose(loc.frac_coords, expected, atol=1e-6)
