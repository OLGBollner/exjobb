"""Tests for the generalized symmetry pipeline.

Validation strategy: the general pipeline must reproduce the legacy
C3v hard-coded classification on real NV/ClV phonon data, plus unit
tests on synthetic structures with known point groups.
"""
from pathlib import Path
import warnings

import numpy as np
import pytest
from pymatgen.core import Structure

from beyblade.symmetry import (
    detect_point_group,
    classify_modes,
    _parent_label,
)
from beyblade.parsers import parse_phonon_npz


# ---------------------------------------------------------------------------
# Synthetic structures: known point groups
# ---------------------------------------------------------------------------

def _fcc_si_structure():
    """Primitive Si diamond structure -> space group Fd-3m, point group m-3m (Oh)."""
    lattice = np.array([[0, 5.43 / 2, 5.43 / 2],
                        [5.43 / 2, 0, 5.43 / 2],
                        [5.43 / 2, 5.43 / 2, 0]])
    return Structure(lattice, ["Si", "Si"], [[0, 0, 0], [0.25, 0.25, 0.25]])


class TestPointGroupDetection:
    def test_diamond_si_is_oh(self):
        pg = detect_point_group(_fcc_si_structure())
        assert pg.symbol == "m-3m"
        assert pg.order == 48

    def test_returns_operations(self):
        pg = detect_point_group(_fcc_si_structure())
        assert len(pg.operations) == 48


# ---------------------------------------------------------------------------
# Mode classification vs. legacy C3v results on real data
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
NV_PATH = REPO_ROOT / "NV_512" / "phonon_data.npz"
CLV_PATH = REPO_ROOT / "ClV_128" / "phonon_data.npz"


@pytest.mark.parametrize("path", [NV_PATH, CLV_PATH])
def test_matches_legacy_c3v_labels(path):
    spec = parse_phonon_npz(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        legacy = spec.analyze_c3v_symmetry()
    labels, _groups = classify_modes(spec)
    match = sum(lab == s for lab, s in zip(labels, legacy))
    assert match / len(legacy) > 0.95, f"only {match}/{len(legacy)} labels match"


def test_e_pairs_are_detected_as_degenerate():
    spec = parse_phonon_npz(NV_PATH)
    labels, groups = classify_modes(spec)
    # In C3v, E modes come in degenerate pairs; Ex/Ey partners share a
    # degeneracy group (same parent irrep, same frequency).
    e_groups = [g for g in groups if labels[g[0]].startswith("E")]
    assert e_groups
    assert all(len(g) >= 2 for g in e_groups)


def test_ex_ey_share_degeneracy_group():
    spec = parse_phonon_npz(CLV_PATH)
    labels, groups = classify_modes(spec)
    # ClV-128 has clean Ex/Ey pairs: every Ex must sit in a 2-member group
    # with its Ey partner at the same frequency.
    for i, lab in enumerate(labels):
        if lab == "Ex":
            g = next(g for g in groups if i in g)
            partners = {labels[j] for j in g if j != i}
            assert partners == {"Ey"}


def test_accidental_degeneracies_do_not_merge():
    # Two modes of different parent irreps at the same frequency must not
    # land in the same degeneracy group.
    spec = parse_phonon_npz(NV_PATH)
    labels, groups = classify_modes(spec)
    for g in groups:
        parents = {_parent_label(labels[j]) for j in g}
        assert len(parents) == 1
