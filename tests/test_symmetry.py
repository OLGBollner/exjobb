"""Tests for the generalized symmetry pipeline.

Validation strategy: the general pipeline must reproduce the legacy
C3v hard-coded classification on real NV/ClV phonon data, plus unit
tests on synthetic structures with known point groups.
"""
from collections import Counter
from pathlib import Path
import warnings

import numpy as np

from beyblade.models import check_original_indices
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


# ---------------------------------------------------------------------------
# filter_sym_pairs: regression tests for the keep-all-Ex rule
# ---------------------------------------------------------------------------

def _tiny_spectrum(freqs, syms):
    """Minimal PhononSpectrum with preset labels (bypasses classification)."""
    import numpy as _np
    from beyblade.models import PhononSpectrum
    n = len(freqs)
    spec = PhononSpectrum(
        frequencies_mev=_np.asarray(freqs, dtype=float),
        eigenvectors=_np.zeros((n, 1, 3)),
        atom_frac_coords=_np.zeros((1, 3)),
        atom_symbols=["C"],
        atomic_masses=_np.array([12.0]),
        lattice=_np.eye(3),
        symmetries=list(syms),
        original_indices=_np.arange(n),
    )
    spec.symmetries = list(syms)  # keep the preset labels
    return spec


def test_filter_keeps_all_ex_drops_all_matched_ey():
    spec = _tiny_spectrum(
        [10.0, 20.005, 20.0, 30.0],
        ["A1", "Ex", "Ey", "A2"],
    )
    red = spec.filter_sym_pairs()
    assert list(red.symmetries) == ["A1", "Ex", "A2"]
    assert red.n_modes == 3
    assert red.original_indices.tolist() == [0, 1, 3]


def test_filter_drops_ey_even_when_ex_is_closer_to_another_ex():
    # Regression: the legacy greedy scan dropped the Ex at 149.0761 instead
    # of the Ey twin at 149.0706, because another Ex sat closer to it.
    spec = _tiny_spectrum(
        [149.0706, 149.0761, 149.0764],
        ["Ex", "Ex", "Ey"],
    )
    red = spec.filter_sym_pairs()
    # All Ex kept; the Ey (within tol of an Ex) dropped.
    assert list(red.symmetries) == ["Ex", "Ex"]
    assert red.original_indices.tolist() == [0, 1]


def test_filter_orphan_ey_is_kept():
    # An Ey with no Ex within tolerance must survive (it cannot be paired).
    spec = _tiny_spectrum([10.0, 10.008, 10.02], ["Ex", "Ey", "Ey"])
    red = spec.filter_sym_pairs()
    assert list(red.symmetries) == ["Ex", "Ey"]
    assert red.original_indices.tolist() == [0, 2]


def test_filter_group_theory_invariant_on_synthetic_c3v():
    # 1 atom in C3v: 3 modes = A1 + 2E. After filtering: A1 + E = 2 modes.
    spec = _tiny_spectrum([100.0, 100.0, 200.0], ["Ex", "Ey", "A1"])
    red = spec.filter_sym_pairs()
    assert red.n_modes == 2
    assert Counter(list(red.symmetries)) == {"Ex": 1, "A1": 1}


def test_filter_rejects_broken_original_indices():
    from beyblade.models import check_original_indices
    with pytest.raises(ValueError):
        check_original_indices(np.array([0, 0, 2]), n_modes=3)
    with pytest.raises(ValueError):
        check_original_indices(np.array([0, 1, 9]), n_modes=3, n_full=3)
    with pytest.raises(TypeError):
        check_original_indices(np.array([0.0, 1.0]), n_modes=2)


def test_filter_original_indices_valid_on_real_data():
    spec = parse_phonon_npz(NV_PATH)
    red = spec.filter_sym_pairs()
    check_original_indices(red.original_indices, red.n_modes, n_full=spec.n_modes)
    labels = [str(s) for s in red.symmetries]
    c = Counter(labels)
    # Group-theoretic invariant for NV-512: 511 atoms -> 1533 raw modes
    # = A1(287) + A2(224) + 2E(511); after filtering, no Ey survives.
    assert c["Ey"] == 0
    assert c["Ex"] == 511
    assert c["A1"] == 287 and c["A2"] == 224
    assert red.n_modes == 1022
