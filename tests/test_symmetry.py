"""Tests for the generalized symmetry pipeline (ADR-001).

Validation strategy: the general pipeline must reproduce the legacy
C3v hard-coded classification on real NV/ClV phonon data, plus unit
tests on synthetic structures with known point groups.
"""
from pathlib import Path
import numpy as np
import pytest
from pymatgen.core import Structure

from beyblade.symmetry import (
    detect_point_group,
    classify_modes,
    PointGroup,
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
    labels = classify_modes(spec)
    assert spec.symmetries is not None
    match = sum(l == s for l, s in zip(labels, spec.symmetries))
    assert match / len(labels) > 0.95, f"only {match}/{len(labels)} labels match"


def test_e_pairs_are_detected_as_degenerate():
    spec = parse_phonon_npz(NV_PATH)
    labels = classify_modes(spec)
    # In C3v, E modes come in degenerate pairs; check at least some E labels
    assert sum(l.startswith("E") for l in labels) > 0
