"""Tests for the full 32 point-group character tables.

Source: gernot-katzers-spice-pages.com, keys translated to spglib
Hermann-Mauguin symbols. Validation = representation-theory sum rules,
independent of how the tables were entered.
"""
import numpy as np
import pytest

from beyblade.character_tables import (
    POINT_GROUP_CHARACTER_TABLES,
    SPGLIB_TO_SCHOENFLIES,
)
from beyblade.symmetry import CHARACTER_TABLES, expand_classes

ALL_32 = [
    "1", "-1", "2", "m", "2/m", "222", "mm2", "mmm",
    "4", "-4", "4/m", "422", "4mm", "-42m", "4/mmm",
    "3", "-3", "32", "3m", "-3m",
    "6", "-6", "6/m", "622", "6mm", "-6m2", "6/mmm",
    "23", "m-3", "432", "-43m", "m-3m",
]


def _classes(symbol):
    return POINT_GROUP_CHARACTER_TABLES[symbol]["classes"]


def _irreps(symbol):
    return POINT_GROUP_CHARACTER_TABLES[symbol]["irreps"]


def test_all_32_groups_present():
    assert set(POINT_GROUP_CHARACTER_TABLES) == set(ALL_32)
    assert set(SPGLIB_TO_SCHOENFLIES) == set(ALL_32)


def test_class_sizes_sum_to_group_order():
    for sym in ALL_32:
        h = sum(m for _, m in _classes(sym))
        assert POINT_GROUP_CHARACTER_TABLES[sym]["order"] == h, sym
        dims = {name: chars[0] for name, chars in _irreps(sym).items()}
        # each row starts with chi(E) = dimension
        assert all(chars[0] == dim for dim, chars in
                   zip(dims.values(), _irreps(sym).values()))


@pytest.mark.parametrize("sym", ALL_32)
def test_row_norm_is_h_or_2h(sym):
    """Sum rule: sum_c n_c chi(c)^2 = h for real irreps; 2h for rows that
    bundle a complex-conjugate pair of 1D irreps (e.g. Eg of C4h)."""
    classes = _classes(sym)
    h = sum(m for _, m in classes)
    for name, chars in _irreps(sym).items():
        s = sum(m * c * c for (_, m), c in zip(classes, chars))
        assert s in (h, 2 * h), f"{sym}/{name}: {s} vs h={h}"


@pytest.mark.parametrize("sym", ALL_32)
def test_rows_mutually_orthogonal(sym):
    classes = _classes(sym)
    names = list(_irreps(sym))
    rows = [np.array(chars, dtype=float) for chars in _irreps(sym).values()]
    mults = np.array([m for _, m in classes], dtype=float)
    gram = (rows * mults) @ np.array(rows).T
    h = mults.sum()
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i == j:
                assert np.isclose(gram[i, j], 2 * h if gram[i, i] == 2 * h else h), \
                    f"{sym}/{name_i}: {gram[i, j]}"
            else:
                assert np.isclose(gram[i, j], 0.0), f"{sym}/{name_i},{name_j}: {gram[i, j]}"


@pytest.mark.parametrize("sym", ALL_32)
def test_dimension_sum_rule(sym):
    """Number of irreps = number of classes; combined rows count twice."""
    classes = _classes(sym)
    # combined rows (row-norm 2h) each bundle two complex 1D irreps and
    # contribute d^2/2 to the dimension sum
    h = sum(m for _, m in classes)
    total = sum(
        chars[0] ** 2 / (2 if sum(m * c * c for (_, m), c in zip(classes, chars)) == 2 * h else 1)
        for chars in _irreps(sym).values()
    )
    assert total == h, sym


def test_expand_classes_matches_existing_pipeline_tables():
    """Per-class tables expand to the per-operation vectors already used by
    classify_modes for the three pipeline groups."""
    # C3v: ops (E, C3, C3^2, sv1, sv2, sv3); classes (E,1), (C3,2), (sv,3)
    t3m = POINT_GROUP_CHARACTER_TABLES["3m"]
    assert expand_classes(t3m, ["E", "C3", "C3", "sv", "sv", "sv"]) == {
        "A1": [1, 1, 1, 1, 1, 1],
        "A2": [1, 1, 1, -1, -1, -1],
        "E": [2, -1, -1, 0, 0, 0],
    }
    # C2v: ops (E, C2, sv, sv')
    t2 = POINT_GROUP_CHARACTER_TABLES["mm2"]
    assert expand_classes(t2, ["E", "C2", "sv", "sd"])["A1"] == [1, 1, 1, 1]
    assert expand_classes(t2, ["E", "C2", "sv", "sd"])["B1"] == [1, -1, 1, -1]
    # D3h: ops (E, C3, C3, C2', C2', C2', sh, S3, S3, sv, sv, sv)
    td3h = POINT_GROUP_CHARACTER_TABLES["-6m2"]
    ops = ["E"] + ["C3"] * 2 + ["C2'"] * 3 + ["sh"] + ["S3"] * 2 + ["sv"] * 3
    got = expand_classes(td3h, ops)
    assert got["A1'"] == [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    assert got["E'"] == [2, -1, -1, 0, 0, 0, 2, -1, -1, 0, 0, 0]
    # -6m2 hand table is stored per-class (6 classes): compare directly
    hand = CHARACTER_TABLES["-6m2"]
    for name, vec in td3h["irreps"].items():
        assert vec == hand[name]
