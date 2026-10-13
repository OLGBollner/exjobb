"""Value-level regression guard for T1 and the transition-rate breakdown.

The rest of the suite checks shapes and finiteness. This module checks *numbers*:
it re-runs the real pipeline on the tracked NV_64, NV_512 and ClV_128 datasets and
compares T1(T) and the first/second/two-phonon rate breakdown against the frozen
fixtures in ``tests/reference/``.

Regenerate the fixtures with ``python scripts/regenerate_t1_reference.py``; see
``tests/reference/README.md`` for provenance and for the tolerances below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from spinlabs.cli import main as spinlabs_main

REPO_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_DIR = Path(__file__).parent / "reference"

# The three systems the reference sweep covers. Asserted explicitly so that
# deleting a fixture cannot quietly shrink the guard to a green subset.
EXPECTED_SYSTEMS = {"NV_64", "NV_512", "ClV_128"}

# ── Tolerances ───────────────────────────────────────────────────────────────
# The same sweep was run under CPython 3.10 (numpy 2.2.6, scipy 1.15.3), 3.12
# and 3.13 (numpy 2.5.4, scipy 1.18.1). Worst relative deviation across
# versions, over all three systems and all 12 temperatures:
#
#   transition rates   4.7e-16   (pure array arithmetic; machine precision)
#   t1_eigenval        8.5e-16   (3x3 eigendecomposition; machine precision)
#   t1_fit             6.9e-10   (odeint + curve_fit; iterative, so looser)
#
# Repeat runs on one machine are bit-identical. The tolerances below sit a few
# orders above the measured spread -- enough headroom for a different BLAS kernel
# or CPU, and still 6+ orders tighter than any physically meaningful change.
RTOL_RATES = 1e-9
RTOL_T1_EIGENVAL = 1e-9
RTOL_T1_FIT = 1e-6

# Absolute floor for transition rates, in s^-1. Below this a rate is numerical
# noise, not physics: 1e-30 s^-1 is a mean waiting time ~1e12 times the age of
# the universe. The floor exists because the pipeline currently emits values like
# 1e-171 s^-1 for the NV_64 one-phonon channel (a Gaussian-tail artefact, see
# tests/reference/README.md), which no relative tolerance can sensibly bound.
# Every rate a paper would quote is >= 1e-8 s^-1, far above this floor.
ATOL_RATE_PER_S = 1e-30

RATE_ORDERS = ("first_order", "second_order", "two_phonon")
RATE_TRANSITIONS = ("0_1", "1_-1")

_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


def _fixture_paths() -> list[Path]:
    return sorted(REFERENCE_DIR.glob("*.json"))


def _require_real_file(path: Path) -> None:
    """Fail loudly when an input is missing or is still an unresolved LFS pointer.

    Skipping here would be worse than failing: it would leave CI green with the
    value-level guard silently switched off, which is the exact hole this module
    exists to close.
    """
    if not path.exists():
        pytest.fail(f"missing pipeline input {path}\nRun 'git lfs pull' to fetch the tracked datasets.")
    if path.stat().st_size < 1024 and path.read_bytes().startswith(_LFS_POINTER_PREFIX):
        pytest.fail(f"{path} is an unresolved Git LFS pointer.\nRun 'git lfs pull' to fetch the tracked datasets.")


def _as_array(values: list[str]) -> np.ndarray:
    """Rebuild a float array from the fixture's exact round-trip strings."""
    return np.array([float(v) for v in values], dtype=float)


def _assert_matches(actual, expected, temperatures: np.ndarray, *, rtol: float, atol: float, label: str) -> None:
    """Compare one frozen quantity, reporting every offending temperature."""
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)

    assert actual.shape == expected.shape, f"{label}: shape {actual.shape} != frozen {expected.shape}"

    # The inf/NaN pattern is part of the frozen result, not an accident to be
    # tolerated away: NV_64 genuinely returns an infinite T1 below 30 K today.
    finite_mismatch = np.isfinite(actual) != np.isfinite(expected)
    if finite_mismatch.any():
        where = ", ".join(
            f"T={temperatures[i]:g} K: frozen {expected[i]!r} -> now {actual[i]!r}"
            for i in np.flatnonzero(finite_mismatch)
        )
        pytest.fail(f"{label}: finite/non-finite pattern changed at {where}")

    nonfinite = ~np.isfinite(expected)
    if nonfinite.any() and not np.array_equal(actual[nonfinite], expected[nonfinite], equal_nan=True):
        pytest.fail(f"{label}: non-finite values changed (inf/-inf/NaN mismatch)")

    finite = np.isfinite(expected)
    close = np.isclose(actual[finite], expected[finite], rtol=rtol, atol=atol)
    if close.all():
        return

    idx = np.flatnonzero(finite)[~close]
    lines = []
    for i in idx:
        exp, act = expected[i], actual[i]
        rel = abs(act - exp) / abs(exp) if exp != 0 else float("inf")
        lines.append(f"  T={temperatures[i]:g} K  frozen={exp!r}  now={act!r}  rel={rel:.3e}")
    pytest.fail(
        f"{label}: {len(idx)} of {finite.sum()} values moved beyond rtol={rtol:g} atol={atol:g}\n"
        + "\n".join(lines)
        + "\n\nIf this change is intended, regenerate with "
        "'python scripts/regenerate_t1_reference.py' and explain the shift in the PR."
    )


def test_every_expected_system_has_a_fixture():
    """Guard the guard: the reference set must not shrink unnoticed."""
    found = {p.stem for p in _fixture_paths()}
    assert found == EXPECTED_SYSTEMS, (
        f"reference fixture set changed: expected {sorted(EXPECTED_SYSTEMS)}, found {sorted(found)}"
    )


@pytest.mark.parametrize("fixture_path", _fixture_paths(), ids=lambda p: p.stem)
def test_pipeline_reproduces_frozen_t1_and_rates(fixture_path, tmp_path, monkeypatch):
    """Re-run the real pipeline and compare against the frozen numbers.

    Runs via ``spinlabs.cli.main`` with the argv recorded in the fixture, so the
    documented command line -- not just the library entry point -- is covered.
    """
    reference = json.loads(fixture_path.read_text(encoding="utf-8"))
    system = reference["system"]
    argv = reference["invocation"]["argv"]

    for arg in argv:
        if arg.endswith(".npz"):
            _require_real_file(REPO_ROOT / arg)

    # The fixture's argv uses repository-root-relative input paths.
    monkeypatch.chdir(REPO_ROOT)
    spinlabs_main([*argv, "-o", str(tmp_path), "--run-name", "reference"])

    run_dir = tmp_path / "reference"
    t1 = np.load(run_dir / "t1_relaxation.npz", allow_pickle=True)
    rates = np.load(run_dir / "transition_rates.npz", allow_pickle=True)

    # Metadata: cheap, and catches a run that silently resolved to another system.
    assert str(t1["defect"]) == reference["defect"], f"{system}: defect changed"
    assert int(t1["cell_size"]) == reference["cell_size"], f"{system}: cell_size changed"
    assert str(t1["calc_method"]) == reference["calc_method"], f"{system}: calc_method changed"
    assert int(t1["order"]) == reference["order"], f"{system}: perturbation order changed"
    assert str(t1["init_state"]) == reference["init_state"], f"{system}: init_state changed"
    assert float(rates["pert_scale"]) == float(reference["pert_scale"]), f"{system}: pert_scale changed"

    temperatures = _as_array(reference["temperatures_K"])
    _assert_matches(
        t1["temperatures"], temperatures, temperatures, rtol=0.0, atol=0.0, label=f"{system} temperature grid"
    )

    _assert_matches(
        t1["t1_eigenval"],
        _as_array(reference["t1_eigenval_s"]),
        temperatures,
        rtol=RTOL_T1_EIGENVAL,
        atol=0.0,
        label=f"{system} T1 (eigenvalue) [s]",
    )
    _assert_matches(
        t1["t1_fit"],
        _as_array(reference["t1_fit_s"]),
        temperatures,
        rtol=RTOL_T1_FIT,
        atol=0.0,
        label=f"{system} T1 (exponential fit) [s]",
    )

    frozen_rates = reference["transition_rates_per_s"]
    for order in RATE_ORDERS:
        channel = rates[order].item()
        for transition in RATE_TRANSITIONS:
            _assert_matches(
                channel[transition],
                _as_array(frozen_rates[order][transition]),
                temperatures,
                rtol=RTOL_RATES,
                atol=ATOL_RATE_PER_S,
                label=f"{system} {order} rate {transition} [1/s]",
            )
