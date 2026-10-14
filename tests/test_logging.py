"""Tests for spinlabs.logging_utils and pipeline logging."""

import logging
import sqlite3

import numpy as np
import pytest

from spinlabs.logging_utils import configure_logging, get_logger
from spinlabs.pipeline import run_full_pipeline
from spinlabs.models import SpinPhononCouplingData, ZFSTensor


@pytest.fixture
def dummy_coupling_file(tmp_path):
    n_modes = 5
    coupling = SpinPhononCouplingData(
        order=2,
        defect="NV",
        cell_size=64,
        pert_scale=0.025,
        calc_method="all_bands",
        frequencies=np.linspace(10.0, 50.0, n_modes) * 1.60218e-22,
        frequency_unit="J",
        ground_state_zfs=ZFSTensor(matrix=np.diag([-1e-25, -1e-25, 2e-25]), unit="J"),
        V_0_0=np.array([1e-25, 2e-25, 1.5e-25, 3e-25, 2.2e-25]),
        V_p_m=np.array([2e-25, 1e-25, 2.5e-25, 1.8e-25, 3.1e-25]),
        V_0_pm=np.array([1.2e-25, 2.1e-25, 1.9e-25, 2.7e-25, 1.4e-25]),
        V2_0_0=np.ones((n_modes, n_modes)) * 1e-25,
        V2_p_m=np.ones((n_modes, n_modes)) * 2e-25,
        V2_0_pm=np.ones((n_modes, n_modes)) * 1.5e-25,
        coupling_unit="J",
        symmetries=["A1", "E", "E", "A1", "E"],
        iprs=np.full(n_modes, 0.2),
    )
    save_file = tmp_path / "test_coupling.npz"
    coupling.save(save_file)
    return save_file


@pytest.fixture(autouse=True)
def _reset_root_logger():
    logger = logging.getLogger("spinlabs")
    saved = logger.level, list(logger.handlers), logger.propagate
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    yield
    level, handlers, propagate = saved
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


def _capture_records():
    class Capture(logging.Handler):
        def __init__(self):
            super().__init__()
            self.records: list[logging.LogRecord] = []

        def emit(self, record):
            self.records.append(record)

    return Capture


def test_configure_logging_default_is_warning():
    logger = get_logger()
    configure_logging()
    assert logger.level == logging.WARNING


def test_configure_logging_verbose_and_quiet():
    logger = get_logger()
    configure_logging(verbose=True)
    assert logger.level == logging.DEBUG
    configure_logging(quiet=True)
    assert logger.level == logging.WARNING
    assert logger.level == logging.WARNING


def test_configure_logging_does_not_stack_handlers():
    logger = get_logger()
    configure_logging()
    configure_logging(verbose=True)
    assert len(logger.handlers) == 1
    assert logger.level == logging.DEBUG


def test_get_logger_returns_child_of_root():
    assert get_logger("spinlabs.pipeline").name == "spinlabs.pipeline"
    assert get_logger().name == "spinlabs"


def test_pipeline_uses_logging_not_print(tmp_path, dummy_coupling_file, capsys):
    """Pipeline messages go through the spinlabs logger, not bare print()."""
    from spinlabs.pipeline import run_full_pipeline

    Capture = _capture_records()()
    logger = logging.getLogger("spinlabs")
    logger.addHandler(Capture)
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        run_full_pipeline(
            coupling_file=str(dummy_coupling_file),
            output_root=str(tmp_path / "runs"),
            save_plots=False,
        )
    finally:
        logger.removeHandler(Capture)
        logger.setLevel(old_level)

    messages = [r.getMessage() for r in Capture.records if r.name.startswith("spinlabs")]
    assert messages, "expected pipeline log records"
    assert any("Initialized run" in m for m in messages)
    assert any("Run successfully completed" in m for m in messages)
    # Nothing may be printed to stdout anymore.
    assert capsys.readouterr().out == ""


def test_pipeline_logs_persist_to_registry(tmp_path, dummy_coupling_file, _isolated_registry):
    """run_full_pipeline attaches the DB handler and its INFO logs land in run_logs."""

    import spinlabs.registry as reg
    from spinlabs.logging_utils import detach_db_logging

    logger = logging.getLogger("spinlabs")
    old_level = logger.level
    logger.setLevel(logging.INFO)
    try:
        run_full_pipeline(
            coupling_file=str(dummy_coupling_file),
            output_root=str(tmp_path / "runs"),
            save_plots=False,
        )
    finally:
        logger.setLevel(old_level)
        detach_db_logging()

    conn = sqlite3.connect(tmp_path / ".spinlabs" / "registry.db")
    rows = reg.list_run_logs(conn)
    assert any("Initialized run" in r["message"] for r in rows)
    assert any(r["level"] == "INFO" for r in rows)
