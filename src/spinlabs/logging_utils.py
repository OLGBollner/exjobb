"""Central logging configuration for spinlabs.

Modules log through ``logging.getLogger(__name__)`` (children of the
``spinlabs`` logger). The CLI configures the root ``spinlabs`` logger once
via :func:`configure_logging`; library users may attach their own handlers.
"""

from __future__ import annotations

import logging
import sqlite3
import sys

from spinlabs.registry import insert_run_log

ROOT_LOGGER_NAME = "spinlabs"

_db_handler: SQLiteLogHandler | None = None


class SQLiteLogHandler(logging.Handler):
    """Persist log records into the ``run_logs`` table of the registry DB.

    ``run_id`` starts as ``None`` (logs emitted before the run is registered)
    and can be set once the run row exists.
    """

    def __init__(self, conn: sqlite3.Connection, run_id: int | None = None) -> None:
        super().__init__()
        self.conn = conn
        self.run_id = run_id

    def emit(self, record: logging.LogRecord) -> None:
        try:
            insert_run_log(
                self.conn, record.levelname, record.name, self.format(record), self.run_id
            )
        except Exception:
            self.handleError(record)


def attach_db_logging(conn: sqlite3.Connection, run_id: int | None = None) -> SQLiteLogHandler:
    """Attach (or re-point) the shared DB log handler for the ``spinlabs`` logger.

    Returns the handler so callers can keep a reference or detach it.
    """
    global _db_handler
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    if _db_handler is None or _db_handler not in logger.handlers:
        _db_handler = SQLiteLogHandler(conn, run_id)
        logger.addHandler(_db_handler)
    else:
        _db_handler.conn = conn
        _db_handler.run_id = run_id
    return _db_handler


def detach_db_logging() -> None:
    """Remove the shared DB log handler, if attached."""
    global _db_handler
    if _db_handler is not None:
        logging.getLogger(ROOT_LOGGER_NAME).removeHandler(_db_handler)
        _db_handler = None


def set_db_run_id(run_id: int | None) -> None:
    """Link subsequent log records to a registered run id."""
    if _db_handler is not None:
        _db_handler.run_id = run_id


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the logger for ``name`` (or the shared ``spinlabs`` logger)."""
    return logging.getLogger(name or ROOT_LOGGER_NAME)


def configure_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Set up the ``spinlabs`` logger with a stream handler.

    Default level is WARNING, i.e. only warnings and errors are shown; pass
    ``verbose=True`` to enable info and debug logging (the CLI ``--verbose``
    flag), or ``quiet=True`` to force WARNING regardless. Repeated calls
    reconfigure the single handler instead of stacking duplicates.
    """
    if verbose:
        level = logging.DEBUG
    else:
        level = logging.WARNING

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    else:
        logger.handlers[0].setLevel(level)
    logger.propagate = False
