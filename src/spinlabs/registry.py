"""SQLite registry of files and analysis runs.

Lives at ~/.spinlabs/registry.db. Initialized lazily by ``spinlabs run`` /
``spinlabs package`` (and any other entry point that calls :func:`ensure_db`),
so there is nothing to set up by hand.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_DIR = Path.home() / ".spinlabs"
REGISTRY_PATH = REGISTRY_DIR / "registry.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    kind TEXT NOT NULL,              -- raw_zfs | packaged_zfs | phonon | coupling | two_phonon
    defect TEXT,
    cell_size INTEGER,
    calc_method TEXT,
    pert_order INTEGER,
    pert_scale REAL,
    sha256 TEXT,
    registered_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    run_dir TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL DEFAULT 'running',   -- running | done | failed
    zfs_files TEXT,                           -- JSON list of file paths
    phonon_file TEXT,
    method TEXT,
    init_state TEXT,
    command TEXT
);
CREATE INDEX IF NOT EXISTS idx_files_defect ON files(defect, cell_size, calc_method);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
"""


def ensure_db() -> sqlite3.Connection:
    """Open (creating if needed) the registry database."""
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(REGISTRY_PATH)
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def register_file(
    conn: sqlite3.Connection,
    path: str | Path,
    kind: str,
    *,
    defect: str | None = None,
    cell_size: int | None = None,
    calc_method: str | None = None,
    pert_order: int | None = None,
    pert_scale: float | None = None,
) -> int:
    """Insert or update a file row (re-registering overwrites metadata and hash)."""
    path = str(Path(path).resolve())
    digest = sha256_of(path) if Path(path).exists() else None
    conn.execute(
        """INSERT INTO files (path, kind, defect, cell_size, calc_method, pert_order, pert_scale, sha256, registered_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(path) DO UPDATE SET
             kind=excluded.kind, defect=excluded.defect, cell_size=excluded.cell_size,
             calc_method=excluded.calc_method, pert_order=excluded.pert_order,
             pert_scale=excluded.pert_scale, sha256=excluded.sha256,
             registered_at=excluded.registered_at""",
        (
            path,
            kind,
            defect,
            cell_size,
            calc_method,
            pert_order,
            pert_scale,
            digest,
            _now(),
        ),
    )
    conn.commit()
    cur = conn.execute("SELECT id FROM files WHERE path = ?", (path,))
    return int(cur.fetchone()[0])


def register_run(
    conn: sqlite3.Connection,
    *,
    run_dir: str | Path | None = None,
    zfs_files: list[str | Path] | None = None,
    phonon_file: str | Path | None = None,
    method: str | None = None,
    init_state: str | None = None,
    command: str | None = None,
) -> int:
    """Start a run row; returns its id for :func:`finish_run`."""
    cur = conn.execute(
        """INSERT INTO runs (run_dir, started_at, status, zfs_files, phonon_file, method, init_state, command)
           VALUES (?, ?, 'running', ?, ?, ?, ?, ?)""",
        (
            str(Path(run_dir).resolve()) if run_dir else None,
            _now(),
            json.dumps([str(Path(p).resolve()) for p in (zfs_files or [])]),
            str(Path(phonon_file).resolve()) if phonon_file else None,
            method,
            init_state,
            command,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_run(conn: sqlite3.Connection, run_id: int, status: str) -> None:
    if status not in ("done", "failed"):
        raise ValueError(f"status must be 'done' or 'failed', got {status!r}")
    conn.execute(
        "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?", (status, _now(), run_id)
    )
    conn.commit()


def find_phonon(
    conn: sqlite3.Connection, defect: str, cell_size: int | str | None
) -> list[Path]:
    """Phonon files registered for this defect/cell that still exist on disk."""
    cur = conn.execute(
        "SELECT path FROM files WHERE kind = 'phonon' AND defect = ? AND cell_size = ? ORDER BY id",
        (defect, int(cell_size) if cell_size is not None else None),
    )
    return [Path(p) for (p,) in cur.fetchall() if Path(p).exists()]
