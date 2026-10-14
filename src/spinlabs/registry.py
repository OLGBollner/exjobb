"""SQLite registry of files and analysis runs.

Lives at ~/.spinlabs/registry.db. Initialized lazily by ``spinlabs run`` /
``spinlabs package`` (and any other entry point that calls :func:`ensure_db`),
so there is nothing to set up by hand.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_env_dir = os.environ.get("SPINLABS_REGISTRY_DIR")
REGISTRY_DIR = Path(_env_dir) if _env_dir else Path.home() / ".spinlabs"
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
    # Old rows: combined files had pert_order=NULL (before the 12 convention),
    # then 12. Both map to the unified list [1, 2].
    cur = conn.execute("SELECT COUNT(*) FROM files WHERE kind = 'packaged_zfs' AND (pert_order IS NULL OR pert_order NOT LIKE '[%')")
    if cur.fetchone()[0]:
        conn.execute(
            "UPDATE files SET pert_order = '[1, 2]' WHERE kind = 'packaged_zfs' AND (pert_order IS NULL OR pert_order NOT LIKE '[%')",
        )
        conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_of(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize_orders(value: list[int] | int | None) -> str | None:
    """Normalize a perturbation order value to a JSON list string, e.g. '[1, 2]'."""
    if value is None:
        return None
    if value == 12:  # legacy combined marker
        return json.dumps([1, 2])
    orders = sorted(set(value if isinstance(value, (list, tuple)) else [value]))
    return json.dumps(orders)


def register_file(
    conn: sqlite3.Connection,
    path: str | Path,
    kind: str,
    *,
    defect: str | None = None,
    cell_size: int | None = None,
    calc_method: str | None = None,
    pert_order: list[int] | int | None = None,
    pert_scale: float | None = None,
) -> int:
    """Insert or update a file row (re-registering overwrites metadata and hash).

    ``pert_order`` is normalized to a JSON list (e.g. ``[1, 2]`` for combined data).
    """
    path = str(Path(path).resolve())
    digest = sha256_of(path) if Path(path).exists() else None
    orders = _normalize_orders(pert_order)
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
            orders,
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


def find_packaged(
    conn: sqlite3.Connection,
    *,
    defect: str | None = None,
    cell_size: int | str | None = None,
    calc_method: str | None = None,
    pert_scale: float | None = None,
    pert_order: list[int] | int | None = None,
    exclude_run: bool = True,
) -> list[Path]:
    """Packaged ZFS files matching the filters.

    With ``exclude_run`` (default) files that already have a successful run
    registered against them are skipped, so repeated query runs pick up only
    new data.
    """
    sql = "SELECT path FROM files WHERE kind = 'packaged_zfs'"
    params: list[object] = []
    if defect is not None:
        sql += " AND defect = ?"
        params.append(defect)
    if cell_size is not None:
        sql += " AND cell_size = ?"
        params.append(int(cell_size))
    if calc_method is not None:
        sql += " AND calc_method = ?"
        params.append(calc_method)
    if pert_scale is not None:
        sql += " AND pert_scale = ?"
        params.append(pert_scale)
    if pert_order is not None:
        wanted = sorted(_normalize_orders(pert_order))
        sql += " AND (" + " AND ".join(f"instr(pert_order, '{o}')" for o in wanted) + ")"
    if exclude_run:
        sql += " AND NOT EXISTS (SELECT 1 FROM runs WHERE runs.status = 'done' AND runs.zfs_files LIKE '%' || files.path || '%')"
    cur = conn.execute(sql, params)  # type: ignore[arg-type]
    return [Path(p_) for (p_,) in cur.fetchall() if Path(p_).exists()]


def find_phonon(
    conn: sqlite3.Connection, defect: str, cell_size: int | str | None
) -> list[Path]:
    """Phonon files registered for this defect/cell that still exist on disk."""
    cur = conn.execute(
        "SELECT path FROM files WHERE kind = 'phonon' AND defect = ? AND cell_size = ? ORDER BY id",
        (defect, int(cell_size) if cell_size is not None else None),
    )
    return [Path(p) for (p,) in cur.fetchall() if Path(p).exists()]
