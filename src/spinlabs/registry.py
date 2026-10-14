"""SQLite registry of files and analysis runs.

Lives at ~/.spinlabs/registry.db. Initialized lazily by ``spinlabs run`` /
``spinlabs package`` (and any other entry point that calls :func:`ensure_db`),
so there is nothing to set up by hand.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
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
    kind TEXT NOT NULL,              -- raw_zfs | packaged_zfs | phonon | coupling | two_phonon | convergence
    defect TEXT,
    cell_size INTEGER,
    calc_method TEXT,
    pert_order TEXT,                 -- JSON list of orders, e.g. '[1, 2]' for combined data
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
    command TEXT,
    defect TEXT,
    cell INTEGER,
    "order" INTEGER,
    pert REAL,
    n_dims TEXT,                              -- JSON list, e.g. '[1, 2]'
    temp_min REAL,
    temp_max REAL,
    n_temps INTEGER,
    npz_files TEXT,                           -- JSON list of npz names
    git_hash TEXT
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
    # then the integer 12. Legacy integer 1/2 rows are left alone — they were
    # registered as single-order data and rewriting them to [1, 2] would be wrong.
    cur = conn.execute(
        "SELECT COUNT(*) FROM files WHERE kind = 'packaged_zfs' AND (pert_order IS NULL OR pert_order = '12')"
    )
    if cur.fetchone()[0]:
        conn.execute(
            "UPDATE files SET pert_order = '[1, 2]' WHERE kind = 'packaged_zfs' AND (pert_order IS NULL OR pert_order = '12')",
        )
        conn.commit()
    # Migration: registrys created before the runs table grew the analysis-
    # index columns (replacing runs_index.csv) need the columns added.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    for column in (
        "defect", "cell", '"order"', "pert", "n_dims",
        "temp_min", "temp_max", "n_temps", "npz_files", "git_hash",
    ):
        if column.strip('"') not in existing:
            conn.execute(f"ALTER TABLE runs ADD COLUMN {column}")
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
    defect: str | None = None,
    cell: int | None = None,
    order: int | None = None,
    pert: float | None = None,
    n_dims: Sequence[int] = (),
    temp_min: float | None = None,
    temp_max: float | None = None,
    n_temps: int | None = None,
    npz_files: Sequence[str | Path] = (),
    git_hash: str | None = None,
) -> int:
    """Start a run row; returns its id for :func:`finish_run`.

    The analysis metadata (defect, cell, order, pert, temperatures, outputs)
    replaces what used to be appended to ``runs_index.csv``; it makes the
    ``runs`` table the single source of truth for analysis runs.
    """
    conn.execute(
        """INSERT INTO runs (run_dir, started_at, status, zfs_files, phonon_file, method, init_state, command,
                            defect, cell, "order", pert, n_dims, temp_min, temp_max, n_temps, npz_files, git_hash)
           VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            str(Path(run_dir).resolve()) if run_dir else None,
            _now(),
            json.dumps([str(Path(p).resolve()) for p in (zfs_files or [])]),
            str(Path(phonon_file).resolve()) if phonon_file else None,
            method,
            init_state,
            command,
            defect,
            cell,
            order,
            pert,
            json.dumps(sorted(set(n_dims))) if n_dims else None,
            temp_min,
            temp_max,
            n_temps,
            json.dumps([Path(p).name for p in npz_files]) if npz_files else None,
            git_hash,
        ),
    )
    conn.commit()
    return int(conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()[0])


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
    sql = "SELECT path, pert_order FROM files WHERE kind = 'packaged_zfs'"
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
        wanted = set(json.loads(_normalize_orders(pert_order)))
    if exclude_run:
        sql += """ AND NOT EXISTS (
            SELECT 1 FROM runs, json_each(COALESCE(runs.zfs_files, '[]')) AS j
            WHERE runs.status = 'done' AND j.value = files.path
        )"""
    cur = conn.execute(sql, params)  # type: ignore[arg-type]
    results = []
    for (p_, stored,) in cur.fetchall():
        if pert_order is not None:
            # Exact set membership on the stored JSON list, not substring matching.
            try:
                stored_orders = set(json.loads(stored)) if stored else set()
            except (TypeError, json.JSONDecodeError):
                stored_orders = set()
            if not wanted.issubset(stored_orders):
                continue
        if Path(p_).exists():
            results.append(Path(p_))
    return results


def find_runs(
    conn: sqlite3.Connection,
    filters: Iterable[tuple[str, str]] = (),
    *,
    latest: bool = False,
    output_root: str | Path | None = None,
) -> list[Path]:
    """Run directories of completed runs matching ``filters``.

    Replaces ``runs_index.csv``: the ``runs`` table is the index. Filter
    fields are the ``runs`` table columns (``defect``, ``cell``, ``order``,
    ``pert``, ``n_dims``, ``temp_min``, ``temp_max``, ``n_temps``,
    ``npz_files``, ``method``, ``init_state``, ``git_hash``, ``run_dir``,
    plus ``timestamp`` as an alias for ``started_at``); unknown fields raise
    SystemExit so typos fail loudly. Values match like the old CSV index:
    exact and case-insensitive, with numeric fields compared numerically and
    list fields (``n_dims`` as ``1+2``, ``npz_files`` per name) matching any
    element.

    With ``latest`` only the most recent matching run (by ``started_at``) is
    returned. Directories stored from another host are resolved against
    ``output_root`` by name when the stored path does not exist.
    """
    columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
    numeric = {"cell", "order", "pert", "temp_min", "temp_max", "n_temps"}
    list_like = {"n_dims", "npz_files"}
    sql = "SELECT run_dir FROM runs WHERE status = 'done'"
    params: list[object] = []
    for field, value in filters:
        column = "started_at" if field == "timestamp" else field
        if column not in columns | {"timestamp"}:
            raise SystemExit(
                f"error: unknown run filter '{field}'; valid fields: "
                + ", ".join(sorted(columns | {"timestamp"}))
            )
        if column in list_like:
            # Match like the CSV did: n_dims as '1+2', npz_files per name.
            wanted = value.strip().lower()
            keep = []
            for (raw,) in conn.execute(
                f"SELECT DISTINCT {column} FROM runs WHERE status = 'done'"
            ):
                try:
                    items = json.loads(raw) if raw else []
                except (TypeError, json.JSONDecodeError):
                    items = []
                if column == "n_dims":
                    text = "+".join(str(i) for i in items)
                else:
                    text = "".join(str(i) for i in items)
                if text.lower() == wanted or wanted in [str(i).lower() for i in items]:
                    keep.append(raw)
            if not keep:
                sql += " AND 0"  # nothing matches this filter
            else:
                sql += f" AND {column} IN ({', '.join('?' * len(keep))})"
                params.extend(keep)
            continue
        sql += f" AND {column} = ?"
        if column in numeric:
            try:
                number = float(value)
                if number.is_integer() and column in ("cell", "order", "n_temps"):
                    number = int(number)
                value = number
            except ValueError:
                pass
        params.append(value)
    sql += " ORDER BY started_at, id"
    if latest:
        sql += " DESC LIMIT 1"
    results: list[Path] = []
    root = Path(output_root).resolve() if output_root else None
    for (run_dir,) in conn.execute(sql, params):
        if not run_dir:
            continue
        path = Path(run_dir)
        if not path.exists() and root is not None:
            candidate = root / path.name
            if candidate.exists():
                path = candidate
        results.append(path)
    return results


def find_phonon(
    conn: sqlite3.Connection, defect: str, cell_size: int | str | None
) -> list[Path]:
    """Phonon files registered for this defect/cell that still exist on disk."""
    cur = conn.execute(
        "SELECT path FROM files WHERE kind = 'phonon' AND defect = ? AND cell_size = ? ORDER BY id",
        (defect, int(cell_size) if cell_size is not None else None),
    )
    return [Path(p) for (p,) in cur.fetchall() if Path(p).exists()]
