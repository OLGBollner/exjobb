"""Append-only index of analysis runs.

Each `beyblade run` invocation appends one row to <output_root>/runs_index.csv.
The index is plain CSV so it can be grepped, sorted, or loaded with pandas.
Rows are never updated or deleted; newer runs simply supersede older ones.

Columns:
    timestamp,defect,cell,method,order,pert,n_dims,
    temp_min,temp_max,n_temps,npz_files,git_hash,run_dir

Missing/unknown values are stored as "-1".
"""

from __future__ import annotations

import csv
import datetime as dt
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path

FIELDS: tuple[str, ...] = (
    "timestamp", "defect", "cell", "method", "order", "pert", "n_dims",
    "temp_min", "temp_max", "n_temps", "npz_files", "git_hash", "run_dir",
)


def index_path(output_root: str | Path) -> Path:
    """Return <output_root>/runs_index.csv."""
    return Path(output_root) / "runs_index.csv"


def _git_hash() -> str:
    """Short hash of HEAD, or '-1' outside a git repo / without git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=5,
        )
        return out.stdout.strip() or "-1"
    except Exception:  # noqa: BLE001 - any failure means "unknown"
        return "-1"


def _fmt(value: object) -> str:
    return "-1" if value is None else str(value)


def append_row(
    output_root: str | Path,
    *,
    defect: str,
    cell: str | int,
    method: str,
    order: int | None,
    pert: float | None,
    n_dims: Sequence[int] = (),
    temp_min: float | None = None,
    temp_max: float | None = None,
    n_temps: int | None = None,
    npz_files: Iterable[str] = (),
    run_dir: Path | str,
) -> Path:
    """Append one row for a completed run. Creates the index file if needed.

    Safe for sequential invocations; not safe for simultaneous writers on
    different nodes (not a use case for the laptop-side pipeline).
    """
    path = index_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    row = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "defect": _fmt(defect),
        "cell": _fmt(cell),
        "method": _fmt(method),
        "order": _fmt(order),
        "pert": _fmt(pert),
        "n_dims": "+".join(str(d) for d in sorted(set(n_dims))) or "-1",
        "temp_min": _fmt(temp_min),
        "temp_max": _fmt(temp_max),
        "n_temps": _fmt(n_temps),
        "npz_files": ";".join(npz_files) or "-1",
        "git_hash": _git_hash(),
        "run_dir": str(run_dir),
    }

    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return path


def read_rows(output_root: str | Path) -> list[dict]:
    """Read all index rows; returns [] if no index exists yet."""
    path = index_path(output_root)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def filter_rows(
    rows: Iterable[dict],
    filters: Iterable[tuple[str, str]] = (),
    *,
    latest: bool = False,
) -> list[dict]:
    """Filter rows by exact field==value pairs (case-insensitive values).

    Raises SystemExit on unknown field names so typos fail loudly.
    """
    filters = list(filters)
    for field, _ in filters:
        if field not in FIELDS:
            raise SystemExit(
                f"error: unknown index field '{field}'; "
                f"valid fields: {', '.join(FIELDS)}"
            )
    kept = rows
    for field, value in filters:
        kept = [r for r in kept if r.get(field, "").strip().lower() == value.strip().lower()]
    if latest:
        kept = kept[-1:]
    return kept


def run_dirs(rows: Iterable[dict], output_root: str | Path) -> list[Path]:
    """Resolve the run_dir column of each row against output_root."""
    root = Path(output_root)
    return [root / r["run_dir"] for r in rows]
