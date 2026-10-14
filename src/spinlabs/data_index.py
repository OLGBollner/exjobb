"""Append-only index of packaged ZFS data.

Each `spinlabs package` invocation appends one row per .npz to
<output_root>/data_index.csv. The index is plain CSV so it can be grepped,
sorted, or loaded with pandas. Rows are never updated or deleted; newer
packaging runs simply supersede older ones.

Columns:
    timestamp,defect,cell,method,order,pert,n_dims,npz_files,git_hash,data_dir

Missing/unknown values are stored as "-1".
"""

from __future__ import annotations

import csv
import datetime as dt
from collections.abc import Iterable, Sequence
from pathlib import Path

from spinlabs.runs_index import _fmt, _git_hash

FIELDS: tuple[str, ...] = (
    "timestamp", "defect", "cell", "method", "order", "pert", "n_dims",
    "npz_files", "git_hash", "data_dir",
)


def index_path(output_root: str | Path) -> Path:
    """Return <output_root>/data_index.csv."""
    return Path(output_root) / "data_index.csv"


def append_row(
    output_root: str | Path,
    *,
    defect: str,
    cell: str | int,
    method: str,
    order: int | str | None = None,
    pert: float | None = None,
    n_dims: Sequence[int] = (),
    npz_files: Iterable[str] = (),
    data_dir: Path | str,
) -> Path:
    """Append one row for a packaged npz. Creates the index file if needed.

    Safe for sequential invocations; not safe for simultaneous writers on
    different nodes (not a use case for the laptop-side pipeline).
    """
    path = index_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()

    # Store data_dir relative to output_root when possible so the index stays
    # valid if the output root is moved (Dardel -> laptop copy, etc.).
    dd = Path(data_dir)
    try:
        dd = dd.resolve().relative_to(Path(output_root).resolve())
    except ValueError:
        pass  # outside output_root: store the absolute path

    row = {
        "timestamp": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "defect": _fmt(defect),
        "cell": _fmt(cell),
        "method": _fmt(method),
        "order": _fmt(order),
        "pert": _fmt(pert),
        "n_dims": "+".join(str(d) for d in sorted(set(n_dims))) or "-1",
        "npz_files": ";".join(npz_files) or "-1",
        "git_hash": _git_hash(),
        "data_dir": str(dd),
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
