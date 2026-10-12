"""Generate a LaTeX table of ground-state D values from packaged ZFS data.

Inputs are defect folders (searched recursively for *_raw_zfs_data_*.npz) or
npz files produced by ``beyblade package``.

    Defect | Configuration | D_experiment | D_corrected | D_contaminated

D_corrected  comes from the defect-band (ZFS_occup) data.
D_contaminated comes from the all-bands (ZFS_hyp) data.
Experimental values are left blank for manual entry.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from beyblade.models import SPIN_CORRECTION_SCALE, RawZFSData, ZFSTensor  # noqa: E402
from beyblade.parsers import parse_outcar_zfs  # noqa: E402

# Method name fragments in the packaged npz filenames -> table columns.
METHOD_COLUMN = {
    "all_bands": "contaminated",   # ZFS_hyp: all-electron bands, no charge correction applied
    "defect_band_approx": "corrected",  # ZFS_occup: defect-band approximation, charge-corrected
}

ORIENTATION_RE = re.compile(r"(?<![A-Za-z])(k{1,2}|hk|kh)(?![A-Za-z])", re.IGNORECASE)


def infer_orientation(path: Path) -> str:
    """Pull the orientation token (k, kk, hk, kh) out of a path, or '' if absent."""
    match = ORIENTATION_RE.search("/".join(path.parts))
    return match.group(1).lower() if match else ""


def infer_cell(path: Path) -> str:
    """Pull the supercell size (e.g. '512' in NV_512_k) out of a path, or '' if absent."""
    match = re.search(r"(?<![\dA-Za-z])(1?\d{2,3})(?![\dA-Za-z])", "/".join(path.parts))
    return match.group(1) if match else ""


def find_npz(folder: Path) -> list[Path]:
    return sorted(folder.rglob("*_raw_zfs_data_*.npz"))


def method_of(npz_path: Path) -> str | None:
    name = npz_path.name
    for fragment, column in METHOD_COLUMN.items():
        if fragment in name:
            return column
    return None


def collect_rows(paths: list[Path]) -> list[dict]:
    """Build one row per data source: packaged npz files and/or relaxed-structure
    ground-state folders containing ZFS_hyp/OUTCAR and/or ZFS_occup/OUTCAR."""
    rows: list[dict] = []
    for path in paths:
        if path.is_dir():
            rows += collect_outcar_rows(path)
        npz_files = find_npz(path) if path.is_dir() else [path]
        for npz_path in npz_files:
            try:
                data = RawZFSData.load(str(npz_path))
            except Exception as error:
                print(f"warning: cannot load {npz_path} ({error.__class__.__name__}, "
                      f"possibly a git-lfs pointer stub), skipping", file=sys.stderr)
                continue
            zfs: ZFSTensor | None = data.ground_state_zfs
            if zfs is None:
                print(f"warning: no ground-state ZFS in {npz_path}, skipping", file=sys.stderr)
                continue
            column = method_of(npz_path)
            if column is None:
                print(f"warning: unknown method in {npz_path.name}, skipping", file=sys.stderr)
                continue
            orientation = infer_orientation(npz_path.parent) or infer_orientation(path)
            rows.append({
                "defect": data.defect,
                "configuration": orientation,
                "column": column,
                "cell": infer_cell(npz_path.parent) or infer_cell(path),
                "d_mhz": zfs.to_unit("MHz").D,
                "source": npz_path,
            })
    return rows


def collect_outcar_rows(folder: Path) -> list[dict]:
    """Read ground-state ZFS straight from relaxed-structure folders.

    ``folder`` is either the defect folder itself (containing ZFS_hyp/ and
    ZFS_occup/) or a parent folder containing defect folders; ZFS_* dirs are
    found recursively. ZFS_hyp/OUTCAR is the contaminated (all-electron bands)
    ground state, ZFS_occup/OUTCAR the corrected one (defect-band approx),
    which gets the spin-correction scaling applied here, matching
    RawZFSData.load's behaviour for packaged approx data.
    """
    rows: list[dict] = []
    zfs_dirs = sorted(folder.rglob("ZFS_hyp")) + sorted(folder.rglob("ZFS_occup"))
    for zfs_dir in zfs_dirs:
        defect_root = zfs_dir.parent
        defect = defect_root.name.split("_")[0]
        orientation = infer_orientation(defect_root)
        zfs_name = zfs_dir.name
        column = "contaminated" if zfs_name == "ZFS_hyp" else "corrected"
        scale = 1.0 if zfs_name == "ZFS_hyp" else SPIN_CORRECTION_SCALE
        outcar = zfs_dir / "OUTCAR"
        if not outcar.is_file():
            continue
        try:
            zfs = parse_outcar_zfs(outcar)
        except Exception as error:
            print(f"warning: cannot parse {outcar} ({error.__class__.__name__}), skipping",
                  file=sys.stderr)
            continue
        if zfs is None:
            print(f"warning: no ZFS block in {outcar}, skipping", file=sys.stderr)
            continue
        if scale != 1.0:
            print(f"Applied spin correction scaling: {scale}")
            zfs = ZFSTensor(matrix=zfs.matrix * scale, unit=zfs.unit)
        rows.append({
            "defect": defect,
            "configuration": orientation,
            "column": column,
            "cell": infer_cell(defect_root),
            "d_mhz": zfs.to_unit("MHz").D,
            "source": outcar,
        })
    return rows


def merge_rows(rows: list[dict]) -> list[dict]:
    """Merge corrected/contaminated values belonging to the same defect+orientation."""
    merged: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["defect"], row["configuration"])
        entry = merged.setdefault(key, {
            "defect": row["defect"],
            "configuration": row["configuration"],
            "cell": row["cell"],
            "corrected": None,
            "contaminated": None,
        })
        entry[row["column"]] = row["d_mhz"]
    return list(merged.values())


def render_tex(rows: list[dict], unit: str) -> str:
    lines = [
        r"% generated by scripts/cluster/generate_d_table.py",
        r"% requires \usepackage{booktabs} and \usepackage{siunitx}",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \begin{tabular}{llccc}",
        r"    \toprule",
        r"    Defect & Configuration & $D_\text{experiment}$ & $D_\text{corrected}$ & $D_\text{contaminated}$ \\",
        r"         &              & (\si{\mega\hertz})            & (\si{\mega\hertz})         & (\si{\mega\hertz})             \\",
        r"    \midrule",
    ]
    for row in sorted(rows, key=lambda r: (r["defect"], r["configuration"])):
        defect = row["defect"]
        if row["configuration"]:
            defect = f"{defect}$_\\mathrm{{{row['configuration']}}}$"
        superscript = rf"$^\mathrm{{{row['cell']}}}$" if row["cell"] else ""
        corrected = (
            f"{row['corrected']:.3f}{superscript}" if row["corrected"] is not None else "---"
        )
        contaminated = (
            f"{row['contaminated']:.3f}{superscript}"
            if row["contaminated"] is not None else "---"
        )
        lines.append(
            f"    {defect} & {row['configuration'] or '---'} & --- & {corrected} & {contaminated} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        rf"  \caption{{Zero-field splitting parameter $D$ ({unit}). "
        r"Corrected: defect-band approximation (ZFS\_occup); contaminated: all-electron bands (ZFS\_hyp). Superscripts denote the supercell size.}",
        r"  \label{tab:zfs_d}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("inputs", nargs="+", type=Path, help="Defect folders or packaged ZFS npz files.")
    parser.add_argument("-o", "--output", type=Path, default=Path("D_table.tex"), help="Output .tex file.")
    args = parser.parse_args()

    for path in args.inputs:
        if not path.exists():
            parser.error(f"input not found: {path}")

    rows = collect_rows(args.inputs)
    if not rows:
        sys.exit("error: no packaged ZFS data found in the given inputs")
    merged = merge_rows(rows)
    args.output.write_text(render_tex(merged, "MHz"))
    print(f"wrote {args.output} ({len(merged)} row(s))")


if __name__ == "__main__":
    main()
