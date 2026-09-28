#!/usr/bin/env python3
"""create_perturbation_dirs.py -- build ready-to-go perturbed ZFS run folders.

Creates, inside a defect folder:

    <defect>/
        first_order/
            pert_<scale>/
                all_bands/
                    input/            INCAR, POSCAR, KPOINTS, POTCAR, CHGCAR
                    run_perturbation_first_order.sh
                defect_band_approx/
                    input/            same files, DOCCUP/DOCCDO reduced to
                                      the unpaired bands only
                    run_perturbation_first_order.sh
        second_order/
            pert_<scale>/
                all_bands/ ...
                defect_band_approx/ ...

The reference input files come from a completed pristine ZFS run directory
(--source, normally the defect's ZFS_occup or ZFS_hyp folder): the run must
contain EIGENVAL, INCAR, POSCAR, KPOINTS, POTCAR.  CHGCAR is copied when
present.

For defect_band_approx the DOCCUP/DOCCDO lines are rewritten so that only the
unpaired bands (up occupied but down not, or vice versa, read off EIGENVAL)
stay occupied; the number of unpaired electrons must match |NUPDOWN| in the
INCAR or the script refuses to run.

Each generated SLURM script has DEFECT, PHONON_PATH and PERT prefilled with
absolute paths, so it is ready for `sbatch` as-is.

Usage:
    python create_perturbation_dirs.py DEFECT --phonon PHONON_NPZ \
        [--pert 0.025 0.05] [--source /path/to/ZFS_occup] \
        [--output /path/to/defect]
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

FIRST_ORDER_SCRIPT = "run_perturbation_first_order.sh"
SECOND_ORDER_SCRIPT = "run_perturbation_second_order.sh"
COPIED_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "CHGCAR")
REQUIRED_FILES = ("EIGENVAL",) + COPIED_FILES[:4]  # CHGCAR optional

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


# --------------------------------------------------------------------------- #
# INCAR / EIGENVAL helpers (mirrors relax_to_zfs.py)
# --------------------------------------------------------------------------- #
def parse_tag(incar: str, tag: str) -> str | None:
    """Return the LAST value of a tag in the INCAR text (VASP semantics)."""
    vals = re.findall(rf"^\s*{tag}\s*=\s*(.+?)\s*(?:!|#|$)", incar, re.M | re.I)
    return vals[-1] if vals else None


def replace_tag(incar: str, tag: str, value: str) -> str:
    """Replace the value of every occurrence of a tag."""
    return re.sub(rf"^(\s*{tag}\s*=\s*)(.+?)\s*(?=$|!|#)",
                  rf"\g<1>{value}", incar, flags=re.M | re.I)


def occupied_bands_from_eigenval(eigenval: Path) -> tuple[int, int, int, set[int], set[int]]:
    """Parse EIGENVAL (ISPIN=2) -> (n_up, n_dn, nbands, unpaired_up, unpaired_dn).

    Occupied means occ > 0.5 in the respective spin column.  Assumes
    Gamma-only (one k-point) or takes the first k-point.  Unpaired bands are
    occupied in one channel only.
    """
    lines = eigenval.read_text().splitlines()
    # header: line 6 (index 5) holds (nelect, kpts, bands, ...)
    header = lines[5].split()
    nkpts, nbands = int(header[1]), int(header[2])
    data = lines[7:] if nkpts == 1 else lines[7 : 7 + nbands + 1]
    rows = [ln.split() for ln in data[1:] if len(ln.split()) >= 3]
    # find the occupation columns: values all within [0, 1].  Energies are
    # typically negative, so this disambiguates the two possible layouts
    # (band E_up occ_up E_dn occ_dn  vs  band E_up E_dn occ_up occ_dn).
    cand = [c for c in range(1, len(rows[0]))
            if all(-0.01 <= float(r[c]) <= 1.01 for r in rows)]
    if len(cand) < 2:
        sys.exit(f"Error: could not locate occupation columns in {eigenval}")
    occ_col_up, occ_col_dn = cand[0], cand[1]
    band_col = [c for c in range(len(rows[0])) if c not in cand][0]
    n_up = n_dn = 0
    unpaired_up: set[int] = set()
    unpaired_dn: set[int] = set()
    for parts in rows:
        occ_up = float(parts[occ_col_up]) > 0.5
        occ_dn = float(parts[occ_col_dn]) > 0.5
        n_up += occ_up
        n_dn += occ_dn
        if occ_up and not occ_dn:
            unpaired_up.add(int(parts[band_col]))
        elif occ_dn and not occ_up:
            unpaired_dn.add(int(parts[band_col]))
    return n_up, n_dn, nbands, unpaired_up, unpaired_dn


def run_pattern(indices: set[int], nbands: int) -> str:
    """VASP run-length occupation string from 1-based band indices.

    E.g. {1023, 1024} with nbands=2048 -> "1022*0.0 2*1.0 1024*0.0".
    """
    if not indices:
        return f"{nbands}*0.0"
    parts = []
    pos = 1
    for i in sorted(indices):
        gap = i - pos
        if gap:
            parts.append(f"{gap}*0.0")
        parts.append("1*1.0" if i == pos else None)  # placeholder, merged below
        pos = i + 1
    # merge consecutive 1*1.0 entries (consecutive unpaired bands)
    merged: list[str] = []
    for p in parts:
        if p is None:
            continue
        if p == "1*1.0" and merged and merged[-1].endswith("*1.0"):
            n = int(merged[-1].split("*")[0]) + 1
            merged[-1] = f"{n}*1.0"
        elif p == "1*1.0":
            merged.append("1*1.0")
        else:
            merged.append(p)
    tail = nbands - (max(indices))
    if tail:
        merged.append(f"{tail}*0.0")
    return " ".join(merged)


# --------------------------------------------------------------------------- #
# directory preparation
# --------------------------------------------------------------------------- #
def prepare_basis(src: Path, dst: Path, basis: str, order: str, phonon: Path,
                  defect: str, pert: float) -> None:
    """Fill dst/input/ and write the prefilled SLURM script."""
    inp = dst / "input"
    inp.mkdir(parents=True, exist_ok=True)

    if basis == "defect_band_approx":
        incar_text = defect_band_incar(src, (src / "INCAR").read_text())

    for name in COPIED_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, inp / name)
    if basis == "defect_band_approx":
        (inp / "INCAR").write_text(incar_text)

    if order == "first":
        script_name = FIRST_ORDER_SCRIPT
    else:
        script_name = SECOND_ORDER_SCRIPT
    tmpl = Path(__file__).resolve().parent / script_name
    if not tmpl.is_file():
        fail(f"missing template script {tmpl}")
        return
    text = tmpl.read_text()
    text = replace_tag(text, "DEFECT", defect)
    text = replace_tag(text, "PHONON_PATH", str(phonon.resolve()))
    text = replace_tag(text, "PERT", str(pert))
    (dst / script_name).write_text(text)


def defect_band_incar(src: Path, incar_text: str) -> str:
    """Rewrite DOCCUP/DOCCDO so only unpaired bands stay occupied."""
    n_up, n_dn, nbands, unp_up, unp_dn = occupied_bands_from_eigenval(src / "EIGENVAL")
    nupdown = parse_tag(incar_text, "NUPDOWN")
    if nupdown is None:
        sys.exit("Error: INCAR has no NUPDOWN; cannot verify unpaired band count")
    if int(float(nupdown)) != len(unp_up) - len(unp_dn):
        sys.exit(f"Error: unpaired bands from EIGENVAL ({len(unp_up)} up, "
                 f"{len(unp_dn)} down) do not match NUPDOWN = {nupdown} "
                 "-- refusing to generate defect_band_approx INCAR")
    print(f"  EIGENVAL: {n_up} occ up, {n_dn} occ dn, unpaired up {sorted(unp_up)}, "
          f"down {sorted(unp_dn)}")
    incar_text = replace_tag(incar_text, "DOCCUP", run_pattern(unp_up, nbands))
    incar_text = replace_tag(incar_text, "DOCCDO", run_pattern(unp_dn, nbands))
    return incar_text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("defect", help="defect name, e.g. ClV_128")
    ap.add_argument("--phonon", type=Path, required=True,
                    help="phonon npz used as-is for both orders")
    ap.add_argument("--pert", type=float, nargs="+", default=[0.025],
                    help="one or more perturbation scales (default: 0.025)")
    ap.add_argument("--source", type=Path, default=None,
                    help="completed pristine ZFS run dir (default: ZFS_occup "
                         "next to --output)")
    ap.add_argument("--output", type=Path, default=None,
                    help="defect folder to write into (default: ./<defect>)")
    args = ap.parse_args()

    out = (args.output or Path(args.defect)).resolve()
    src = (args.source or out.parent / "ZFS_occup").resolve()
    phonon = args.phonon.resolve()

    if not src.is_dir():
        sys.exit(f"Error: source ZFS dir {src} does not exist (--source ...)")
    if not phonon.is_file():
        sys.exit(f"Error: phonon file {phonon} does not exist")
    for name in REQUIRED_FILES:
        if not (src / name).is_file():
            sys.exit(f"Error: missing {name} in {src}")

    print(f"=== {args.defect}: source {src}, phonon {phonon} -> {out} ===")
    for order in ("first_order", "second_order"):
        for basis in ("all_bands", "defect_band_approx"):
            for pert in args.pert:
                dst = out / order / f"pert_{pert}" / basis
                if dst.exists():
                    print(f"  note: {dst} already exists -- skipping")
                    continue
                print(f"  creating {dst}")
                prepare_basis(src, dst, basis, order.split("_")[0], phonon,
                              args.defect, pert)

    if FAILURES:
        print(f"\n{len(FAILURES)} problem(s) found -- inspect before launching")
        return 1
    print("\nDone. All folders ready for sbatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
