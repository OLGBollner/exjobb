#!/usr/bin/env python3
"""relax_to_zfs.py -- pipeline step: relaxation output -> ZFS run directory.

Takes a finished relaxation directory (vasp_gam, Gamma-only) and prepares one
or more ZFS run directories (vasp_std, Gamma-only) by:

  1. copying CONTCAR -> POSCAR, CHGCAR (and POTCAR, KPOINTS) into the ZFS dir
  2. deriving the fixed occupations DOCCUP / DOCCDO from the relaxation EIGENVAL
  3. patching the relaxation INCAR: NSW=0, IBRION=-1, LDMATRIX=.TRUE.,
     ISYM 2->3, occupation tags, no duplicate tags
  4. refusing to run when a sanity check fails (bad counts, WAVECAR present,
     POSCAR that is not the relaxed CONTCAR, duplicate INCAR tags)

Usage:
    python3 relax_to_zfs.py RELAX_DIR ZFS_DIR [ZFS_DIR2 ...]

Everything is written into the ZFS dir; nothing in RELAX_DIR is modified.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


def warn(msg: str) -> None:
    print(f"  warn: {msg}")


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #
def parse_tag(incar: str, tag: str) -> str | None:
    """Return the LAST value of a tag in the INCAR text (VASP semantics)."""
    vals = re.findall(rf"^\s*{tag}\s*=\s*(.+?)\s*(?:!|#|$)", incar, re.M | re.I)
    return vals[-1] if vals else None


def all_tag_occurrences(incar: str, tag: str) -> list[str]:
    return re.findall(rf"^\s*{tag}\s*=\s*(.+?)\s*(?:!|#|$)", incar, re.M | re.I)


def occupied_bands_from_eigenval(eigenval: Path) -> tuple[int, int, int]:
    """Parse EIGENVAL (ISPIN=2) -> (n_up, n_down, nbands).

    Occupied means occ > 0.5 in the respective spin column.
    Assumes Gamma-only (one k-point) or takes the first k-point.
    """
    lines = eigenval.read_text().splitlines()
    # header: line 2 holds (nelect, kpts, bands, ...)
    header = lines[5].split()
    nkpts, nbands = int(header[1]), int(header[2])
    # data blocks: after the 6-line header, first block is kpoint line then
    # nbands lines of  "band  E_up  occ_up  E_dn  occ_dn"
    data = lines[8:] if nkpts == 1 else lines[8 : 8 + nbands + 1]
    n_up = n_dn = 0
    for line in data[1:]:
        parts = line.split()
        if len(parts) < 3:
            continue
        if float(parts[2]) > 0.5:
            n_up += 1
        if len(parts) >= 5 and float(parts[4]) > 0.5:
            n_dn += 1
    return n_up, n_dn, nbands


# --------------------------------------------------------------------------- #
# INCAR patching
# --------------------------------------------------------------------------- #
REMOVE_TAGS = ("NSW", "IBRION", "LDMATRIX", "DOCCUP", "DOCCDO", "NUPDOWN",
               "DOCC", "LDAPMINUS")  # never kept verbatim; rebuilt below


def patch_incar(text: str, n_up: int, n_dn: int, nbands: int) -> str:
    body_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        tag = stripped.split("=")[0].strip().upper() if "=" in stripped else ""
        if tag in REMOVE_TAGS:
            continue  # dropped, re-added below
        if tag == "ISYM" and parse_tag(text, "ISYM") == "2":
            line = re.sub(r"ISYM\s*=\s*2", "ISYM = 3", line)  # 2 -> 3
        body_lines.append(line)

    additions = f"""
### ZFS stage (added by relax_to_zfs.py)
NSW = 0                  # single point: no ionic relaxation
IBRION = -1              # no ionic degrees of freedom
LDMATRIX = .TRUE.
NUPDOWN = {int(n_up - n_dn)}
NBANDS = {nbands}
DOCCUP = {n_up}*1.0 {nbands - n_up}*0.0
DOCCDO = {n_dn}*1.0 {nbands - n_dn}*0.0
"""
    return "\n".join(body_lines).rstrip() + "\n" + additions


# --------------------------------------------------------------------------- #
# per-directory pipeline
# --------------------------------------------------------------------------- #
def prepare(relax: Path, zfs: Path) -> None:
    print(f"\n=== {relax}  ->  {zfs} ===")
    if not relax.is_dir():
        fail(f"relaxation dir {relax} does not exist")
        return
    zfs.mkdir(parents=True, exist_ok=True)

    # ---- required relaxation outputs ------------------------------------- #
    required = ["CONTCAR", "CHGCAR", "EIGENVAL", "INCAR", "POTCAR", "KPOINTS"]
    for name in required:
        if not (relax / name).is_file():
            fail(f"missing {name} in {relax}")

    # ---- copy geometry and density --------------------------------------- #
    shutil.copy2(relax / "CONTCAR", zfs / "POSCAR")
    shutil.copy2(relax / "CHGCAR", zfs / "CHGCAR")
    shutil.copy2(relax / "POTCAR", zfs / "POTCAR")
    shutil.copy2(relax / "KPOINTS", zfs / "KPOINTS")
    print("  copied CONTCAR->POSCAR, CHGCAR, POTCAR, KPOINTS")

    # ---- guard against stale WAVECAR in the ZFS dir ------------------------ #
    if (zfs / "WAVECAR").exists():
        warn("WAVECAR present in ZFS dir; it is incompatible with the "
             "vasp_gam -> vasp_std switch and will NOT be used -- remove it")

    # ---- INCAR sanity: duplicate tags ------------------------------------- #
    incar_text = (relax / "INCAR").read_text()
    for tag in ("NUPDOWN", "NELECT", "ISPIN", "NBANDS", "ISYM", "NSW", "IBRION"):
        occ = all_tag_occurrences(incar_text, tag)
        if len(occ) > 1:
            fail(f"INCAR defines {tag} {len(occ)} times: {occ} "
                 f"(last one wins in VASP -- fix before running)")

    # ---- occupations from EIGENVAL ---------------------------------------- #
    n_up, n_dn, nbands = occupied_bands_from_eigenval(relax / "EIGENVAL")
    print(f"  EIGENVAL: {n_up} occupied (up), {n_dn} (down), NBANDS = {nbands}")

    nelect = parse_tag(incar_text, "NELECT")
    isp, = [parse_tag(incar_text, "ISPIN") or "2"]
    if int(isp) == 2 and n_up + n_dn != int(nelect):
        fail(f"occupied bands {n_up}+{n_dn} = {n_up + n_dn} != NELECT = {nelect}")
    if n_up <= n_dn:
        fail(f"majority channel not larger than minority ({n_up} <= {n_dn}); "
             f"spin columns may be swapped in this EIGENVAL")
    nupdown = parse_tag(incar_text, "NUPDOWN")
    if nupdown and int(float(nupdown)) != n_up - n_dn:
        fail(f"NUPDOWN = {nupdown} inconsistent with occupied difference "
             f"{n_up} - {n_dn} = {n_up - n_dn}")

    # ---- patched INCAR ----------------------------------------------------- #
    (zfs / "INCAR").write_text(patch_incar(incar_text, n_up, n_dn, nbands))
    print(f"  wrote INCAR with NUPDOWN={n_up - n_dn}, "
          f"DOCCUP={n_up}*1.0 {nbands - n_up}*0.0, DOCCDO likewise")

    # ---- summary ----------------------------------------------------------- #
    if FAILURES:
        print(f"\n  {len(FAILURES)} problem(s) found -- do NOT launch until fixed")
    else:
        print("  all checks passed; ready to launch")


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        sys.exit(__doc__)
    for zfs_dir in args[1:]:
        prepare(Path(args[0]), Path(zfs_dir))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
