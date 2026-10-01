"""Static verification of a generated perturbation-run folder.

Checks the INCAR and the prefilled SLURM script in a run folder so that
problems are caught *before* sbatch queues the job, not hours later in
the scheduler.  The checks mirror the preflight in the run templates but
run locally at generation time.
"""
from __future__ import annotations

import re
from pathlib import Path

PLACEHOLDER = re.compile(r"<[^<>\s]+>")


def _read(path: Path) -> str:
    return path.read_text() if path.is_file() else ""


def _tag_value(incar: str, tag: str) -> str | None:
    m = re.search(rf"^\s*{tag}\s*=\s*(\S+)", incar, re.MULTILINE | re.IGNORECASE)
    return m.group(1) if m else None


def verify_setup(folder: Path, script_name: str) -> list[str]:
    """Verify the folder's INCAR and SLURM script.

    Returns a list of problem descriptions (empty when everything is
    consistent).  Path problems are FAILs; INCAR policy issues that only
    risk slow or unusual runs are WARNINGs.
    """
    problems: list[str] = []
    folder = folder.resolve()

    # --- SLURM script: no leftover template placeholders -------------------
    script = folder / script_name
    if not script.is_file():
        problems.append(f"FAIL: missing run script {script}")
    else:
        text = _read(script)
        for line_no, line in enumerate(text.splitlines(), 1):
            for ph in PLACEHOLDER.findall(line):
                problems.append(f"FAIL: {script_name}:{line_no} "
                                f"placeholder not filled: {ph}")
        # path existence for the four referenced files
        for key in ("BINARY", "PHONON_PATH", "CREATE_STRUCT", "GET_N_MODES"):
            vals = re.findall(rf"^{key}=(.*)$", text.lower(), re.MULTILINE)
            if not vals:
                problems.append(f"FAIL: {script_name}: {key} not set")
                continue
            val = vals[-1].strip().strip("'\"")
            if not val or PLACEHOLDER.search(val):
                continue  # placeholder already reported above
            if not Path(val).exists():
                problems.append(f"FAIL: {script_name}: {key} path missing: {val}")

    # --- INCAR: policy checks ---------------------------------------------
    incar = folder / "input" / "INCAR"
    if not incar.is_file():
        problems.append(f"FAIL: missing {incar}")
    else:
        itext = _read(incar)
        if (_tag_value(itext, "LDMATRIX") or "").lower() != ".true.":
            problems.append("FAIL: INCAR LDMATRIX is not .TRUE. -- no ZFS tensor")
        if (_tag_value(itext, "ALGO") or "").lower() == "none":
            problems.append("FAIL: INCAR ALGO=None is not self-consistent -- "
                            "eigenvalues/occupations are meaningless")
        ibrion = _tag_value(itext, "IBRION")
        nsw = int(_tag_value(itext, "NSW") or "0")
        if ibrion is not None and int(ibrion) >= 0 and nsw > 0:
            problems.append(f"FAIL: INCAR relaxes ions (IBRION={ibrion}, "
                            "NSW>0) -- ZFS must be static")
        isym = _tag_value(itext, "ISYM")
        if isym is not None and int(isym) in (1, 2) and \
                (_tag_value(itext, "LDMATRIX") or "").lower() == ".true.":
            problems.append("FAIL: INCAR ISYM=1/2 together with LDMATRIX "
                            "-- use ISYM=3")
        gam = "gam" in (_tag_value(_read(script), "BINARY") or "").lower()
        if gam:
            problems.append("WARNING: the SLURM script uses a gamma-only "
                            "VASP build; 5.4.4 gamma-only builds produced "
                            "broken D tensors with NCORE>1 (issue #33)")
        if (ncore := _tag_value(itext, "NCORE")) is not None and gam:
            problems.append(f"WARNING: gamma-only build with NCORE={ncore} "
                            "-- untested combination for ZFS")
    return problems
