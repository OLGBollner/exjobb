#!/usr/bin/env python3
"""create_perturbation_tree.py -- build ready-to-go perturbed ZFS run folders.

Creates, inside a defect folder:

    <defect>/
        first_order/
            pert_<scale>/
                all_bands/
                    input/            INCAR, POSCAR, KPOINTS, POTCAR, CHGCAR
                    run_perturbation_first_order.sh
                defect_band_approx/
                    input/            same files
                    run_perturbation_first_order.sh
        second_order/
            pert_<scale>/
                all_bands/ ...
                defect_band_approx/ ...

The defect name is read from the --output folder name.

Input files are copied verbatim, no rewriting:
  - all_bands input  <- <output>/ZFS_hyp    (fully occupied bands)
  - defect_band_approx input <- <output>/ZFS_occup (defect-band occupation)

Both ZFS folders must exist; otherwise the script aborts before creating
anything.  CHGCAR is copied when present.  Without --force, existing pert_<scale>
folders are skipped, never overwritten; --force regenerates them after an
explicit APPROVE confirmation.

Each generated SLURM script has DEFECT, PHONON_PATH and PERT prefilled with
absolute paths, plus a job name of the form <defect>_<order>_<basis>.  The
reference ZFS simulation's OUTCAR under each source folder is parsed for its
elapsed wall time; the script's time limit and array size are then computed:

    per_sim_time * n_tasks / n_array_jobs

where n_tasks is the number of single-mode (first order) or mode-pair
(second order) simulations.  The number of array jobs is set with
--array-jobs.

Usage:
    python create_perturbation_tree.py --phonon NV_512/data/phonon_data.npz \
        [--pert 0.025 0.05] [--output /path/to/NV_512] [--array-jobs 10]
"""
from __future__ import annotations

import argparse
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

FIRST_ORDER_SCRIPT = "run_perturbation_first_order.sh"
SECOND_ORDER_SCRIPT = "run_perturbation_second_order.sh"
COPIED_FILES = ("INCAR", "POSCAR", "KPOINTS", "POTCAR", "CHGCAR")
REQUIRED_FILES = COPIED_FILES[:4]  # CHGCAR optional

# basis -> source ZFS folder (relative to the defect/output folder)
BASIS_SOURCE = {
    "all_bands": "ZFS_hyp",
    "defect_band_approx": "ZFS_occup",
}

FAILURES: list[str] = []


def fail(msg: str) -> None:
    FAILURES.append(msg)
    print(f"  FAIL: {msg}")


def replace_tag(incar: str, tag: str, value: str) -> str:
    """Replace the value of every occurrence of a tag."""
    return re.sub(rf"^(\s*{tag}\s*=\s*)(.+?)[ \t]*(?=$|!|#)",
                  rf"\g<1>{value}", incar, flags=re.M | re.I)


def replace_sbatch(text: str, directive: str, value: str) -> str:
    """Replace the value of an #SBATCH directive (e.g. -J, -t, -a)."""
    return re.sub(rf"(#SBATCH\s+{re.escape(directive)}\s+).*",
                  rf"\g<1>{value}", text, flags=re.M)


def read_outcar_time(src: Path) -> float | None:
    """Elapsed VASP wall time (sec) from the top-level OUTCAR under src."""
    outcar = src / "OUTCAR"
    if not outcar.is_file():
        return None
    m = re.findall(r"Elapsed time \(sec\):\s*([\d\.]+)",
                   outcar.read_text(errors="replace"))
    return float(m[-1]) if m else None


def sbatch_time(seconds: float) -> str:
    """Format seconds as a SLURM time limit, rounded up to whole minutes."""
    minutes = max(1, math.ceil(seconds / 60))
    h, m = divmod(minutes, 60)
    return f"{h}:{m:02d}:00"


def n_modes_from_phonon(phonon: Path) -> int:
    """Number of phonon modes to perturb along (same logic as get_n_modes)."""
    with np.load(phonon) as data:
        if "idx" in data:
            return len(data["idx"])
        return int(data["freqs"].shape[0])


def n_tasks_for(order: str, pair_mode: str | None, n_modes: int) -> int:
    """Number of simulations a full perturbation sweep requires."""
    if order == "first_order":
        return n_modes
    if pair_mode == "all":
        return n_modes * (n_modes + 1) // 2
    return n_modes  # diag


# --------------------------------------------------------------------------- #
# directory preparation
# --------------------------------------------------------------------------- #
def prepare_basis(src: Path, dst: Path, script_name: str, phonon: Path,
                  defect: str, pert: float, order: str,
                  n_array_jobs: int, n_modes: int, basis: str,
                  vasp_binary: Path | None = None,
                  pair_mode: str | None = None,
                  max_hours: float | None = None) -> None:
    """Fill dst/input/ and write the prefilled SLURM script."""
    inp = dst / "input"
    inp.mkdir(parents=True, exist_ok=True)

    for name in COPIED_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, inp / name)

    tmpl = Path(__file__).resolve().parent / script_name
    if not tmpl.is_file():
        fail(f"missing template script {tmpl}")
        return

    # timing: per-sim wall time from the reference ZFS run's OUTCARs
    per_sim = read_outcar_time(src)
    n_tasks = n_tasks_for(order, pair_mode, n_modes)
    if per_sim is None:
        fail(f"no Elapsed time found in OUTCARs under {src} -- "
             "leaving the template time limit and array size untouched")
    total = per_sim * n_tasks if per_sim is not None else None
    per_job = total / n_array_jobs if total is not None else None
    if per_job is not None and max_hours is not None:
        cap = max_hours * 3600
        if per_job > cap:
            n_array_jobs = max(n_array_jobs,
                               math.ceil(total / cap))
            per_job = total / n_array_jobs

    text = tmpl.read_text()
    text = replace_tag(text, "DEFECT", defect)
    text = replace_tag(text, "PHONON_PATH", str(phonon.resolve()))
    text = replace_tag(text, "PERT", str(pert))
    if vasp_binary:
        text = replace_tag(text, "BINARY", str(vasp_binary))
    if pair_mode:
        text = replace_tag(text, "PAIR_MODE", pair_mode)
    cluster_scripts_dir = Path(__file__).resolve().parent
    name = "create_combined_phonon_struct.py" \
        if "second_order" in script_name else "create_phonon_struct.py"
    text = replace_tag(text, "CREATE_STRUCT",
                       str(cluster_scripts_dir / name))
    text = replace_tag(text, "GET_N_MODES",
                       str(cluster_scripts_dir / "get_n_modes.py"))

    # job name reflects defect, order and basis
    text = replace_sbatch(text, "-J", f"{defect}_{order}_{basis}")
    if per_job is not None:
        text = replace_sbatch(text, "-t", sbatch_time(per_job))
    text = replace_sbatch(text, "-a", f"0-{n_array_jobs - 1}")

    (dst / script_name).write_text(text)

    if per_sim is not None:
        print(f"    time: {per_sim:.0f} s/sim x {n_tasks} tasks / "
              f"{n_array_jobs} array jobs = {per_job:.0f} s/job "
              f"-> {sbatch_time(per_job)}")



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phonon", type=Path, required=True,
                    help="phonon npz used as-is for both orders")
    ap.add_argument("--pert", type=float, nargs="+", default=[0.025],
                    help="one or more perturbation scales (default: 0.025)")
    ap.add_argument("--output", type=Path, default=None,
                    help="defect folder to write into; the basename is used as "
                         "the defect name (default: ./<defect> requires a name)")
    ap.add_argument("--vasp-binary", type=Path, default=None,
                    help="fill the BINARY placeholder with this path")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing perturbation folders instead of "
                         "skipping them; requires typing APPROVE to confirm")
    ap.add_argument("--pair-mode", choices=("diag", "all"), default="diag",
                    help="second-order pairs: diag = only (i, i) terms (the "
                         "original behaviour), all = every pair with i <= j "
                         "(default: diag)")
    ap.add_argument("--array-jobs", type=int, default=10,
                    help="number of SLURM array jobs to split the sweep into "
                         "(default: 10)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="cap on the per-job time limit in hours; the array "
                         "job count is raised until the limit fits (e.g. "
                         "--max-hours 8)")
    args = ap.parse_args()

    if args.force:
        try:
            answer = input("Overwrite existing perturbation folders? "
                           "Type APPROVE to continue: ")
        except EOFError:
            answer = ""
        if answer.strip() != "APPROVE":
            sys.exit("Aborted: overwrite not confirmed")

    if args.output is None:
        sys.exit("Error: --output is required (the defect name is read from it)")
    out = args.output.resolve()
    defect = out.name
    phonon = args.phonon.resolve()

    if not phonon.is_file():
        sys.exit(f"Error: phonon file {phonon} does not exist")

    # both source ZFS folders must exist before anything is created
    sources: dict[str, Path] = {}
    for basis, folder in BASIS_SOURCE.items():
        src = out / folder
        if not src.is_dir():
            sys.exit(f"Error: missing {src} -- both ZFS_hyp and ZFS_occup must "
                     "exist before creating perturbation folders")
        for name in REQUIRED_FILES:
            if not (src / name).is_file():
                sys.exit(f"Error: missing {name} in {src}")
        sources[basis] = src

    n_modes = n_modes_from_phonon(phonon)
    print(f"=== {defect}: phonon {phonon} ({n_modes} modes) -> {out} ===")
    for order in ("first_order", "second_order"):
        for basis, src in sources.items():
            for pert in args.pert:
                dst = out / order / f"pert_{pert}" / basis
                if dst.exists():
                    if args.force:
                        print(f"  overwriting {dst}")
                        shutil.rmtree(dst)
                    else:
                        print(f"  note: {dst} already exists -- skipping")
                        continue
                print(f"  creating {dst}")
                script_name = FIRST_ORDER_SCRIPT if order == "first_order" \
                    else SECOND_ORDER_SCRIPT
                prepare_basis(src, dst, script_name, phonon,
                              defect=defect, pert=pert, order=order,
                              n_array_jobs=args.array_jobs,
                              n_modes=n_modes, basis=dst.name,
                              vasp_binary=args.vasp_binary,
                              pair_mode=args.pair_mode if order == "second_order" else None,
                              max_hours=args.max_hours)

    if FAILURES:
        print(f"\n{len(FAILURES)} problem(s) found -- inspect before launching")
        return 1
    print("\nDone. All folders ready for sbatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
