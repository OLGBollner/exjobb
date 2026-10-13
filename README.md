# spinlabs

Analysis pipeline for spin–phonon coupling and spin relaxation in defect
systems (NV, ClV), built on phonon and zero-field-splitting (ZFS) data from
VASP simulations. Written by Hendriks et al., companion code to the
accompanying paper.

## Installation

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e .

Python >= 3.10. Pinned versions of every dependency (matching the
environment used for the results in the paper) are in `requirements.lock`;
install them with `pip install -r requirements.lock` for a reproducible
environment.

## Quick start

Run the full pipeline (coupling → transition rates → T1 relaxation) on the
bundled NV 64-atom all-bands dataset, order 2, perturbation scale 0.1,
0–300 K, and plot:

    spinlabs run --raw-zfs-file NV_64/NV_64_raw_zfs_data_all_bands_1d.npz \
        NV_64/NV_64_raw_zfs_data_all_bands_2d.npz \
        --phonon-file NV_64/phonon_data.npz \
        --method all --order 2 --pert-scale 0.1 \
        --defect NV --cell-size 64 \
        --t-start 0 --t-end 300 --t-step 1 --plot

The method, defect name and cell size are inferred from the file names
when omitted, so a minimal equivalent is:

    spinlabs run --raw-zfs-file NV_64/NV_64_raw_zfs_data_all_bands_1d.npz \
        NV_64/NV_64_raw_zfs_data_all_bands_2d.npz \
        --phonon-file NV_64/phonon_data.npz \
        --order 2 --pert-scale 0.1 \
        --t-start 0 --t-end 300 --plot

Runtime is a few seconds on a single core. Reference value with these
inputs: T1(300 K) ≈ 22.1 ms.

## Input modes

The `run` command accepts exactly one of:

- `--sim-folder DIR...` — VASP simulation folders (OUTCAR etc.); the
  coupling is computed from scratch (slow).
- `--raw-zfs-file FILE...` — pre-computed raw ZFS `.npz` dataset(s); the
  usual mode.
- `--raw-zfs-file-1d` + `--raw-zfs-file-2d` — combined 1d+2d mode.
- `--coupling-file FILE` — pre-computed spin–phonon coupling `.npz`.

`--method` is `all` (all_bands, ZFS_hyp) or `approx`
(defect_band_approx, ZFS_occup); it is inferred from the input file name
when omitted. `--order` is 1 or 2. Temperature grid: `--t-start`,
`--t-end`, `--t-step` or an explicit `--temperatures` list.

## Output

Each run writes a directory under `runs/<defect>_<size>/<method>/
pert_<scale>/<timestamp>/` containing:

- `run_info.json` — inputs, options and versions
- `spin_phonon_coupling.npz`
- `transition_rates.npz` (`first_order`, `second_order`, `two_phonon`)
- `t1_relaxation.npz` (`temperatures`, `t1_fit`)
- figures (with `--plot`)

## Repository layout

    src/spinlabs/        pipeline, CLI, models, parsers, plotting
    tests/               pytest suite (422 tests)
    NV_64/ ClV_128/ NV_512/  bundled VASP-derived datasets
    scripts/             rates files and helper scripts
    config/              configuration

## Development

    pip install -e ".[dev]"
    pytest tests

If the tests fail with a missing module when run under a bare `python`,
make sure the virtual environment's interpreter is first on `PATH`
(`source .venv/bin/activate`).
