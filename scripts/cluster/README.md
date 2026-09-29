# Cluster run scripts

Parameterized SLURM array scripts for ZFS perturbation runs, shared across
defects. Copy the relevant script into the defect's run folder (e.g.
`first_order/pert_0.025/all_bands/`) with `input/` next to it.

Config is set in the config block near the top or overridden per submission:

```bash
DEFECT=NV_512 PHONON_PATH=/cfs/klemming/.../NV_512/data/phonon_data.npz sbatch run_perturbation_first_order.sh
```

Variables: `DEFECT`, `BINARY`, `CREATE_STRUCT`, `GET_N_MODES`, `PHONON_PATH`,
`PERT`. Defaults currently point at the ClV_128 layout; update
`PHONON_PATH` for each new defect.

- `run_perturbation_first_order.sh`: single-mode perturbations via
  `create_phonon_struct.py`, run dirs `runs/<mode>`.
- `run_perturbation_second_order.sh`: mode-pair perturbations via
  `create_combined_phonon_struct.py`, run dirs `runs/<i>_<j>`.

Both skip run dirs whose OUTCAR already contains a complete
spin-spin ZFS tensor block, so restarts resume where they stopped.
- `create_perturbation_tree.py` -- build ready-to-go perturbed run trees (`first_order`/`second_order` x `all_bands`/`defect_band_approx` x pert scales). Defect name is read from the --output folder; input files are copied verbatim from `<output>/ZFS_hyp` (all_bands) and `<output>/ZFS_occup` (defect_band_approx); both must exist or the script aborts. Existing pert_<scale> folders are never overwritten unless `--force` is passed, which requires typing `APPROVE` to confirm.
  - SLURM templates (`run_perturbation_first_order.sh` / `_second_order.sh`) ship with placeholders (`<defect>`, `<vasp_binary>`, `<path_to>/create_phonon_struct.py`, `<path_to_phonon_data.npz>`, `<pert_scale>`) so they can be copied and filled in manually. The generator fills DEFECT, PHONON_PATH and PERT always, and BINARY when `--vasp-binary` is passed; CREATE_STRUCT and GET_N_MODES are always filled with paths into the scripts/cluster directory the generator itself lives in.
