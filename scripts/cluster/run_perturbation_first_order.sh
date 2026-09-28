#!/bin/sh
#SBATCH -A naiss2025-3-15
#SBATCH -J defect_1ph
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -t 2:30:00
#SBATCH -p main
#SBATCH -a 0-9

# First-order ZFS perturbation runs: perturb POSCAR along single phonon modes.
# Copy this script into a defect's first_order/<pert>/all_bands/ folder and set
# the config variables below (or export them before sbatch). Personal paths
# belong in the config block only, not in the loop logic.

ml PDC/24.11
ml miniconda3
source activate /cfs/klemming/projects/supr/adaq/obollner/conda-dirs/envs/sim-env

# ---- config (override via environment, e.g. DEFECT=NV_512 sbatch ...) ----
DEFECT=${DEFECT:-<defect>}
binary=${BINARY:-<vasp_binary>}
create_struct=${CREATE_STRUCT:-<path_to>/create_phonon_struct.py}
get_n_modes=${GET_N_MODES:-<path_to>/get_n_modes.py}
PHONON_PATH=${PHONON_PATH:-<path_to_phonon_data.npz>}
PERT=${PERT:-<pert_scale>}
# --------------------------------------------------------------------------

INPUT_DIR="input"
HEAD_DIR=$(pwd)

MODES=$(python $get_n_modes $PHONON_PATH)

ID=$SLURM_ARRAY_TASK_ID
NJOBS=$SLURM_ARRAY_TASK_COUNT
ZFS_REGEX_PATTERN='Spin-spin contribution to zero-field splitting tensor \(MHz\)\s*-+\s*D_xx\s+D_yy\s+D_zz\s+D_xy\s+D_xz\s+D_yz\s*-+\s*([\s\d\.\-]+?)(?=\s*-{3,})'

echo $ID
echo $NJOBS

mkdir runs

PAIR_ID=0
for i in $MODES; do
  if [ $((PAIR_ID % NJOBS)) -eq $ID ]; then
    echo "$ID: processing mode $i"
    TARGET_DIR="runs/"$i
    if [ -f "$TARGET_DIR"/OUTCAR ] && grep -qzP "$ZFS_REGEX_PATTERN" "$TARGET_DIR"/OUTCAR; then
      echo "$ID: skipping $i"
      PAIR_ID=$((PAIR_ID + 1))
      continue
    fi
    mkdir -p $TARGET_DIR
    cp "$INPUT_DIR"/* "$TARGET_DIR"/
    cd "$TARGET_DIR"
    echo "$ID: starting job for mode $i"
    MSG=$(python $create_struct -o POSCAR POSCAR $PHONON_PATH $i $PERT)
    if echo "$MSG" | grep -q "Success"; then
      echo "Successfully created perturbed structure"
    else
      echo "Failed to create structure."
      echo "Shutting down job $ID"
      echo "$MSG"
      exit 1
    fi
    srun $binary
    echo "$ID: Done! $i"
    echo "$ID: removing excess WAVECAR"
    rm WAVECAR
    cd "$HEAD_DIR"
  fi
  PAIR_ID=$((PAIR_ID + 1))
done
