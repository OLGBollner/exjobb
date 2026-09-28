#!/bin/sh
#SBATCH -A naiss2025-3-15
#SBATCH -J defect_2ph
#SBATCH -N 2
#SBATCH --ntasks-per-node=128
#SBATCH -t 2:30:00
#SBATCH -p main
#SBATCH -a 0-9

# Second-order ZFS perturbation runs: perturb POSCAR along mode pairs.
# Copy this script into a defect's second_order/<pert>/all_bands/ folder and set
# the config variables below (or export them before sbatch). Personal paths
# belong in the config block only, not in the loop logic.

ml PDC/24.11
ml miniconda3
source activate /cfs/klemming/projects/supr/adaq/obollner/conda-dirs/envs/sim-env

# ---- config (override via environment, e.g. DEFECT=NV_512 sbatch ...) ----
DEFECT=${DEFECT:-ClV_128}
binary=${BINARY:-/cfs/klemming/projects/supr/adaq/oscarlb/to_bollner/vasp_binary/vasp_std}
create_struct=${CREATE_STRUCT:-/cfs/klemming/projects/supr/adaq/obollner/exjobb/scripts/create_combined_phonon_struct.py}
get_n_modes=${GET_N_MODES:-/cfs/klemming/projects/supr/adaq/obollner/exjobb/scripts/get_n_modes.py}
PHONON_PATH=${PHONON_PATH:-/cfs/klemming/projects/supr/adaq/obollner/${DEFECT}/data/phonon_data_sym_n254.npz}
PERT=${PERT:-0.025}
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
  j=$i
  if [ $((PAIR_ID % NJOBS)) -eq $ID ]; then
    echo "$ID: processing pair $PAIR_ID (modes $i, $j)"
    TARGET_DIR="runs/"$i"_"$j
    if [ -f "$TARGET_DIR"/OUTCAR ] && grep -qzP "$ZFS_REGEX_PATTERN" "$TARGET_DIR"/OUTCAR; then
      echo "$ID: skipping $i, $j"
      PAIR_ID=$((PAIR_ID + 1))
      continue
    fi
    mkdir -p $TARGET_DIR
    cp "$INPUT_DIR"/* "$TARGET_DIR"/
    cd "$TARGET_DIR"
    echo "$ID: starting job for mode $i, $j"
    MSG=$(python $create_struct -o POSCAR POSCAR $PHONON_PATH $i $j $PERT)
    if echo "$MSG" | grep -q "Error:"; then
      echo "Failed to create structure."
      echo "Shutting down job $ID"
      echo "$MSG"
      exit 1
    elif echo "$MSG" | grep -q "Success"; then
      echo "Successfully created perturbed structure"
    fi
    echo "$MSG"
    srun $binary
    echo "$ID: Done! $i, $j"
    echo "$ID: removing excess WAVECAR"
    rm WAVECAR
    cd "$HEAD_DIR"
  fi
  PAIR_ID=$((PAIR_ID + 1))
done
