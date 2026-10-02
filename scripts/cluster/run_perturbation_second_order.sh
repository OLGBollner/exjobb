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

ml PDC/26.03
ml miniconda3
source activate /cfs/klemming/projects/supr/adaq/obollner/conda-dirs/envs/sim-env || {
  echo "FATAL: failed to activate conda env -- aborting before any VASP run" >&2
  exit 1
}

# ---- config (override via environment, e.g. DEFECT=NV_512 sbatch ...) ----
DEFECT=${DEFECT:-<defect>}
binary=${BINARY:-<vasp_binary>}
create_struct=${CREATE_STRUCT:-<path_to>/create_combined_phonon_struct.py}
get_n_modes=${GET_N_MODES:-<path_to>/get_n_modes.py}
PHONON_PATH=${PHONON_PATH:-<path_to_phonon_data.npz>}
PERT=${PERT:-<pert_scale>}
# --------------------------------------------------------------------------

INPUT_DIR="input"
HEAD_DIR=$(pwd)
PAIR_MODE=${PAIR_MODE:-<pair_mode>}

# ---- preflight: verify every dependency BEFORE burning core hours ----
fail() { echo "FATAL (task ${SLURM_ARRAY_TASK_ID:-?}): $*" >&2; exit 1; }
[ -d "$INPUT_DIR" ] || fail "input dir '$INPUT_DIR' missing (run this from the all_bands folder)"
case $DEFECT in *'<defect>'*) fail "DEFECT still has placeholder: '$DEFECT'" ;; esac
case $binary in *'<'*|*'/path/to/'*) fail "BINARY still has placeholder: '$binary'" ;; esac
case $create_struct in *'<path_to>'*) fail "CREATE_STRUCT still has placeholder: '$create_struct'" ;; esac
case $get_n_modes in *'<path_to>'*) fail "GET_N_MODES still has placeholder: '$get_n_modes'" ;; esac
case $PHONON_PATH in *'<path_to_phonon_data.npz>'*) fail "PHONON_PATH still has placeholder: '$PHONON_PATH'" ;; esac
case $PERT in *'<pert_scale>'*) fail "PERT still has placeholder: '$PERT'" ;; esac
case $PAIR_MODE in *'<pair_mode>'*|'') fail "PAIR_MODE unset or still placeholder: '$PAIR_MODE'" ;; esac
[ -x "$binary" ] || fail "VASP binary not found/executable: $binary"
[ -f "$create_struct" ] || fail "create_struct script missing: $create_struct"
[ -f "$get_n_modes" ] || fail "get_n_modes script missing: $get_n_modes"
[ -f "$PHONON_PATH" ] || fail "phonon data file missing: $PHONON_PATH"
python -c "import numpy, pymatgen.core" >/dev/null 2>&1 \
  || fail "python env broken (numpy/pymatgen import failed) -- conda env did not activate?"
python -c "import beyblade" >/dev/null 2>&1 \
  || fail "python cannot import beyblade -- install/check sim-env"
MODES=$(python "$get_n_modes" "$PHONON_PATH") || fail "get_n_modes exited nonzero on $PHONON_PATH"
[ -n "$MODES" ] || fail "get_n_modes returned no modes for $PHONON_PATH"
echo "Preflight OK: $(echo $MODES | wc -w) modes, pair_mode=$PAIR_MODE, binary=$binary"
# ------------------------------------------------------------------------

ID=$SLURM_ARRAY_TASK_ID
NJOBS=$SLURM_ARRAY_TASK_COUNT
ZFS_REGEX_PATTERN='Spin-spin contribution to zero-field splitting tensor \(MHz\)\s*-+\s*D_xx\s+D_yy\s+D_zz\s+D_xy\s+D_xz\s+D_yz\s*-+\s*([\s\d\.\-]+?)(?=\s*-{3,})'

echo $ID
echo $NJOBS

set -eu

mkdir -p runs

PAIR_ID=0
for i in $MODES; do
  for j in $MODES; do
    if [ "$PAIR_MODE" = "diag" ] && [ "$i" != "$j" ]; then
      continue
    fi
    if [ "$PAIR_MODE" = "all" ] && [ $i -gt $j ]; then
      continue
    fi
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
      echo "$ID: starting job for pair $i, $j"
      MSG=$(python $create_struct -o POSCAR POSCAR $PHONON_PATH $i $j $PERT)
      if echo "$MSG" | grep -q "Success"; then
        echo "Successfully created perturbed structure"
      else
        echo "Failed to create structure."
        echo "Shutting down job $ID"
        echo "$MSG"
        exit 1
      fi
      srun $binary
      echo "$ID: Done! $i, $j"
      echo "$ID: removing excess WAVECAR"
      rm -f WAVECAR
      cd "$HEAD_DIR"
    fi
    PAIR_ID=$((PAIR_ID + 1))
  done
done
