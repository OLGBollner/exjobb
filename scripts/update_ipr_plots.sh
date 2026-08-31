#!/bin/bash
# Script to update all IPR plots by running plot_ipr.py for every coefficients*.npz file
# Assumes you run it from the project root (where plot_ipr.py lives).

BASES=("ClV_128") # "NV_64" "NV_512"

for base in "${BASES[@]}"; do
    # Look for all matching coefficient files in the derivatives/ folder
    for datafile in derivatives/${base}_*coefficients*.npz; do
        # Skip if no files matched the pattern
        [ -e "$datafile" ] || continue

        echo "Processing: $datafile"
        python plot_ipr.py \
            --data "$datafile" \
            --phonon_data "../${base}/phonon_data.npz"
    done
done
