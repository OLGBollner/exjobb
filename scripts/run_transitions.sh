#!/bin/bash
# Script to run process_transitions.py for every one-phonon / two-phonon pair.
# Assumes you run it from the project root (where process_transitions.py is located).

BASES=("NV_64" "NV_512" "ClV_128")

# Fixed arguments for the transition processing
T_END=500
T_STEP=2

for base in "${BASES[@]}"; do
    # Find all one-phonon coefficient files for this base.
    # The pattern matches any filename that contains "zfs_coefficients" but NOT "2d".
    for one_phonon in derivatives/${base}_*zfs_coefficients*.npz; do
        [ -e "$one_phonon" ] || continue

        # Construct the two-phonon filename by replacing "zfs_coefficients" with "zfs2d_coefficients"
        two_phonon="${one_phonon/zfs_coefficients/zfs2d_coefficients}"

        # Check if the two-phonon file exists; skip and warn if not
        if [ ! -f "$two_phonon" ]; then
            echo "Warning: missing two-phonon file for $one_phonon ($two_phonon not found), skipping"
            continue
        fi

        echo "Processing pair:"
        echo "  One-phonon : $one_phonon"
        echo "  Two-phonon : $two_phonon"
        python process_transitions.py "$one_phonon" \
            --two-phonon "$two_phonon" \
            --calc \
            --t-end "$T_END" \
            --t-step "$T_STEP"
    done
done
