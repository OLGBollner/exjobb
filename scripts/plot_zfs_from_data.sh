#!/bin/bash
# Script to plot ZFS from already calculated coefficients using zfs_analysis.py --plot
# Run from project root (where zfs_analysis.py is located).

BASES=("ClV_128")
ORDER=2

for base in "${BASES[@]}"; do
    # Loop over one-phonon coefficient files (contain "zfs_coefficients" but NOT "2d")
    for one_phonon in derivatives/${base}_*zfs_coefficients*.npz; do
        [ -e "$one_phonon" ] || continue

        # Skip if this is already a two-phonon file (contains '2d')
        if [[ "$one_phonon" == *2d* ]]; then
            continue
        fi

        # Construct the corresponding two-phonon filename
        two_phonon="${one_phonon/zfs_coefficients/zfs2d_coefficients}"

        # Check if the two-phonon file exists; skip and warn if not
        if [ ! -f "$two_phonon" ]; then
            echo "Warning: missing two-phonon file for $one_phonon ($two_phonon not found), skipping"
            continue
        fi

        echo "Plotting pair:"
        if [[ $ORDER -eq 1 ]]; then
            echo "  One-phonon : $one_phonon"
            python zfs_analysis.py \
                --data_files "$one_phonon" --plot
        else if [[ $ORDER -eq 2 ]]; then
            echo "  Two-phonon : $two_phonon"
            python zfs_analysis.py \
                --data_files "$two_phonon" --plot
        fi
        fi
    done
done
