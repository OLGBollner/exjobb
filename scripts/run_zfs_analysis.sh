#!/bin/bash
# Script to run zfs_analysis.py for all raw ZFS 1d/2d file pairs across bases.
# Works with any middle descriptor (e.g., all_bands, defect_band_approx).
# Run from project root (where zfs_analysis.py is located).

BASES=("NV_512" "NV_64" "ClV_128")
ORDER=3

for base in "${BASES[@]}"; do
    phonon="../${base}/phonon_data.npz"

    # Check phonon data exists; skip base if missing
    if [ ! -f "$phonon" ]; then
        echo "Skipping ${base}: phonon data not found ($phonon)"
        continue
    fi

    # Find all 1d raw ZFS files for this base
    for raw_1d in ../"${base}"/"${base}"_raw_zfs_data_*_1d.npz; do
        [ -e "$raw_1d" ] || continue

        # Extract the middle descriptor between "raw_zfs_data_" and "_1d.npz"
        # e.g., from "NV_64_raw_zfs_data_all_bands_1d.npz" -> "all_bands"
        filename=$(basename "$raw_1d")
        middle=${filename#*_raw_zfs_data_}   # remove prefix up to "_raw_zfs_data_"
        middle=${middle%_1d.npz}            # remove "_1d.npz" suffix

        # Construct the corresponding 2d filename
        raw_2d="../${base}/${base}_raw_zfs_data_${middle}_2d.npz"

        if [ ! -f "$raw_2d" ]; then
            echo "Warning: missing 2d file for ${filename} (expected $raw_2d), skipping"
            continue
        fi

        echo "Processing ${base} with descriptor '${middle}':"
        echo "  1d: $raw_1d"
        echo "  2d: $raw_2d"
        python zfs_analysis.py \
            --raw_zfs_file "$raw_1d" "$raw_2d" \
            --ph "$phonon" \
            --calc \
            --order="$ORDER"
            #--plot 
            #--debug
    done
done
