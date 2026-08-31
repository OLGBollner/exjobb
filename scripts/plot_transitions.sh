#!/bin/bash
# Plot rates for NV and ClV:
#  - For NV  (multiple cell sizes): combine all cell sizes for each rate suffix.
#  - For ClV (single cell size):       combine all rate suffixes into one plot.

BASE_FLAGS="--log --plot --save"
MATERIALS=("NV" "ClV")

for material in "${MATERIALS[@]}"; do
    echo "=========================================="
    echo " Processing material: $material"
    echo "=========================================="

    # ------------------------------------------------------------------
    # 1. Collect all non‑directional rate files for this material
    # ------------------------------------------------------------------
    all_files=()
    while IFS= read -r -d '' file; do
        all_files+=("$file")
    done < <(find rates -maxdepth 1 -name "${material}_*_rates_*.npz" ! -name "*directional*" -print0 2>/dev/null)

    if [ ${#all_files[@]} -eq 0 ]; then
        echo "No rate files found for $material, skipping."
        continue
    fi

    # ------------------------------------------------------------------
    # 2. Extract unique cell sizes from filenames
    #    Filename pattern: ${material}_${cellsize}_rates_${suffix}.npz
    # ------------------------------------------------------------------
    declare -A cellsize_seen
    declare -A suffix_groups   # will be used only if multiple cell sizes

    for file in "${all_files[@]}"; do
        base=$(basename "$file")
        # Remove material prefix
        rest="${base#${material}_}"
        # Extract cell size (everything up to the first underscore after material_)
        cellsize="${rest%%_*}"
        cellsize_seen["$cellsize"]=1
    done

    # ------------------------------------------------------------------
    # 3. Choose grouping strategy
    # ------------------------------------------------------------------
    if [ ${#cellsize_seen[@]} -gt 1 ]; then
        # ------------------------------------------------------------------
        # Case A: multiple cell sizes → group by suffix
        # ------------------------------------------------------------------
        echo "Detected multiple cell sizes: ${!cellsize_seen[*]}"
        echo "Grouping files by rate suffix."

        # Build suffix groups (as in your original script)
        for file in "${all_files[@]}"; do
            base=$(basename "$file")
            suffix="${base#*_rates_}"
            suffix="${suffix%.npz}"
            suffix_groups["$suffix"]+="$file "
        done

        # Now iterate over the three analysis flags
        for flag in "--first-order" "--second-order" "--second-phonon"; do
            echo "---------- Flag: $flag ----------"
            for suffix in "${!suffix_groups[@]}"; do
                read -ra files <<< "${suffix_groups[$suffix]}"
                echo "Plotting '$suffix' with ${#files[@]} cell sizes: ${files[*]}"
                python process_transitions.py "${files[@]}" $flag $BASE_FLAGS
            done
            echo
        done

        unset suffix_groups

    else
        # ------------------------------------------------------------------
        # Case B: single cell size → combine all suffixes in one plot
        # ------------------------------------------------------------------
        echo "Detected only one cell size: ${!cellsize_seen[*]}"
        echo "Combining all rate suffixes into a single plot per flag."

        for flag in "--first-order" "--second-order" "--second-phonon"; do
            echo "---------- Flag: $flag ----------"
            echo "Plotting ${#all_files[@]} files: ${all_files[*]}"
            python process_transitions.py "${all_files[@]}" $flag $BASE_FLAGS
            echo
        done
    fi

    unset cellsize_seen
    echo
done
