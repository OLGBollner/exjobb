import sys

import numpy as np

if "--sym" in sys.argv:
    sys.argv.remove("--sym")
    with_sym = True
else:
    with_sym = False

if ".npz" in sys.argv[1]:
    file = sys.argv[1]
    phonon_data = np.load(file, allow_pickle=True)
    has_idx = "idx" in phonon_data.keys()
    if has_idx:
        idx = np.array(phonon_data["idx"])
    else:
        idx = np.arange(1, phonon_data["freqs"].shape[0] + 1)

    if with_sym:
        from beyblade.parsers import parse_phonon_npz
        from beyblade.symmetry import classify_and_pair

        spectrum = parse_phonon_npz(file)
        if spectrum.symmetries is None:
            classify_and_pair(spectrum)
        labels = list(spectrum.symmetries)
        if len(labels) != len(idx):
            sys.exit(
                f"error: {len(labels)} symmetry labels for {len(idx)} modes; "
                "re-run sym_analysis.py to regenerate this npz"
            )
        print(" ".join(["mode", "freq[cm-1]", "sym"]))
        for pos, (i, label) in enumerate(zip(idx, labels)):
            freq = phonon_data["freqs"][pos] if has_idx else phonon_data["freqs"][i - 1]
            mode_no = i + 1 if has_idx else i
            print(f"{mode_no} {freq:.2f} {label}")
    else:
        print(" ".join([str(i + 1) if has_idx else str(i) for i in idx]))
else:
    print("usage: get_n_modes.py <phonon.npz> [--sym]")
