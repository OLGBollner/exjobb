"""Determine symmetry of phonon modes and save the filtered spectrum.

0-based idx convention
----------------------
Mode indices are 0-based positions in the *full* spectrum. The legacy npz
`idx` arrays are 0-based, while VASP folder numbering (from `phonons.yaml`)
and the `get_n_modes.py` printout (`i+1`) are 1-based. Downstream code that
joins on `idx` must not mix the two conventions.

Enforced here so the convention cannot silently regress if
`beyblade.symmetry.filter_degenerate_partners` changes:

1. The filter must return an `original_indices` array whose values
   are exactly the 0-based positions of the kept modes in the full spectrum.
2. The filtered spectrum's mode count must match `len(original_indices)`.
"""

from argparse import ArgumentParser as Parser

from beyblade.parsers import parse_phonon_npz, save_phonon_npz
from beyblade.symmetry import (
    classify_and_pair,
    filter_degenerate_partners,
    symmetrize_degenerate_groups,
)


if __name__ == "__main__":
  parser = Parser("Determine symmetry of phonon modes.")
  parser.add_argument("phonon_path", metavar="phonon_path", help="Path to phonon data.")
  parser.add_argument("out_path", metavar="out_path", nargs="?",
                      help="Where to save the filtered spectrum (default: overwrite phonon_path).")

  args = parser.parse_args()

  spectrum = parse_phonon_npz(args.phonon_path)
  classify_and_pair(spectrum)
  symmetrize_degenerate_groups(spectrum)
  filtered = filter_degenerate_partners(spectrum)
  save_phonon_npz(filtered, args.out_path or args.phonon_path)
  check = spectrum.sym_check
  note = "" if check is None else f"; sym_check: {check.summary if hasattr(check, 'summary') else check}"
  print(f"saved filtered spectrum ({filtered.n_modes} of {spectrum.n_modes} modes) "
        f"to {args.out_path or args.phonon_path}{note}")
