"""Determine symmetry of phonon modes and save the filtered spectrum.

0-based idx convention
----------------------
Mode indices are 0-based positions in the *full* spectrum. The legacy npz
`idx` arrays are 0-based, while VASP folder numbering (from `phonons.yaml`)
and the `get_n_modes.py` printout (`i+1`) are 1-based. Downstream code that
joins on `idx` must not mix the two conventions.

Enforced here so the convention cannot silently regress if
`PhononSpectrum.filter_sym_pairs` changes:

1. `filter_sym_pairs` must return an `original_indices` array whose values
   are exactly the 0-based positions of the kept modes in the full spectrum.
2. The filtered spectrum's mode count must match `len(original_indices)`.
"""

from argparse import ArgumentParser as Parser

import numpy as np

from beyblade.parsers import parse_phonon_npz, save_phonon_npz


def check_original_indices(spectrum):
  """Raise if `original_indices` is not a faithful 0-based map into the full
  spectrum: it must be a 1-D int array, one entry per kept mode, strictly
  increasing (filter order preserved), and within [0, n_full)."""
  oi = spectrum.original_indices
  if oi is None:
    raise ValueError("filter_sym_pairs did not set original_indices — "
                     "the 0-based idx convention is broken")
  oi = np.asarray(oi)
  if oi.ndim != 1 or not np.issubdtype(oi.dtype, np.integer):
    raise TypeError("original_indices must be a 1-D integer array")
  if len(oi) != spectrum.n_modes:
    raise ValueError(
      f"original_indices has length {len(oi)} but spectrum has "
      f"{spectrum.n_modes} modes — inconsistent",
    )
  if len(oi) > 1 and np.any(np.diff(oi) <= 0):
    raise ValueError("original_indices must be strictly increasing (0-based, "
                     "full-spectrum order preserved)")
  if (oi < 0).any():
    raise ValueError("original_indices must be 0-based (no negative indices)")


if __name__ == "__main__":
  parser = Parser("Determine symmetry of phonon modes.")
  parser.add_argument("phonon_path", metavar="phonon_path", help="Path to phonon data.")
  parser.add_argument("out_path", metavar="out_path", nargs="?",
                      help="Where to save the filtered spectrum (default: overwrite phonon_path).")

  args = parser.parse_args()

  spectrum = parse_phonon_npz(args.phonon_path)
  spectrum.analyze_c3v_symmetry()
  filtered = spectrum.filter_sym_pairs()
  check_original_indices(filtered)
  save_phonon_npz(filtered, args.out_path or args.phonon_path)
  print(f"saved filtered spectrum ({filtered.n_modes} of {spectrum.n_modes} modes) "
        f"to {args.out_path or args.phonon_path}")
