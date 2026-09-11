from argparse import ArgumentParser as Parser
from beyblade.parsers import parse_phonon_npz, save_phonon_npz

if __name__ == "__main__":
  parser = Parser("Determine symmetry of phonon modes.")
  parser.add_argument("phonon_path", metavar="phonon_path", help="Path to phonon data.")

  args = parser.parse_args()

  spectrum = parse_phonon_npz(args.phonon_path)
  spectrum.analyze_c3v_symmetry()
  filtered = spectrum.filter_sym_pairs()
  save_phonon_npz(filtered)
