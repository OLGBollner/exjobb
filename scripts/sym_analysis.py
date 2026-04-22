from argparse import ArgumentParser as Parser
from beyblade.phonon_manager import PhononManager

if __name__ == "__main__":
  parser = Parser("Determine symmetry of phonon modes.")
  parser.add_argument("phonon_path", metavar="phonon_path", help="Path to phonon data.")

  args = parser.parse_args()

  phonon_mgr = PhononManager(args.phonon_path)
  phonon_mgr.analyze_c3v_symmetry()
  #phonon_mgr.filter_sym_pairs(save=True)

