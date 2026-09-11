import sys
from beyblade.parsers import parse_phonon_npz, save_phonon_npz

spectrum = parse_phonon_npz(sys.argv[1])
save_phonon_npz(spectrum)
