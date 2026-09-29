import sys

from beyblade.parsers import parse_phonon_data, save_phonon_npz

spectrum = parse_phonon_data(sys.argv[1])
save_phonon_npz(spectrum)
