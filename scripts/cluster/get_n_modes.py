import numpy as np
import sys

if ".npz" in sys.argv[1]:
  file = sys.argv[1]
  phonon_data = np.load(file)
  if "idx" in phonon_data.keys():
    idx_str = " ".join([str(i+1) for i in phonon_data["idx"]])
    print(idx_str)
  else:
    modes = range(1, phonon_data["freqs"].shape[0] + 1)
    idx_str = " ".join([str(i) for i in modes])
    print(idx_str)
