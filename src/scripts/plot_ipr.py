import numpy as np
import matplotlib.pyplot as plt
from scipy import constants as Cn
from pathlib import Path
from argparse import ArgumentParser as Parser
from utils import smear_data, calc_ipr

if __name__ == "__main__":
  parser = Parser("Plots IPR.")
  parser.add_argument("sim_folder", metavar="sim_folder", help="Folder containing simulation results")
  parser.add_argument("-f", "--format", help="File format of plot")
  parser.add_argument("-o", "--output", help="Specify the output filename for the plot or data")
  parser.add_argument("-s", "--smooth", action="store_true", help="Plot smooth IPR")

  args = parser.parse_args()
  sim_folder = Path(args.sim_folder)

  phonons = np.load(sim_folder/ "data/phonon_data.npz")
  ipr = calc_ipr(phonons)

  freqs = [f for f in phonons["freqs"] if f > 0]

  plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 10
  })

  fig, ax = plt.subplots(figsize=(6, 5))

  if args.smooth:
    smooth_x, smooth_y = smear_data(freqs, ipr, 1, 7.5)
    ax.plot(smooth_x, smooth_y, '-', markersize=4, alpha=0.6, label='IPR', color='tab:blue')
  else:
    ax.plot(freqs, ipr, '-', markersize=4, alpha=0.6, label='IPR', color='tab:blue')
  ax.set_xlabel(r'Vibration frequency $\omega$ [meV]')
  ax.set_ylabel('IPR (Localization)')
  ax.set_title('IPR vs Frequency')
  ax.grid(True, which='both', linestyle='--', alpha=0.5)
  ax.set_xlim(0, 200)
  ax.set_ylim(0)

  ax.legend(loc='upper right')

  plt.tight_layout()

  filename = ""
  if args.output:
    filename = args.output
  else:
    filename += f"ipr_{sim_folder.name}"

  filename += args.format if args.format else ".png"
  print("Saving figure in: ", "figures/"+filename)
  plt.savefig("figures/"+filename)
