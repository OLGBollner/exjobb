import sys
from beyblade.relaxation_dynamics import RelaxationDynamics
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
  "axes.titlesize": 16,
  "axes.labelsize": 16,
  "xtick.labelsize": 12,
  "ytick.labelsize": 12,
  "legend.fontsize": 14
  })

file_path = sys.argv[1:]
show_plot = False

initial_population = np.array([0.0, 1.0, 0.0])
plt.figure(figsize=(10, 6))

if file_path[0] == "--plot":
  show_plot = True
  file_path = file_path[1:]

for file in file_path:

  print("processing file: ", file)
  simulator = RelaxationDynamics(init_state=initial_population, rates_data=file)
  T1_times = simulator.compute_T1_range()
  T1_room_temp = simulator.get_T1_fit(T=300)
  T1_low_temp = simulator.get_T1_fit(T=1e-3)
  print("T1 in room temp: ", T1_room_temp)
  print("Low temp limit: ", T1_low_temp)

  plt.plot(simulator.temperatures, T1_times, linewidth=2, label=f"{simulator.defect} {simulator.cell_size} {simulator.sim_type}")

#plt.title(r"$T_1$ relaxation time")
plt.xlabel("Temperature (K)")
plt.ylabel(r"$T_1$ (s)")
plt.yscale("log")
plt.tick_params(axis='y', which='both', direction='in')

plt.grid(True, linestyle=":", alpha=0.6)
plt.legend()
plt.tight_layout()
if show_plot:
  plt.show()
else:
  save_name = f"figures/{simulator.defect}_T1_time"
  print("Saving figure in: ", save_name+".png")
  plt.savefig(save_name)
