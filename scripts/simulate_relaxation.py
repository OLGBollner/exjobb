import sys
from beyblade.relaxation_dynamics import RelaxationDynamics
import matplotlib.pyplot as plt
import numpy as np

file_path = sys.argv[1]

initial_population = np.array([0.0, 1.0, 0.0])
target_temperature = 50

simulator = RelaxationDynamics(init_state=initial_population, rates_data=file_path)

T1_eigenval, eigenvalues, eigenvectors = simulator.get_T1_eigenval(T=target_temperature)
steady_state = simulator.get_steady_state(T=target_temperature)[simulator.population.argmax()]

time_points = np.linspace(0, 10 * T1_eigenval, 10000)

population = simulator.simulate_relaxation(time_points, T=target_temperature)

polarized_pop = population[:, simulator.population.argmax()]
T1_fit = simulator.fit_T1_exp(polarized_pop, time_points, steady_state, T1_eigenval)


print("T1 population: ", T1_fit)
print("T1 eigenvalue: ", T1_eigenval)

plt.figure(figsize=(10, 6))
plt.plot(time_points, population[:, 0], label="State 1", linewidth=2)
plt.plot(time_points, population[:, 1], label="State 0", linewidth=2)
plt.plot(time_points, population[:, 2], label="State -1", linewidth=2)
plt.plot(time_points, polarized_pop, label="Exp Fit", linewidth=2, linestyle="--")
plt.plot(time_points, np.sum(population, axis=1), label="Total Population", linestyle="--", color="black")
plt.axvline(x=T1_fit, label=r"$T_1$", linestyle="--", color="gray")
plt.yscale("log")
plt.xscale("log")

plt.title(f"Population Relaxation Dynamics at T = {target_temperature} K")
plt.xlabel("Time (s)")
plt.ylabel("Population Fraction")
plt.grid(True, linestyle=":", alpha=0.6)
plt.xlim(0, time_points[-1])
plt.legend()
plt.show()
