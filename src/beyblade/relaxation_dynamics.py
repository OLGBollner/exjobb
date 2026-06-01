import numpy as np
from scipy.integrate import odeint
from scipy.optimize import curve_fit

class RelaxationDynamics:
  def __init__(self, init_state, rates_data):
    self.defect: str =    None
    self.cell_size: int = None
    self.sim_type: str =  None

    self.population: np.ndarray = np.array(init_state)
    self.rate_matrices, self.temperatures = self.read_data(rates_data)


  def read_data(self, data_file):
    data = np.load(data_file, allow_pickle=True)
    self.defect = data.get("defect")
    self.cell_size = data.get("cell_size")
    self.sim_type = "(spin corrected)" if data.get("sub_folder") == "defect_band_approx" else ""

    rate_matrices = []

    print("Creating rate matrices...")
    for i, t in enumerate(data.get("temperatures")):
      A = [
        [ -data["1_to_0"][i] - data["1_to_-1"][i],          data["0_to_1"][i],                       data["-1_to_1"][i]           ],
        [           data["1_to_0"][i],            -data["0_to_1"][i] - data["0_to_-1"][i],           data["-1_to_0"][i]           ],
        [           data["1_to_-1"][i],                     data["0_to_-1"][i],           -data["-1_to_0"][i] - data["-1_to_1"][i]],
      ]

      rate_matrices.append(A)

    temperatures = np.array(data.get("temperatures", []))

    return np.array(rate_matrices), temperatures

  def _population_derivative(self, P, t, A):
    return np.dot(A, P)

  def _get_temp_index(self, T):
    if T:
      temp_index = np.argmin(np.abs(self.temperatures - T))
    else:
      raise ValueError(f"Incorrect temperature format: {T}")
    return temp_index

  def fit_T1_exp(self, population, time_points, steady_state, T1_eigenval):
    def exp_func(t, T1, c):
      exp = np.exp(-t / T1) + c
      return exp

    initial_guess = [T1_eigenval, steady_state]

    popt, _, info_dict, _, _ = curve_fit(exp_func, time_points, population, p0=initial_guess, full_output=True)
    # print("steady_state: ", steady_state)
    # print("c: ", popt[1])
    error = np.mean(info_dict["fvec"])
    if error > 1e-9:
      print("Error: ", error)

    extracted_T1 = popt[0]

    return extracted_T1

  def simulate_relaxation(self, time_points, T=None, temp_index=None):
    if temp_index is None:
      temp_index = self._get_temp_index(T)
    A = self.rate_matrices[temp_index]

    population_evolution = odeint(
        func=self._population_derivative,
        y0=self.population,
        t=time_points,
        args=(A,)
    )

    return population_evolution

  def get_T1_eigenval(self, T=None, temp_index=None):
    if temp_index is None:
      temp_index = self._get_temp_index(T)
    A = self.rate_matrices[temp_index]

    eigenvalues, eigenvectors = np.linalg.eig(A)

    non_zero_eigenvalues = eigenvalues[np.abs(eigenvalues) > 1e-9]
    
    if len(non_zero_eigenvalues) > 0:
        relaxation_time = -1.0 / np.max(non_zero_eigenvalues)
    else:
        relaxation_time = np.inf

    return relaxation_time, eigenvalues, eigenvectors

  def get_steady_state(self, T=None, temp_index=None):
    if temp_index is None:
      temp_index = self._get_temp_index(T)
    A = self.rate_matrices[temp_index]

    eigenvalues, eigenvectors = np.linalg.eig(A)

    steady_state = np.concatenate(eigenvectors[:, np.abs(eigenvalues) < 1e-9])

    return steady_state

  def compute_T1_range(self):
    T1_range = []

    for i in range(len(self.temperatures)):
      T1_fit = self.get_T1_fit(temp_index=i)

      T1_range.append(T1_fit)

    return T1_range

  def get_T1_fit(self, T=None, temp_index=None):
    T1_eigenval, _, _ = self.get_T1_eigenval(T=T, temp_index=temp_index)
    steady_state = self.get_steady_state(T=T, temp_index=temp_index)[self.population.argmax()]

    if T1_eigenval > 1e9:
      return T1_eigenval

    time_points = np.linspace(0, 10 * T1_eigenval, 10000)

    population = self.simulate_relaxation(time_points, T=T, temp_index=temp_index)[:, self.population.argmax()]

    T1_fit = self.fit_T1_exp(population, time_points, steady_state, T1_eigenval)

    return T1_fit
