import numpy as np
from scipy import constants as Cn
from pathlib import Path

from beyblade.utils import MathUtils
from beyblade.constants import CONSTANTS

# (sign_l, sign_lp) for energy conservation: omega_spin = s*omega_l + s'*omega_lp
PHONON_PROCESSES = {
    "abs_em":  (+1, -1),
    "em_abs":  (-1, +1),
    "abs_abs": (+1, +1),
    "em_em":   (-1, -1),
}

class Phonons:
  @staticmethod
  def bose_einstein(omega: np.ndarray, T: float) -> np.ndarray:
    if T <= 0:
      return np.zeros_like(omega)
    kB = Cn.k / Cn.e * 1000
    x = omega / (kB * T)
    n = np.zeros_like(omega)
    mask = x > 1e-5
    n[mask] = 1.0 / (np.exp(x[mask]) - 1.0)
    return n

  @staticmethod
  def _bose_factors_2ph(n_l, n_lp):
    return {
      "abs_em":  n_l       * (n_lp + 1),
      "em_abs":  (n_l + 1) * n_lp,
      "abs_abs": n_l       * n_lp,
      "em_em":   (n_l + 1) * (n_lp + 1),
    }

  @staticmethod
  def bose_einstein_2d(omega: np.ndarray, T: float) -> np.ndarray:
    n = Phonons.bose_einstein(omega, T)

    # outer products: shape (N, N)
    N_l  = n[None, :]
    N_lp = n[:, None]

    bose  = Phonons._bose_factors_2ph(N_l, N_lp)

    return bose


class TransitionRate:
  def __init__(self, data_file: str | None=None, two_phonon_data_file: Path | None=None):
    self.transition_rate: dict[str, dict[str, float]] = {
        "first_order": {}, "second_order": {}, "two_phonon": {}
    }
    self.ms_values = [1, 0, -1]
    self.Fz_elements = {1: 1.0, 0: -2.0, -1: 1.0}
    self.data = None
    self.data_2ph = None
    if data_file:
      self.load_data(data_file)
    if two_phonon_data_file is not None:
      self.load_data_2ph(two_phonon_data_file)

  def load_data_2ph(self, filename: Path) -> None:
    self.data_2ph = np.load(filename)

  def load_data(self, filename: str) -> None:
    self.data = np.load(filename)

  def get_spectral_density(self, V_key: str, res: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    if self.data is None:
      raise ValueError("Data not loaded. Please call load_data() first.")
    freqs = self.data["freqs"]
    values = self.data[V_key]
    return MathUtils.smear_data(freqs, values, res, sigma)

  def get_2ph_coupling(self, V_key: str) -> np.ndarray:
    if self.data_2ph is None:
      raise ValueError("Data not loaded. Please call load_data_2ph() first.")
    if V_key not in self.data_2ph:
      raise KeyError(f"Key '{V_key}' not found in data file.")
    return self.data_2ph[V_key]

  def _get_V2ph(self, m1, m2, V_0_pm_2ph, V_p_m_2ph, V_0_0_2ph):
    if m1 == m2:
      return (self.Fz_elements[m1]**2) * V_0_0_2ph
    diff = abs(m1 - m2)
    if diff == 1:
      return V_0_pm_2ph
    if diff == 2:
      return V_p_m_2ph
    return np.zeros_like(V_0_pm_2ph)

  def _compute_two_phonon_rates(self, T, omega_spin_meV, res=1, sigma=7.5):
    if self.data is None:
      raise ValueError("Data not loaded. Please call load_data() first.")

    freqs = self.data["freqs"]  # raw mode frequencies (N_modes,)

    omega_x, omega_y, V_0_pm_2ph = MathUtils.expand_data_2d(freqs, self.get_2ph_coupling("V_0_pm"), res, sigma)
    _, _, V_p_m_2ph  = MathUtils.expand_data_2d(freqs, self.get_2ph_coupling("V_p_m"), res, sigma)
    _, _, V_0_0_2ph  = MathUtils.expand_data_2d(freqs, self.get_2ph_coupling("V_0_0"), res, sigma)

    J_0_pm     = MathUtils.get_2d_spectral_density(freqs, V_0_pm_2ph, res, sigma)
    J_p_m      = MathUtils.get_2d_spectral_density(freqs, V_p_m_2ph, res, sigma)
    J_0_0_base = MathUtils.get_2d_spectral_density(freqs, V_0_0_2ph, res, sigma)

    def get_J_path(m1, m2) -> np.ndarray:
      if m1 == m2:
        return (self.Fz_elements[m1]**2) * J_0_0_base
      diff = abs(m1 - m2)
      if diff == 1: return J_0_pm
      if diff == 2: return J_p_m
      return np.zeros_like(omega_x)

    mask = (omega_x > 0.1) & (omega_y > 0.1)
    bose = Phonons.bose_einstein_2d(omega_x[0, :], T)

    f2ph     = (2 * np.pi / Cn.hbar) * (CONSTANTS["meV2J"] * res)**2

    print("Calculating integral...")
    for ms in self.ms_values:
      for ms_prime in self.ms_values:
        if ms == ms_prime:
          continue

        rate_key = f"{ms}_to_{ms_prime}"
        V2 = np.abs(self._get_V2ph(ms_prime, ms, V_0_pm_2ph, V_p_m_2ph, V_0_0_2ph)[mask])**2
        J = get_J_path(ms_prime, ms)
        J_m = J[mask]
        total = 0.0

        for proc, (s, sp) in PHONON_PROCESSES.items():
          delta       = MathUtils.broad_delta(omega_x, omega_y, sigma)
          total      += np.sum(V2 * bose[proc][mask] * delta[mask])

        self.transition_rate["two_phonon"][rate_key] = f2ph * total

    self.transition_rate["two_phonon"]["0_to_1"] = (
        self.transition_rate["two_phonon"].get("0_to_1",  0)
      + self.transition_rate["two_phonon"].get("0_to_-1", 0)
      + self.transition_rate["two_phonon"].get("-1_to_0", 0)
      + self.transition_rate["two_phonon"].get("1_to_0",  0)
    )
    self.transition_rate["two_phonon"]["1_to_-1"] = (
        self.transition_rate["two_phonon"].get("1_to_-1", 0)
      + self.transition_rate["two_phonon"].get("-1_to_1", 0)
    )

  def compute_transition_rates(self, T, res=0.01, sigma=7.5):
    if self.data is None:
      raise ValueError("Data not loaded. Please call load_data() first.")

    V_0_0 = self.data["V_0_0"]
    V_0_pm = self.data["V_0_pm"]
    V_p_m = self.data["V_p_m"]

    omega, V_0_0 = MathUtils.expand_data(self.data["freqs"], V_0_0, res, sigma)
    _, V_0_pm = MathUtils.expand_data(self.data["freqs"], V_0_pm, res, sigma)
    _, V_p_m = MathUtils.expand_data(self.data["freqs"], V_p_m, res, sigma)

    _, J_0_pm    = self.get_spectral_density("V_0_pm", res, sigma)
    _, J_p_m         = self.get_spectral_density("V_p_m",  res, sigma)
    _, J_0_0    = self.get_spectral_density("V_0_0",  res, sigma)
    print("Spectral functions:")
    print(f"J_0_pm: {J_0_pm.max()}")
    print(f"J_p_m: {J_p_m.max()}")
    print(f"J_0_0: {J_0_0.max()}")

    def get_J_path(m1, m2):
      if m1 == m2:
        return (self.Fz_elements[m1]**2) * J_0_0
      diff = abs(m1 - m2)
      if diff == 1: return J_0_pm
      if diff == 2: return J_p_m
      return np.zeros_like(omega)

    omega_spin_meV = 2.87 * CONSTANTS["GHz2meV"]
    n        = Phonons.bose_einstein(omega, T)
    mask     = omega > 0.1
    delta = MathUtils.broad_delta(omega, omega_spin_meV, sigma)

    f1_factor = (2 * np.pi / Cn.hbar) * CONSTANTS["meV2J"] * res
    print("f1: ", f1_factor)
    self.transition_rate["first_order"]["0_to_1"]  = f1_factor * np.sum(J_0_pm[mask] * (2 * n[mask] + 1) * delta[mask]) * 4
    self.transition_rate["first_order"]["1_to_-1"] = f1_factor * np.sum(J_p_m[mask] * (2 * n[mask] + 1) * delta[mask]) * 2

    f2_factor = (4 * np.pi / Cn.hbar) * res * CONSTANTS["meV2J"]
    print("f2: ", f2_factor)

    print("Integrand values for transition:")
    print("E_sq: ", np.max(omega[mask]**2))
    print("n: ", np.max(n[mask]**2 + n[mask]))

    for ms in self.ms_values:
      for ms_prime in self.ms_values:
        if ms == ms_prime:
          continue
        rate_key = f"{ms}_to_{ms_prime}"
        total_integrand = np.zeros_like(omega[mask])
        for ms_double_prime in self.ms_values:
          J_a    = get_J_path(ms_prime, ms_double_prime)[mask]
          J_b    = get_J_path(ms_double_prime, ms)[mask]
          E_sq   = omega[mask]**2
          total_integrand += (J_a * J_b / E_sq) * n[mask] * (n[mask] + 1)
        self.transition_rate["second_order"][rate_key] = f2_factor * np.sum(total_integrand)

    self.transition_rate["second_order"]["0_to_1"]  = (
        self.transition_rate["second_order"]["0_to_1"]
      + self.transition_rate["second_order"]["0_to_-1"]
      + self.transition_rate["second_order"]["-1_to_0"]
      + self.transition_rate["second_order"]["1_to_0"]
    )
    self.transition_rate["second_order"]["1_to_-1"] = (
        self.transition_rate["second_order"]["1_to_-1"]
      + self.transition_rate["second_order"]["-1_to_1"]
    )

    if self.data_2ph is not None:
      self._compute_two_phonon_rates(T, omega_spin_meV, res=res, sigma=sigma)

    return self.transition_rate
