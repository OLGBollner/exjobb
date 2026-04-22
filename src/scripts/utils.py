import numpy as np
from scipy import constants as Cn


def smear_data(freqs, values, res, sigma):
    freqs = np.array(freqs)
    values = np.array(values)
    f_min = np.min(0)
    f_max = np.max(freqs) + 5 * sigma
    x_grid = np.arange(f_min, f_max, res)

    y_dense = np.zeros_like(x_grid)

    indices = np.round(freqs / res).astype(int)
    for i, idx in enumerate(indices):
        if 0 <= idx < len(y_dense):
            y_dense[idx] += values[i]

    x_kernel = np.arange(-4 * sigma, 4 * sigma + res, res)
    kernel = np.exp(-0.5 * (x_kernel / sigma)**2) / (sigma*np.sqrt(2*Cn.pi))

    y_smooth = np.convolve(y_dense**2, kernel, mode="same")

    return x_grid, y_smooth

def calc_ipr(phonons):
  phonon_modes = phonons["eigs"]
  freqs = phonons["freqs"]
  nphonon = freqs.shape[0]
  lattice_points = phonons["atoms"]
  lattice_vecs = phonons["lattice"]
  cartesian_points = np.zeros(lattice_points.shape)

  for i, point in enumerate(lattice_points):
    cartesian_point = np.zeros((1,3))
    for x, vector in zip(point, np.split(lattice_vecs, 3, axis=0)):
        if x > 0.5:
            x -= 1
        cartesian_point += x*np.reshape(vector, (1,3))
    cartesian_points[i] = cartesian_point

  defect_pos = np.array([0,0,0])

  mode_iprs = np.zeros(nphonon)
  for i in range(nphonon):
    disp_4 = 0
    disp_total = 0
    for mode in phonon_modes[i]:
        disp = np.dot(mode, mode)
        disp_4 += disp**2
        disp_total += disp
    mode_iprs[i] = disp_4/(disp_total**2)

  return mode_iprs
