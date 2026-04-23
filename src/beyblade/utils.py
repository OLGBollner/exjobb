import numpy as np
from scipy import constants as Cn
from scipy.ndimage import convolve
from typing import Tuple, Dict, Any

class MathUtils:

    @staticmethod
    def broad_delta(omega, omega_p, sigma):
        delta = np.exp(-0.5 * (omega_p - omega)**2 / (sigma**2)) * 1 / (np.sqrt(2*Cn.pi)*sigma)**omega.ndim
        print(np.sum(delta))
        return delta

    @staticmethod
    def smear_data(freqs: np.ndarray, values: np.ndarray, res: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        freqs = np.array(freqs)
        values = np.array(values)
        f_min = 0
        f_max = np.max(freqs) + 5 * sigma
        x_grid = np.arange(f_min, f_max, res)

        y_dense = np.zeros_like(x_grid)
        indices = np.round(freqs / res).astype(int)
        for i, idx in enumerate(indices):
            if 0 <= idx < len(y_dense):
                y_dense[idx] += values[i]

        x_kernel = np.arange(-4 * sigma, 4 * sigma + res, res)
        kernel = np.exp(-0.5 * (x_kernel / sigma)**2) / (sigma * np.sqrt(2 * Cn.pi))
        kernel /= np.sum(kernel)

        y_smooth = np.convolve(y_dense**2, kernel, mode="same")

        return x_grid, y_smooth

    @staticmethod
    def get_2d_spectral_density(freqs, values, sigma_phys, res):
        sigma_pixel = sigma_phys / res

        freqs = np.array(freqs)
        values = np.array(values)

        # print("frequency shape: ", freqs.shape)
        # print("values shape: ", values.shape)
        values = values.flatten()

        f_min = 0
        f_max = np.max(freqs) + 5 * sigma_phys

        x_grid = np.arange(f_min, f_max, res)
        y_grid = np.arange(f_min, f_max, res)
        dense_values = np.zeros((len(x_grid), len(y_grid)))

        freq_x, freq_y = np.meshgrid(freqs, freqs)
        freq_x_flat = freq_x.flatten()
        freq_y_flat = freq_y.flatten()

        idx_x = np.round(freq_x_flat / res).astype(int)
        idx_y = np.round(freq_y_flat / res).astype(int)

        mask = (idx_x >= 0) & (idx_x < len(x_grid)) & \
               (idx_y >= 0) & (idx_y < len(y_grid))

        np.add.at(dense_values, (idx_x[mask], idx_y[mask]), values[mask]**2)

        radius_pixel = int(np.ceil(4 * sigma_pixel))
        grid_1d = np.arange(-radius_pixel, radius_pixel + 1)
        x_kernel, y_kernel = np.meshgrid(grid_1d, grid_1d)

        kernel = np.exp(-0.5 * (x_kernel**2 + y_kernel**2) / sigma_pixel**2) / (2 * np.pi * sigma_pixel**2)
        kernel /= np.sum(kernel)

        X, Y = np.meshgrid(x_grid, y_grid)

        return X, Y, convolve(dense_values, kernel)

    @staticmethod
    def calc_ipr(phonons: Dict[str, Any]) -> np.ndarray:
        phonon_modes = phonons["eigs"]
        freqs = phonons["freqs"]
        nphonon = freqs.shape[0]
        lattice_points = phonons["atoms"]
        lattice_vecs = phonons["lattice"]
        cartesian_points = np.zeros(lattice_points.shape)

        for i, point in enumerate(lattice_points):
            cartesian_point = np.zeros((1, 3))
            for x, vector in zip(point, np.split(lattice_vecs, 3, axis=0)):
                if x > 0.5:
                    x -= 1
                cartesian_point += x * np.reshape(vector, (1, 3))
            cartesian_points[i] = cartesian_point

        mode_iprs = np.zeros(nphonon)
        for i in range(nphonon):
            disp_4 = 0
            disp_total = 0
            for mode in phonon_modes[i]:
                disp = np.dot(mode, mode)
                disp_4 += disp**2
                disp_total += disp
            mode_iprs[i] = disp_4 / (disp_total**2)

        return mode_iprs

    @staticmethod
    def get_c3_111_matrix():
        return np.array([[0, 0, 1],
                         [1, 0, 0],
                         [0, 1, 0]])

    @staticmethod
    def get_sv_matrix(axis_sv: list = [1, -1, 0]):
        n = np.array(axis_sv) / np.linalg.norm(axis_sv)
        return np.eye(3) - 2 * np.outer(n, n)

