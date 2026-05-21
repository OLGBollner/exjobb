import numpy as np
from scipy import constants as Cn
from scipy.ndimage import convolve
from typing import Tuple, Dict, Any

class MathUtils:

    @staticmethod
    def broad_delta(omega, omega_p, sigma):
        delta = np.exp(-0.5 * (omega_p - omega)**2 / (sigma**2)) * 1 / (np.sqrt(2*Cn.pi)*sigma)**omega.ndim
        delta /= delta.sum()
        return delta

    @staticmethod
    def expand_data(freqs: np.ndarray, values: np.ndarray, res: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
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

        return x_grid, y_dense

    @staticmethod
    def smear_data(freqs: np.ndarray, values: np.ndarray, res: float, sigma: float) -> Tuple[np.ndarray, np.ndarray]:
        x_grid, y_dense = MathUtils.expand_data(freqs, values, res, sigma)

        x_kernel = np.arange(-4 * sigma, 4 * sigma + res, res)
        kernel = np.exp(-0.5 * (x_kernel / sigma)**2) / (sigma * np.sqrt(2 * Cn.pi))

        y_smooth = np.convolve(y_dense, kernel, mode="same")

        return x_grid, y_smooth

    @staticmethod
    def expand_data_2d(freqs, values, res, sigma):
        # TODO: debugging
        # print("Expanding data...")
        freqs = np.array(freqs)

        values = np.array(values)

        values = values.flatten()

        f_min = 0
        f_max = np.max(freqs) + 5 * sigma

        x_array = np.arange(f_min, f_max, res)
        y_array = np.arange(f_min, f_max, res)
        # print("Grid size:")
        # print(x_array.shape, y_array.shape)
        dense_values = np.zeros((len(x_array), len(y_array)))

        freq_x, freq_y = np.meshgrid(freqs, freqs)
        freq_x_flat = freq_x.flatten()
        freq_y_flat = freq_y.flatten()

        idx_x = np.round(freq_x_flat / res).astype(int)
        idx_y = np.round(freq_y_flat / res).astype(int)

        mask = (idx_x >= 0) & (idx_x < len(x_array)) & \
               (idx_y >= 0) & (idx_y < len(y_array))

        np.add.at(dense_values, (idx_x[mask], idx_y[mask]), values[mask])

        return x_array, y_array, dense_values

    @staticmethod
    def get_2d_spectral_density(freqs, values, res, sigma_phys):

        x_array, y_array, dense_values = MathUtils.expand_data_2d(freqs, values, sigma=sigma_phys, res=res)

        sigma_pixel = sigma_phys / res
        radius_pixel = int(np.ceil(4 * sigma_pixel))
        grid_1d = np.arange(-radius_pixel, radius_pixel + 1)
        x_kernel, y_kernel = np.meshgrid(grid_1d, grid_1d)

        kernel = np.exp(-0.5 * (x_kernel**2 + y_kernel**2) / sigma_pixel**2) / (2 * np.pi * sigma_pixel**2)
        kernel /= np.sum(kernel)

        X, Y = np.meshgrid(x_array, y_array)

        return X, Y, convolve(dense_values, kernel, cval=0, mode="constant")

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
    def rotation_matrix_around_axis(axis, angle):
        """
        Compute the 3x3 rotation matrix for a rotation by `angle` (in radians)
        around the given axis.

        Parameters
        ----------
        axis : array_like (3,)
            The rotation axis. It will be normalized internally.
        angle : float
            Rotation angle in radians (right-hand rule).

        Returns
        -------
        R : numpy.ndarray (3,3)
            Rotation matrix.
        """
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        omc = 1 - cos_a  # 1 - cos(angle)

        R = np.array([
            [cos_a + x*x*omc,      x*y*omc - z*sin_a,   x*z*omc + y*sin_a],
            [y*x*omc + z*sin_a,    cos_a + y*y*omc,     y*z*omc - x*sin_a],
            [z*x*omc - y*sin_a,    z*y*omc + x*sin_a,   cos_a + z*z*omc]
        ])
        return R

    @staticmethod
    def rotation_around_symmetry_axis(eigvec, order=3):
        """
        Return the rotation matrix for a C<order> rotation around the given
        eigenvector (symmetry axis).

        eigvec : array_like (3,)
            The “z‑axis” eigenvector (e.g., the one with unique eigenvalue).
        order  : int
            Rotation order (2 → 180°, 3 → 120°, etc.).
        """
        angle = 2 * np.pi / order
        return MathUtils.rotation_matrix_around_axis(eigvec, angle)

    @staticmethod
    def reflection_matrix(plane_normal):
        """
        Return the 3x3 Householder matrix that reflects a vector
        across the plane whose unit normal is `plane_normal`.

        Parameters
        ----------
        plane_normal : array_like (3,)
            A vector perpendicular to the mirror plane. It will be normalised.

        Returns
        -------
        M : numpy.ndarray (3,3)
        """
        n = np.asarray(plane_normal, dtype=float)
        n = n / np.linalg.norm(n)
        return np.eye(3) - 2.0 * np.outer(n, n)

    @staticmethod
    def fmt(val, precision=6):
        """Return a nicely formatted string for either a number or a tuple of numbers."""
        if isinstance(val, tuple):
            # For tuples, show each element with the given precision,
            # separated by commas, inside parentheses.
            return "(" + ", ".join(f"{v:.{precision}f}" for v in val) + ")"
        else:
            # Assume it's a single number (float/int).
            return f"{val:.{precision}f}"

    @staticmethod
    def calc_symmetry(sym_a: str, sym_b: str) -> list[str]:
        """Calculate resulting symmetry irreps for the C3v point group."""
        
        def parse_irrep(sym: str) -> str:
            sym_upper = sym.upper()
            if sym_upper in ("EX", "EY"):
                return "E"
            return sym_upper

        a_parsed = parse_irrep(sym_a)
        b_parsed = parse_irrep(sym_b)

        product_table = {
            ("A1", "A1"): ["A1"],
            ("A1", "A2"): ["A2"],
            ("A1", "E"): ["E"],
            ("A2", "A1"): ["A2"],
            ("A2", "A2"): ["A1"],
            ("A2", "E"): ["E"],
            ("E", "A1"): ["E"],
            ("E", "A2"): ["E"],
            ("E", "E"): ["A1", "A2", "E"],
        }
        
        pair = (a_parsed, b_parsed)
        if pair in product_table:
            return product_table[pair]
            
        reverse_pair = (b_parsed, a_parsed)
        if reverse_pair in product_table:
            return product_table[reverse_pair]
            
        raise ValueError(f"Direct product for irreps {sym_a} and {sym_b} is not defined.")

