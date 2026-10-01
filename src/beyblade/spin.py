"""General-spin operator construction.

Given spin S (integer or half-integer), builds the angular-momentum
operators in the m-descending basis (|S>, |S-1>, ..., |-S>), matching the
S=1 matrices from the thesis. Plain numpy arrays; no state classes.

    <m+1 | S_+ | m> = sqrt(S(S+1) - m(m+1))
"""
import numpy as np


def spin_z(S: float) -> np.ndarray:
    """Diagonal S_z in the m-descending basis."""
    dim = int(round(2 * S + 1))
    m = S - np.arange(dim)
    return np.diag(m.astype(float))


def spin_raise(S: float) -> np.ndarray:
    """S_+: |m> -> sqrt(S(S+1) - m(m+1)) |m+1>."""
    dim = int(round(2 * S + 1))
    m = S - np.arange(dim)          # ket index i has m_i
    coeffs = np.sqrt(S * (S + 1) - m * (m + 1))
    # <m_i+1 | S_+ | m_i> : row i-1, column i, for i = 1..dim-1
    mat = np.zeros((dim, dim))
    mat[np.arange(dim - 1), np.arange(1, dim)] = coeffs[1:]
    return mat


def spin_lower(S: float) -> np.ndarray:
    """S_- = S_+^dagger."""
    return spin_raise(S).conj().T


def spin_x(S: float) -> np.ndarray:
    return 0.5 * (spin_raise(S) + spin_lower(S))


def spin_y(S: float) -> np.ndarray:
    return -0.5j * (spin_raise(S) - spin_lower(S))


def spin_i(axis: str, S: float) -> np.ndarray:
    """Cartesian spin operator by axis name."""
    return {"x": spin_x, "y": spin_y, "z": spin_z}[axis](S)
