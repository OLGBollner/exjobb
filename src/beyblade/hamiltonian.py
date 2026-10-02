"""Hamiltonian assembly for the generalized spin-phonon model.

Zero-field scope: the Zeeman term is omitted (no B-field in the current
calculations). It can be added as a single function when needed.

H_ZFS = sum_ij D_ij S_i S_j

The isotropic part of D only shifts all eigenvalues by a constant and is
projected out (trace removal), matching the thesis convention where the
ZFS tensor is traceless by construction.
"""
import numpy as np

from beyblade.projection import rank2_spin_operators
from beyblade.spin import spin_i


def zfs_hamiltonian(S: float, D: np.ndarray) -> np.ndarray:
    """Zero-field ZFS Hamiltonian from a (3, 3) tensor in the defect frame.

    H = sum_ij D_ij S_i S_j. A non-isotropic trace (proportional to S(S+1)*1)
    shifts all levels equally and is removed for numerical cleanliness; the
    splitting is unaffected.
    """
    D = np.asarray(D, dtype=float)
    if D.shape != (3, 3):
        raise ValueError(f"D must be (3, 3), got {D.shape}")
    dim = int(round(2 * S)) + 1
    if abs(dim - (2 * S + 1)) > 1e-9:
        raise ValueError(f"S must be integer or half-integer, got {S}")
    H = np.zeros((dim, dim), dtype=complex)
    for i in range(3):
        for j in range(3):
            H += D[i, j] * (spin_i("xyz"[i], S) @ spin_i("xyz"[j], S))
    trace = np.trace(D)
    if abs(trace) > 0:
        H -= trace * S * (S + 1) / 3 * np.eye(dim)
    # D symmetric => H Hermitian with vanishing imaginary part
    assert np.allclose(H, H.conj().T), "H not Hermitian — is D symmetric?"
    return np.real_if_close(H)


# Channel-name to spin-operator mapping (thesis naming: the spin structure
# entering the ZFS tensor derivative for that term).
_CHANNEL_SPIN_OPS = {
    # V_0_pm: Delta m = +-1 (0<->+-1 transitions), pairs with F_xp = sqrt2{Sx,Sz}
    # V_p_m:  Delta m = +-2 (+-1<->+-1 transitions), pairs with F_x = (S+^2+S-^2)/2
    "V_0_0": ("F_z",),
    "V_0_pm": ("F_xp",),
    "V_p_m": ("F_x",),
}


def spin_phonon_coupling(S: float, channels: dict) -> np.ndarray:
    """First-order spin-phonon coupling operator for one mode, in units of Q.

    channels: {"V_0_0": v, "V_0_pm": v, "V_p_m": v} for an A1/E mode
    (missing keys = zero, by symmetry). The returned matrix multiplies the
    (dimensionless) normal coordinate Q: H_l = sum_mu V_mu F_mu.
    """
    dim = int(round(2 * S + 1))
    F = rank2_spin_operators(S)
    H = np.zeros((dim, dim), dtype=complex)
    for name, v in channels.items():
        if name not in _CHANNEL_SPIN_OPS:
            raise ValueError(f"unknown channel: {name!r}")
        for op in _CHANNEL_SPIN_OPS[name]:
            H += v * F[op]
    return np.real_if_close(H)
