from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
import numpy as np
from scipy import constants as Cn

from beyblade.constants import CONSTANTS


class EnergyUnit(str, Enum):
    MHZ = "MHz"
    JOULE = "J"
    MEV = "meV"
    GHZ = "GHz"
    THZ = "THz"


@dataclass
class ZFSTensor:
    """
    Represents a Zero-Field Splitting (ZFS) tensor in 3x3 Cartesian matrix form.
    Default unit is MHz.
    """
    matrix: np.ndarray  # Shape (3, 3)
    unit: str = "MHz"

    def __post_init__(self):
        self.matrix = np.asarray(self.matrix, dtype=float)
        if self.matrix.shape != (3, 3):
            raise ValueError(f"ZFS tensor matrix must have shape (3, 3), got {self.matrix.shape}")

    def traceless(self) -> np.ndarray:
        """Returns the traceless part of the ZFS matrix."""
        tr = np.trace(self.matrix) / 3.0
        return self.matrix - tr * np.eye(3)

    def principal_components(self) -> tuple[float, float, float, np.ndarray]:
        """
        Calculates principal values (D_xx, D_yy, D_zz) and eigenvectors
        following standard EPR convention: |D_zz| >= |D_yy| >= |D_xx| (with trace=0).
        
        Returns:
            (D_xx, D_yy, D_zz, eigenvectors)
        """
        D_tl = self.traceless()
        evals, evecs = np.linalg.eigh(D_tl)

        # Sort by absolute magnitude so that |D_zz| is largest
        abs_order = np.argsort(np.abs(evals))  # [smallest, middle, largest]
        ix, iy, iz = abs_order[0], abs_order[1], abs_order[2]

        D_xx_val = evals[ix]
        D_yy_val = evals[iy]
        D_zz_val = evals[iz]

        # Ensure right-handed coordinate system for eigenvectors
        R = np.column_stack([evecs[:, ix], evecs[:, iy], evecs[:, iz]])
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]

        return float(D_xx_val), float(D_yy_val), float(D_zz_val), R

    @property
    def D(self) -> float:
        """Axial ZFS parameter D = 3/2 * D_zz in the principal frame."""
        _, _, D_zz_val, _ = self.principal_components()
        return 1.5 * D_zz_val

    @property
    def E(self) -> float:
        """Rhombic ZFS parameter E = (D_xx - D_yy) / 2 in the principal frame."""
        D_xx_val, D_yy_val, _, _ = self.principal_components()
        return 0.5 * (D_xx_val - D_yy_val)

    @property
    def xx(self) -> float:
        return float(self.matrix[0, 0])

    @property
    def yy(self) -> float:
        return float(self.matrix[1, 1])

    @property
    def zz(self) -> float:
        return float(self.matrix[2, 2])

    @property
    def xy(self) -> float:
        return float(self.matrix[0, 1])

    @property
    def xz(self) -> float:
        return float(self.matrix[0, 2])

    @property
    def yz(self) -> float:
        return float(self.matrix[1, 2])

    def to_unit(self, target_unit: str) -> ZFSTensor:
        """Convert tensor matrix into target energy unit."""
        if self.unit == target_unit:
            return ZFSTensor(matrix=self.matrix.copy(), unit=self.unit)

        # Convert to MHz first
        if self.unit == "MHz":
            val_mhz = self.matrix
        elif self.unit == "J":
            val_mhz = self.matrix / CONSTANTS["MHz2J"]
        elif self.unit == "meV":
            val_mhz = self.matrix / CONSTANTS["MHz2meV"]
        elif self.unit == "GHz":
            val_mhz = self.matrix * 1e3
        else:
            raise ValueError(f"Unsupported source unit: {self.unit}")

        # Convert from MHz to target
        if target_unit == "MHz":
            out_mat = val_mhz
        elif target_unit == "J":
            out_mat = val_mhz * CONSTANTS["MHz2J"]
        elif target_unit == "meV":
            out_mat = val_mhz * CONSTANTS["MHz2meV"]
        elif target_unit == "GHz":
            out_mat = val_mhz * 1e-3
        else:
            raise ValueError(f"Unsupported target unit: {target_unit}")

        return ZFSTensor(matrix=out_mat, unit=target_unit)

    def rotate(self, rotation_matrix: np.ndarray) -> ZFSTensor:
        """Rotate tensor: D' = R @ D @ R.T"""
        R = np.asarray(rotation_matrix, dtype=float)
        rotated_mat = R @ self.matrix @ R.T
        return ZFSTensor(matrix=rotated_mat, unit=self.unit)


@dataclass
class PhononMode:
    """Represents a single vibrational mode."""
    index: int
    frequency_mev: float
    eigenvector: np.ndarray  # Shape (N_atoms, 3)
    symmetry: Optional[str] = None
    ipr: Optional[float] = None

    @property
    def frequency_thz(self) -> float:
        return self.frequency_mev / CONSTANTS["THz2meV"]

    @property
    def frequency_rads(self) -> float:
        return self.frequency_mev * CONSTANTS["meV2rads"]


@dataclass
class PhononSpectrum:
    """Represents a collection of phonon modes in a supercell."""
    frequencies_mev: np.ndarray             # Shape (N_modes,)
    eigenvectors: np.ndarray                # Shape (N_modes, N_atoms, 3)
    atom_frac_coords: np.ndarray            # Shape (N_atoms, 3)
    atom_symbols: list[str]                 # Length N_atoms
    atomic_masses: np.ndarray               # Shape (N_atoms,)
    lattice: np.ndarray                     # Shape (3, 3)
    symmetries: Optional[list[str]] = None  # Length N_modes
    iprs: Optional[np.ndarray] = None       # Shape (N_modes,)

    @property
    def n_modes(self) -> int:
        return len(self.frequencies_mev)

    @property
    def n_atoms(self) -> int:
        return len(self.atom_symbols)

    def get_mode(self, idx: int) -> PhononMode:
        sym = self.symmetries[idx] if self.symmetries is not None else None
        ipr_val = float(self.iprs[idx]) if self.iprs is not None else None
        return PhononMode(
            index=idx,
            frequency_mev=float(self.frequencies_mev[idx]),
            eigenvector=self.eigenvectors[idx],
            symmetry=sym,
            ipr=ipr_val
        )

    def filter_by_energy(self, min_mev: float = -np.inf, max_mev: float = np.inf) -> np.ndarray:
        """Returns mode indices within energy range [min_mev, max_mev]."""
        mask = (self.frequencies_mev >= min_mev) & (self.frequencies_mev <= max_mev)
        return np.where(mask)[0]

    def filter_by_symmetry(self, symmetry_label: str) -> np.ndarray:
        """Returns mode indices matching the given symmetry label."""
        if self.symmetries is None:
            return np.array([], dtype=int)
        return np.array([i for i, sym in enumerate(self.symmetries) if sym == symmetry_label], dtype=int)

    def get_phonon_pert(self, perturbation_scale_si: float = 1.0) -> dict[str, Any]:
        """
        Computes mass-weighted perturbation displacements (SI), frequencies (J), symmetries, and IPRs.

        The perturbation amplitude for each mode follows the mass-weighted
        phonon coordinate  q = q0 * sqrt(2*omega/hbar), matching the original
        PhononManager implementation. Modes with frequency <= 0 get None.
        """
        omega_rads = self.frequencies_mev * CONSTANTS["meV2rads"]
        displacements = [
            perturbation_scale_si * np.sqrt(2.0 * omega / Cn.hbar) if freq > 0 else None
            for freq, omega in zip(self.frequencies_mev, omega_rads)
        ]
        freqs_j = self.frequencies_mev * CONSTANTS["meV2J"]
        iprs = self.iprs if self.iprs is not None else np.zeros(self.n_modes)
        syms = self.symmetries if self.symmetries is not None else ["A1"] * self.n_modes

        return {
            "eigs": displacements,
            "freqs": freqs_j,
            "sym": syms,
            "ipr": iprs,
        }


@dataclass
class PerturbationEntry:
    """Represents a single 1D or 2D perturbed calculation."""
    order: int  # 1 for 1D (dD/dq), 2 for 2D (d2D/dq_i dq_j)
    mode_indices: tuple[int, ...]
    amplitude: Union[float, tuple[float, float]]
    zfs_tensor: ZFSTensor
    energy: Optional[float] = None


@dataclass
class RawZFSData:
    """Container for raw unperturbed and perturbed ZFS simulation data."""
    defect: str
    cell_size: int
    pert_scale: float
    ground_state_zfs: Optional[ZFSTensor] = None
    first_order: dict[int, PerturbationEntry] = field(default_factory=dict)
    second_order: dict[tuple[int, int], PerturbationEntry] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
