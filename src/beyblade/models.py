from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import numpy as np
from scipy import constants as Cn

from beyblade.constants import CONSTANTS


class EnergyUnit(str, Enum):
    MHZ = "MHz"
    JOULE = "J"
    MEV = "meV"
    GHZ = "GHz"
    THZ = "THz"


def convert_energy(values: Union[float, np.ndarray], from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
    """
    Converts energy / frequency values between MHz, J, meV, GHz, and THz.
    """
    if from_unit == to_unit:
        return values.copy() if isinstance(values, np.ndarray) else values

    # Convert source unit to Joules (SI)
    if from_unit == "J":
        val_j = values
    elif from_unit == "MHz":
        val_j = values * CONSTANTS["MHz2J"]
    elif from_unit == "meV":
        val_j = values * CONSTANTS["meV2J"]
    elif from_unit == "GHz":
        val_j = values * (CONSTANTS["MHz2J"] * 1e3)
    elif from_unit == "THz":
        val_j = values * (CONSTANTS["THz2meV"] * CONSTANTS["meV2J"])
    else:
        raise ValueError(f"Unsupported source unit: '{from_unit}'")

    # Convert Joules (SI) to target unit
    if to_unit == "J":
        return val_j
    elif to_unit == "MHz":
        return val_j / CONSTANTS["MHz2J"]
    elif to_unit == "meV":
        return val_j / CONSTANTS["meV2J"]
    elif to_unit == "GHz":
        return val_j / (CONSTANTS["MHz2J"] * 1e3)
    elif to_unit == "THz":
        return val_j / (CONSTANTS["THz2meV"] * CONSTANTS["meV2J"])
    else:
        raise ValueError(f"Unsupported target unit: '{to_unit}'")


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
        """Convert tensor matrix into target energy unit using convert_energy."""
        if self.unit == target_unit:
            return ZFSTensor(matrix=self.matrix.copy(), unit=self.unit)
        out_mat = convert_energy(self.matrix, self.unit, target_unit)
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
    frequency_unit: str = "meV"

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

    def frequencies_to_unit(self, target_unit: str) -> np.ndarray:
        """Convert mode frequencies from meV to target energy/frequency unit."""
        return convert_energy(self.frequencies_mev, "meV", target_unit)

    def get_phonon_pert(self, perturbation_scale_si: float = 1.0) -> dict[str, Any]:
        """
        Computes mass-weighted perturbation displacements (SI), frequencies (J), symmetries, and IPRs.

        The perturbation amplitude for each mode follows the mass-weighted
        phonon coordinate  q = q0 * sqrt(2*omega/hbar). Modes with frequency <= 0 get None.
        """
        omega_rads = self.frequencies_mev * CONSTANTS["meV2rads"]
        displacements = [
            perturbation_scale_si * np.sqrt(2.0 * omega / Cn.hbar) if freq > 0 else None
            for freq, omega in zip(self.frequencies_mev, omega_rads)
        ]
        freqs_j = self.frequencies_to_unit("J")
        iprs = self.iprs if self.iprs is not None else np.zeros(self.n_modes)
        syms = self.symmetries if self.symmetries is not None else ["A1"] * self.n_modes

        return {
            "disp": displacements,
            "freqs": freqs_j,
            "sym": syms,
            "ipr": iprs,
        }

    def save(self, out_path: Union[str, Path]) -> str:
        """Saves spectrum to .npz file with explicit frequency unit tag."""
        path = str(out_path)
        if not path.endswith(".npz"):
            path += ".npz"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            frequencies=self.frequencies_mev,
            frequencies_mev=self.frequencies_mev,
            frequency_unit="meV",
            eigenvectors=self.eigenvectors,
            atom_frac_coords=self.atom_frac_coords,
            atom_symbols=self.atom_symbols,
            atomic_masses=self.atomic_masses,
            lattice=self.lattice,
            symmetries=self.symmetries,
            iprs=self.iprs,
        )
        return path

    @classmethod
    def load(cls, in_path: Union[str, Path]) -> PhononSpectrum:
        """Loads spectrum from .npz file, converting frequencies using explicit unit tags."""
        data = np.load(str(in_path), allow_pickle=True)
        unit = str(data["frequency_unit"]) if "frequency_unit" in data else "meV"
        freqs_raw = data["frequencies"] if "frequencies" in data else data["frequencies_mev"]
        freqs_mev = convert_energy(freqs_raw, unit, "meV")

        syms = list(data["symmetries"]) if "symmetries" in data and data["symmetries"] is not None else None
        iprs = data["iprs"] if "iprs" in data else None

        return cls(
            frequencies_mev=freqs_mev,
            eigenvectors=data["eigenvectors"],
            atom_frac_coords=data["atom_frac_coords"],
            atom_symbols=list(data["atom_symbols"]),
            atomic_masses=data["atomic_masses"],
            lattice=data["lattice"],
            symmetries=syms,
            iprs=iprs,
            frequency_unit="meV",
        )


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
    calc_method: Optional[str] = None
    order: Optional[int] = None
    ground_state_zfs: Optional[ZFSTensor] = None
    eigen_rotation: Optional[np.ndarray] = None
    first_order: dict[int, Union[PerturbationEntry, dict[str, Any]]] = field(default_factory=dict)
    second_order: dict[tuple[int, int], Union[PerturbationEntry, dict[str, Any]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_unit(self, target_unit: str) -> RawZFSData:
        """Converts all tensors in the container to the specified unit."""
        new_gs = self.ground_state_zfs.to_unit(target_unit) if self.ground_state_zfs else None

        new_first = {}
        for k, v in self.first_order.items():
            if isinstance(v, PerturbationEntry):
                new_first[k] = PerturbationEntry(
                    order=v.order,
                    mode_indices=v.mode_indices,
                    amplitude=v.amplitude,
                    zfs_tensor=v.zfs_tensor.to_unit(target_unit),
                    energy=v.energy,
                )
            elif isinstance(v, dict) and "tensor" in v:
                curr_unit = v.get("unit", "J")
                new_v = dict(v)
                new_v["tensor"] = convert_energy(v["tensor"], curr_unit, target_unit)
                new_v["unit"] = target_unit
                new_first[k] = new_v
            else:
                new_first[k] = v

        new_second = {}
        for k, v in self.second_order.items():
            if isinstance(v, PerturbationEntry):
                new_second[k] = PerturbationEntry(
                    order=v.order,
                    mode_indices=v.mode_indices,
                    amplitude=v.amplitude,
                    zfs_tensor=v.zfs_tensor.to_unit(target_unit),
                    energy=v.energy,
                )
            elif isinstance(v, dict) and "tensor" in v:
                curr_unit = v.get("unit", "J")
                new_v = dict(v)
                new_v["tensor"] = convert_energy(v["tensor"], curr_unit, target_unit)
                new_v["unit"] = target_unit
                new_second[k] = new_v
            else:
                new_second[k] = v

        return RawZFSData(
            defect=self.defect,
            cell_size=self.cell_size,
            pert_scale=self.pert_scale,
            calc_method=self.calc_method,
            order=self.order,
            ground_state_zfs=new_gs,
            eigen_rotation=self.eigen_rotation.copy() if self.eigen_rotation is not None else None,
            first_order=new_first,
            second_order=new_second,
            metadata=dict(self.metadata),
        )

    def save(self, out_path: Union[str, Path]) -> str:
        """Saves RawZFSData to a .npz file with latest naming conventions and explicit unit metadata."""
        path = str(out_path)
        if not path.endswith(".npz"):
            path += ".npz"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        gs_mat = self.ground_state_zfs.matrix if self.ground_state_zfs else None
        gs_unit = self.ground_state_zfs.unit if self.ground_state_zfs else "MHz"

        # Format dictionaries into clean serializable entries
        saved_1d = {}
        for idx, entry in self.first_order.items():
            if isinstance(entry, PerturbationEntry):
                saved_1d[idx] = {
                    "tensor": entry.zfs_tensor.matrix,
                    "unit": entry.zfs_tensor.unit,
                    "pert": entry.amplitude,
                }
            elif isinstance(entry, dict):
                saved_1d[idx] = entry

        saved_2d = {}
        for idx, entry in self.second_order.items():
            # For 2D keys, tuple (i, j) can be stored as "i_j" string for numpy compatibility
            key = f"{idx[0]}_{idx[1]}" if isinstance(idx, tuple) else str(idx)
            if isinstance(entry, PerturbationEntry):
                saved_2d[key] = {
                    "tensor": entry.zfs_tensor.matrix,
                    "unit": entry.zfs_tensor.unit,
                    "pert": entry.amplitude,
                }
            elif isinstance(entry, dict):
                saved_2d[key] = entry

        np.savez(
            path,
            order=self.order if self.order is not None else (1 if self.first_order else 2),
            defect=self.defect,
            cell_size=self.cell_size,
            pert_scale=self.pert_scale,
            calc_method=self.calc_method,
            eigen_rotation=self.eigen_rotation,
            ground_state_zfs_matrix=gs_mat,
            ground_state_zfs_unit=gs_unit,
            first_order=saved_1d if saved_1d else None,
            second_order=saved_2d if saved_2d else None,
        )
        return path

    @classmethod
    def load(cls, in_path: Union[str, Path, Sequence[Union[str, Path]]]) -> RawZFSData:
        """
        Loads RawZFSData from one or more .npz files.
        Maintains backward compatibility with legacy keys (zfs_tensors, zfs_tensors_2d, zfs_relaxed).
        """
        if isinstance(in_path, (list, tuple)):
            paths = [Path(p) for p in in_path]
        else:
            paths = [Path(in_path)]

        raw_data = [np.load(str(p), allow_pickle=True) for p in paths]
        base = raw_data[0]

        defect = str(base["defect"])
        cell_size = int(base["cell_size"])
        pert_scale = float(base["pert_scale"])
        calc_method = str(base["calc_method"]) if "calc_method" in base else None
        order = int(base["order"]) if "order" in base else None
        eigen_rot = base["eigen_rotation"] if "eigen_rotation" in base and base["eigen_rotation"] is not None else None

        # Reconstruct ground state ZFSTensor with explicit unit
        if "ground_state_zfs_matrix" in base and base["ground_state_zfs_matrix"] is not None:
            mat = base["ground_state_zfs_matrix"]
            unit = str(base.get("ground_state_zfs_unit", "MHz"))
            gs_tensor = ZFSTensor(matrix=mat, unit=unit)
        elif "zfs_relaxed" in base and base["zfs_relaxed"] is not None:
            # Legacy files store zfs_relaxed in Joules
            mat_j = base["zfs_relaxed"]
            gs_tensor = ZFSTensor(matrix=mat_j, unit="J")
        else:
            gs_tensor = None

        first_order = {}
        second_order = {}

        for data in raw_data:
            # Check latest key first, fallback to legacy
            if "first_order" in data and data["first_order"] is not None:
                d1 = data["first_order"][()]
                if isinstance(d1, dict):
                    first_order.update(d1)
            elif "zfs_tensors" in data and data["zfs_tensors"] is not None:
                d1 = data["zfs_tensors"][()]
                if isinstance(d1, dict):
                    first_order.update(d1)

            if "second_order" in data and data["second_order"] is not None:
                d2 = data["second_order"][()]
                if isinstance(d2, dict):
                    for k, v in d2.items():
                        # Parse tuple key from string "i_j" if needed
                        parsed_key = tuple(int(x) for x in k.split("_")) if isinstance(k, str) and "_" in k else k
                        second_order[parsed_key] = v
            elif "zfs_tensors_2d" in data and data["zfs_tensors_2d"] is not None:
                d2 = data["zfs_tensors_2d"][()]
                if isinstance(d2, dict):
                    for k, v in d2.items():
                        parsed_key = tuple(int(x) for x in k.split("_")) if isinstance(k, str) and "_" in k else k
                        second_order[parsed_key] = v

        return cls(
            defect=defect,
            cell_size=cell_size,
            pert_scale=pert_scale,
            calc_method=calc_method,
            order=order,
            ground_state_zfs=gs_tensor,
            eigen_rotation=eigen_rot,
            first_order=first_order,
            second_order=second_order,
        )


@dataclass
class SpinPhononCouplingData:
    """
    Container for calculated spin-phonon coupling coefficients (V-tensors),
    finite-difference derivatives, and frequencies with explicit unit tracking.
    """
    order: int
    defect: str
    cell_size: int
    pert_scale: float
    calc_method: Optional[str] = None
    frequencies: np.ndarray = field(default_factory=lambda: np.array([]))
    frequency_unit: str = "J"
    V_0_0: np.ndarray = field(default_factory=lambda: np.array([]))
    V_p_m: np.ndarray = field(default_factory=lambda: np.array([]))
    V_0_pm: np.ndarray = field(default_factory=lambda: np.array([]))
    coupling_unit: str = "J"
    ground_state_zfs: Optional[ZFSTensor] = None
    zfs_derivs: Optional[np.ndarray] = None
    derivs_unit: str = "J"
    symmetries: Optional[list[str]] = None
    iprs: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_unit(self, target_unit: str) -> SpinPhononCouplingData:
        """
        Returns a new instance with coupling coefficients (V_0_0, V_p_m, V_0_pm, zfs_derivs)
        and ground_state_zfs converted to target_unit.
        """
        if self.coupling_unit == target_unit:
            new_v00 = self.V_0_0.copy()
            new_vpm = self.V_p_m.copy()
            new_v0pm = self.V_0_pm.copy()
        else:
            new_v00 = convert_energy(self.V_0_0, self.coupling_unit, target_unit)
            new_vpm = convert_energy(self.V_p_m, self.coupling_unit, target_unit)
            new_v0pm = convert_energy(self.V_0_pm, self.coupling_unit, target_unit)

        new_derivs = None
        if self.zfs_derivs is not None:
            new_derivs = convert_energy(self.zfs_derivs, self.derivs_unit, target_unit)

        new_gs = self.ground_state_zfs.to_unit(target_unit) if self.ground_state_zfs else None

        return SpinPhononCouplingData(
            order=self.order,
            defect=self.defect,
            cell_size=self.cell_size,
            pert_scale=self.pert_scale,
            calc_method=self.calc_method,
            frequencies=self.frequencies.copy(),
            frequency_unit=self.frequency_unit,
            V_0_0=new_v00,
            V_p_m=new_vpm,
            V_0_pm=new_v0pm,
            coupling_unit=target_unit,
            ground_state_zfs=new_gs,
            zfs_derivs=new_derivs,
            derivs_unit=target_unit,
            symmetries=list(self.symmetries) if self.symmetries is not None else None,
            iprs=self.iprs.copy() if self.iprs is not None else None,
            metadata=dict(self.metadata),
        )

    def frequencies_to_unit(self, target_unit: str) -> SpinPhononCouplingData:
        """Returns a new instance with mode frequencies converted to target_unit."""
        if self.frequency_unit == target_unit:
            new_freqs = self.frequencies.copy()
        else:
            new_freqs = convert_energy(self.frequencies, self.frequency_unit, target_unit)

        return SpinPhononCouplingData(
            order=self.order,
            defect=self.defect,
            cell_size=self.cell_size,
            pert_scale=self.pert_scale,
            calc_method=self.calc_method,
            frequencies=new_freqs,
            frequency_unit=target_unit,
            V_0_0=self.V_0_0.copy(),
            V_p_m=self.V_p_m.copy(),
            V_0_pm=self.V_0_pm.copy(),
            coupling_unit=self.coupling_unit,
            ground_state_zfs=self.ground_state_zfs,
            zfs_derivs=self.zfs_derivs.copy() if self.zfs_derivs is not None else None,
            derivs_unit=self.derivs_unit,
            symmetries=list(self.symmetries) if self.symmetries is not None else None,
            iprs=self.iprs.copy() if self.iprs is not None else None,
            metadata=dict(self.metadata),
        )

    def save(self, out_path: Union[str, Path]) -> str:
        """Saves coupling data to .npz file with explicit unit metadata and legacy keys."""
        path = str(out_path)
        if not path.endswith(".npz"):
            path += ".npz"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        gs_mat = self.ground_state_zfs.matrix if self.ground_state_zfs else None
        gs_unit = self.ground_state_zfs.unit if self.ground_state_zfs else None
        gs_d_joule = self.ground_state_zfs.to_unit("J").D if self.ground_state_zfs else 0.0

        np.savez(
            path,
            order=self.order,
            defect=self.defect,
            cell_size=self.cell_size,
            pert_scale=self.pert_scale,
            calc_method=self.calc_method,
            # Frequencies
            frequencies=self.frequencies,
            freqs=self.frequencies,  # legacy alias
            frequency_unit=self.frequency_unit,
            # Coupling coefficients
            V_0_0=self.V_0_0,
            V_p_m=self.V_p_m,
            V_0_pm=self.V_0_pm,
            coupling_unit=self.coupling_unit,
            # Ground state
            ground_state_zfs_matrix=gs_mat,
            ground_state_zfs_unit=gs_unit,
            zfs=gs_d_joule,          # legacy scalar in Joules
            # Derivatives
            zfs_derivs=self.zfs_derivs,
            derivs_unit=self.derivs_unit,
            # Symmetry and locality
            sym=self.symmetries,
            symmetries=self.symmetries,
            ipr=self.iprs,
            iprs=self.iprs,
        )
        return path

    @classmethod
    def load(cls, in_path: Union[str, Path]) -> SpinPhononCouplingData:
        """Loads SpinPhononCouplingData from a .npz file, parsing explicit units."""
        data = np.load(str(in_path), allow_pickle=True)

        # Frequencies and unit
        freq_unit = str(data["frequency_unit"]) if "frequency_unit" in data else None
        freqs = data["frequencies"] if "frequencies" in data else data["freqs"]
        if freq_unit is None:
            # Infer legacy: if values are tiny (~1e-20), they are in Joules; else meV
            freq_unit = "J" if np.mean(freqs[freqs > 0]) < 1e-15 else "meV"

        # Coupling coefficients and unit
        coupling_unit = str(data["coupling_unit"]) if "coupling_unit" in data else "J"
        V_0_0 = data["V_0_0"]
        V_p_m = data["V_p_m"]
        V_0_pm = data["V_0_pm"]

        # Ground state ZFS
        if "ground_state_zfs_matrix" in data and data["ground_state_zfs_matrix"] is not None:
            gs_mat = data["ground_state_zfs_matrix"]
            gs_unit = str(data.get("ground_state_zfs_unit", "MHz"))
            gs = ZFSTensor(matrix=gs_mat, unit=gs_unit)
        elif "zfs" in data:
            # Legacy scalar D value in Joules
            d_val_j = float(data["zfs"])
            gs_mat = np.diag([-d_val_j / 3.0, -d_val_j / 3.0, 2.0 * d_val_j / 3.0])
            gs = ZFSTensor(matrix=gs_mat, unit="J")
        else:
            gs = None

        derivs = data["zfs_derivs"] if "zfs_derivs" in data else None
        derivs_unit = str(data.get("derivs_unit", coupling_unit))

        syms = list(data["sym"]) if "sym" in data and data["sym"] is not None else None
        iprs = data["ipr"] if "ipr" in data else None

        return cls(
            order=int(data["order"]) if "order" in data else 1,
            defect=str(data["defect"]) if "defect" in data else "unknown",
            cell_size=int(data["cell_size"]) if "cell_size" in data else 0,
            pert_scale=float(data["pert_scale"]) if "pert_scale" in data else 0.0,
            calc_method=str(data["calc_method"]) if "calc_method" in data else None,
            frequencies=freqs,
            frequency_unit=freq_unit,
            V_0_0=V_0_0,
            V_p_m=V_p_m,
            V_0_pm=V_0_pm,
            coupling_unit=coupling_unit,
            ground_state_zfs=gs,
            zfs_derivs=derivs,
            derivs_unit=derivs_unit,
            symmetries=syms,
            iprs=iprs,
        )
