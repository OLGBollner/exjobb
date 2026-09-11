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
    pair_id: int = -1              # index of degenerate partner mode; -1 = unknown (no spectrum)
    original_index: int = -1       # mode index in the full DFT run; -1 = unknown

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
    e_pair_complete: Optional[list[bool]] = None  # Length N_modes
    pair_ids: Optional[np.ndarray] = None   # Shape (N_modes,): degenerate partner index, n_modes if unpaired (out-of-bounds sentinel)
    original_indices: Optional[np.ndarray] = None  # Shape (N_modes,): mode index in the full DFT run
    frequency_unit: str = "meV"

    def __post_init__(self):
        if self.symmetries is None and self.eigenvectors is not None and len(self.eigenvectors) > 0:
            self.analyze_c3v_symmetry()
        if self.e_pair_complete is None and self.symmetries is not None and len(self.frequencies_mev) > 0:
            self.check_e_pair_completeness()
        if self.pair_ids is None and self.e_pair_complete is not None:
            # Backfill pair_ids from e_pair_complete matching (legacy files)
            self.build_pair_ids()
        if self.original_indices is None:
            self.original_indices = np.arange(self.n_modes, dtype=int)

    @property
    def n_modes(self) -> int:
        return len(self.frequencies_mev)

    @property
    def n_atoms(self) -> int:
        return len(self.atom_symbols)

    def check_e_pair_completeness(self, tol_mev: float = 0.05) -> list[bool]:
        """
        Determines whether each vibrational mode has both Ex and Ey pairs in the dataset.
        For A1 and A2 modes, the value is False.
        For E modes (Ex or Ey), it is True if a matching partner of opposite symmetry
        exists within frequency tolerance tol_mev, otherwise False.
        """
        if self.symmetries is None or len(self.frequencies_mev) == 0:
            self.e_pair_complete = None
            return []

        n = len(self.frequencies_mev)
        completeness = [False] * n
        matched = set()

        for i in range(n):
            sym_i = self.symmetries[i]
            if sym_i not in ("Ex", "Ey"):
                continue

            target_sym = "Ey" if sym_i == "Ex" else "Ex"
            freq_i = self.frequencies_mev[i]

            best_j = None
            min_diff = float("inf")
            for j in range(n):
                if j == i or j in matched:
                    continue
                if self.symmetries[j] == target_sym:
                    diff = abs(self.frequencies_mev[j] - freq_i)
                    if diff < tol_mev and diff < min_diff:
                        min_diff = diff
                        best_j = j

            if best_j is not None:
                completeness[i] = True
                completeness[best_j] = True
                matched.add(i)
                matched.add(best_j)

        self.e_pair_complete = completeness
        return completeness

    def build_pair_ids(self, tol_mev: float = 0.05) -> np.ndarray:
        """
        Builds the symmetric pair_ids mapping between degenerate E-mode partners.

        pair_ids[i] is the index of mode i's degenerate partner (Ex <-> Ey within
        tol_mev), or n_modes if unpaired — the sentinel is deliberately out of bounds
        so that unguarded indexing (e.g. freqs[pair_ids[i]]) raises IndexError instead
        of silently wrapping around to the last element (as -1 would).
        The relation must be a strict involution:
        pair_ids[i] = j implies pair_ids[j] = i. Degenerate chains (a mode whose
        partner is already paired with a different mode) raise ValueError loudly.
        """
        if self.symmetries is None:
            self.pair_ids = np.full(self.n_modes, self.n_modes, dtype=int)
            return self.pair_ids

        n = self.n_modes
        pair_ids = np.full(n, n, dtype=int)

        for i in range(n):
            sym_i = self.symmetries[i]
            if sym_i not in ("Ex", "Ey") or pair_ids[i] != n:
                continue

            target_sym = "Ey" if sym_i == "Ex" else "Ex"
            freq_i = self.frequencies_mev[i]

            best_j = None
            min_diff = float("inf")
            for j in range(n):
                if j == i or pair_ids[j] != n:
                    continue
                if self.symmetries[j] == target_sym:
                    diff = abs(self.frequencies_mev[j] - freq_i)
                    if diff < tol_mev and diff < min_diff:
                        min_diff = diff
                        best_j = j

            if best_j is not None:
                # Strict involution: best_j must be unpaired here because of the
                # pair_ids[j] != n guard above, so no chains can form.
                pair_ids[i] = best_j
                pair_ids[best_j] = i

        self.pair_ids = pair_ids
        return self.pair_ids

    def validate_pair_ids(self) -> None:
        """
        Raises ValueError if pair_ids is not a strict symmetric involution
        (pair_ids[i] = j implies pair_ids[j] = i). Chains fail loudly.
        """
        if self.pair_ids is None:
            return
        n = len(self.pair_ids)
        for i, j in enumerate(self.pair_ids):
            if j >= n:
                continue
            if self.pair_ids[j] != i:
                raise ValueError(
                    f"Invalid pair_ids: mode {i} pairs with {j}, but mode {j} "
                    f"pairs with {self.pair_ids[j]}. "
                    "pair_ids must be a strict symmetric involution (no degenerate chains)."
                )

    def index_of_original(self, original_index: int) -> Optional[int]:
        """
        Maps a mode index from the full DFT run (original indexing) to its
        position in this spectrum. Returns None if the mode was dropped.
        """
        if self.original_indices is None:
            return original_index if 0 <= original_index < self.n_modes else None
        hits = np.where(self.original_indices == original_index)[0]
        return int(hits[0]) if len(hits) else None

    def infer_original_indices(self, raw_perturbations: dict[int, float], tol_rel: float = 0.05) -> bool:
        """
        Legacy fallback: reconstructs original_indices by matching raw perturbation
        amplitudes to phonon frequencies. Since q = q0 * sqrt(2*omega/hbar), the
        perturbation amplitude squared is proportional to the mode frequency, so a
        rank matching (sorted pert^2 vs sorted frequency) recovers the mapping.

        Only runs when the raw index set clearly exceeds this spectrum's modes
        (i.e. the phonon file dropped modes without recording which). Returns True
        if a consistent mapping was found, False otherwise.

        Assumes degenerate E pairs share a frequency; ties are matched in index order.
        """
        n_raw = max(raw_perturbations) + 1 if raw_perturbations else 0
        if n_raw <= self.n_modes:
            return False

        # Fit the proportionality constant q0^2 from rank matching
        raw_sorted = sorted(raw_perturbations.items(), key=lambda kv: kv[1])
        freq_pos = np.argsort(self.frequencies_mev)
        if len(raw_sorted) != self.n_modes:
            print(
                f"Warning: raw ZFS data covers {len(raw_sorted)} modes but spectrum has "
                f"{self.n_modes}; cannot infer original_indices by rank matching. "
                "Modes will be dropped loudly instead of silently mismatched."
            )
            return False

        # Check proportionality quality first
        p2 = np.array([p for _, p in raw_sorted]) ** 2
        f = self.frequencies_mev[freq_pos]
        valid = f > 0
        if valid.sum() < 2:
            return False
        c = np.corrcoef(p2[valid], f[valid])[0, 1]
        if c < 1 - tol_rel:
            print(
                f"Warning: perturbation-frequency correlation is only {c:.4f}; "
                "refusing to infer original_indices from a dubious match."
            )
            return False

        q0_2 = float(np.median(p2[valid] / f[valid]))
        residuals = np.abs(p2 - q0_2 * f) / (q0_2 * f + 1e-30)
        bad = int(np.sum(residuals[valid] > tol_rel))
        if bad > 0:
            print(
                f"Warning: {bad} modes deviate > {tol_rel:.0%} from the fitted "
                f"q0^2 = {q0_2:.3e}; inferred original_indices may be unreliable."
            )

        original_indices = np.full(self.n_modes, -1, dtype=int)
        # Raw indices are 1-based folders minus 1; they map in sorted-perturbation order
        # to sorted-frequency positions. Degenerate twins (same frequency) are matched
        # in raw-index order, mirroring how the perturbation folders were generated.
        for pos, (raw_idx, _) in zip(freq_pos, raw_sorted):
            original_indices[pos] = raw_idx

        self.original_indices = original_indices
        print(
            f"Inferred original_indices for legacy phonon file by amplitude-frequency "
            f"rank matching (corr={c:.5f}, q0^2={q0_2:.3e}). "
            "Regenerate the phonon npz with explicit original_indices for a principled mapping."
        )
        return True

    def get_mode(self, idx: int) -> PhononMode:
        sym = self.symmetries[idx] if self.symmetries is not None else None
        ipr_val = float(self.iprs[idx]) if self.iprs is not None else None
        pair = int(self.pair_ids[idx]) if self.pair_ids is not None else -1  # -1 = unknown (no spectrum attached)
        orig = int(self.original_indices[idx]) if self.original_indices is not None else idx
        return PhononMode(
            index=idx,
            frequency_mev=float(self.frequencies_mev[idx]),
            eigenvector=self.eigenvectors[idx],
            symmetry=sym,
            ipr=ipr_val,
            pair_id=pair,
            original_index=orig,
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

    def translate_defect_to_origin(
        self, defect_pos: Optional[np.ndarray] = None, wrap: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        frac_atoms = self.atom_frac_coords
        lattice = self.lattice
        inv_lat = np.linalg.inv(lattice)
        symbols = self.atom_symbols

        if defect_pos is None:
            if "N" in symbols and "C" in symbols:
                n_indices = [i for i, s in enumerate(symbols) if s == "N"]
                if len(n_indices) != 1:
                    raise ValueError(f"Expected exactly one N, found {len(n_indices)}")
                defect_frac = frac_atoms[n_indices[0]]
            elif "Cl" in symbols and "Si" in symbols:
                cl_indices = [i for i, s in enumerate(symbols) if s == "Cl"]
                defect_frac = frac_atoms[cl_indices].mean(axis=0)
            else:
                print("Warning: Could not identify defect centre, no shift applied.")
                return frac_atoms, np.zeros(3)
        else:
            defect_pos = np.asarray(defect_pos, dtype=float)
            defect_frac = defect_pos @ inv_lat

        shifted_frac = frac_atoms - defect_frac
        if wrap:
            shifted_frac = np.mod(shifted_frac, 1.0)

        return shifted_frac, defect_frac

    def analyze_c3v_symmetry(self) -> list[str]:
        """
        Analyzes C3v point group symmetry representations (A1, A2, Ex, Ey) for each phonon mode.
        """
        from beyblade.utils import MathUtils

        try:
            frac_atoms, _ = self.translate_defect_to_origin()
        except Exception:
            self.symmetries = ["A1"] * self.n_modes
            return self.symmetries

        symbols = np.array(self.atom_symbols)
        freqs = self.frequencies_mev
        eigs = self.eigenvectors
        lattice = self.lattice

        if "Si" in symbols and "Cl" in symbols:
            principal_axis = [0, 0, 1]
            reflection_normal = [1, 0, 0]
        elif "C" in symbols and "N" in symbols:
            principal_axis = [1, 1, 1]
            reflection_normal = [1, -1, 0]
        else:
            principal_axis = [0, 0, 1]
            reflection_normal = [1, 0, 0]

        R_C3 = MathUtils.rotation_around_symmetry_axis(principal_axis, 3)
        R_sv = MathUtils.reflection_matrix(reflection_normal)

        inv_lat = np.linalg.inv(lattice)
        num_atoms = frac_atoms.shape[0]
        num_modes = eigs.shape[0]
        cart_atoms = frac_atoms @ lattice

        def get_mapping(R):
            mapping = np.zeros(num_atoms, dtype=int)
            rotated_cart = cart_atoms @ R.T
            rot_frac = np.mod(rotated_cart @ inv_lat, 1.0)
            orig_frac = np.mod(frac_atoms, 1.0)

            for i in range(num_atoms):
                diffs = np.mod(orig_frac - rot_frac[i] + 0.5, 1.0) - 0.5
                dists = np.linalg.norm(diffs @ lattice, axis=1)
                valid_indices = np.where(symbols == symbols[i])[0]
                if len(valid_indices) == 0:
                    mapping[i] = i
                else:
                    mapping[i] = valid_indices[np.argmin(dists[valid_indices])]
            return mapping

        map_C3 = get_mapping(R_C3)
        map_sv = get_mapping(R_sv)

        results_dict = {"idx": [], "freqs": [], "sym": [], "char_C3": [], "char_sv": []}

        for m in range(num_modes):
            eig = eigs[m]
            char_C3 = np.trace(np.dot(eig[map_C3], R_C3 @ eig.T))
            char_sv = np.trace(np.dot(eig[map_sv], R_sv @ eig.T))

            if char_C3 > 0.8:
                sym = "A1" if char_sv > 0.0 else "A2"
            else:
                sym = "Ex" if char_sv > 0.0 else "Ey"

            results_dict["idx"].append(m)
            results_dict["freqs"].append(freqs[m])
            results_dict["sym"].append(sym)
            results_dict["char_C3"].append(char_C3)
            results_dict["char_sv"].append(char_sv)

        sort_indices = np.argsort(results_dict["idx"])
        output_dict = {key: np.array(value)[sort_indices] for key, value in results_dict.items()}

        self.symmetries = [str(s) for s in output_dict["sym"]]
        return self.symmetries

    def get_phonon_pert(
        self,
        perturbation_scale_si: float = 1.0,
        perturbation_scale: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        Computes mass-weighted perturbation displacements (SI), frequencies (J), symmetries, and IPRs.

        The perturbation amplitude for each mode follows the mass-weighted
        phonon coordinate  q = q0 * sqrt(2*omega/hbar). Modes with frequency <= 0 get None.
        """
        scale = perturbation_scale if perturbation_scale is not None else perturbation_scale_si
        omega_rads = self.frequencies_mev * CONSTANTS["meV2rads"]
        displacements = [
            scale * np.sqrt(2.0 * omega / Cn.hbar) if freq > 0 else None
            for freq, omega in zip(self.frequencies_mev, omega_rads)
        ]
        freqs_j = self.frequencies_to_unit("J")
        iprs = self.get_ipr()
        syms = self.symmetries if self.symmetries is not None else ["A1"] * self.n_modes

        return {
            "disp": displacements,
            "freqs": freqs_j,
            "sym": syms,
            "ipr": iprs,
            "eigs": self.eigenvectors,
        }

    def calc_ipr(self) -> np.ndarray:
        """Calculates and caches the Inverse Participation Ratio (IPR) for all phonon modes."""
        from beyblade.utils import MathUtils
        if self.iprs is None:
            self.iprs = MathUtils.calc_ipr(self.eigenvectors)
        return self.iprs

    def get_ipr(self) -> np.ndarray:
        """Returns the IPR array, computing it if not already present."""
        if self.iprs is None:
            return self.calc_ipr()
        return self.iprs

    def filter_sym_pairs(self, tol_mev: float = 0.01) -> "PhononSpectrum":
        """
        Removes redundant degenerate partner modes from Ex/Ey doublets.

        For each Ex/Ey pair closer than ``tol_mev`` in frequency, keeps the Ex
        twin and drops the Ey one. The reduced spectrum carries 0-based
        ``original_indices`` pointing into the *full* spectrum, so downstream
        code can map each reduced mode back to its source position.

        Requires symmetry labels: computes them if not already present.
        """
        syms = self.symmetries if self.symmetries is not None else self.analyze_c3v_symmetry()
        freqs = self.frequencies_mev
        n = self.n_modes

        skip_indices = set()
        for i in range(n):
            if i in skip_indices:
                continue
            if "E" in syms[i]:
                for j in range(i + 1, n):
                    if j not in skip_indices and "E" in syms[j]:
                        if abs(freqs[j] - freqs[i]) < tol_mev:
                            skip_idx = j if syms[i] == "Ex" else i
                            skip_indices.add(skip_idx)
                            break

        mask = np.array([i not in skip_indices for i in range(n)])
        return PhononSpectrum(
            frequencies_mev=freqs[mask],
            eigenvectors=self.eigenvectors[mask],
            atom_frac_coords=self.atom_frac_coords,
            atom_symbols=self.atom_symbols,
            atomic_masses=self.atomic_masses,
            lattice=self.lattice,
            symmetries=[s for k, s in enumerate(syms) if mask[k]],
            iprs=self.iprs[mask] if self.iprs is not None else None,
            # 0-based indices into the ORIGINAL (full) spectrum, so downstream
            # code can map each reduced mode back to its source position.
            original_indices=np.where(mask)[0],
        )

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
            e_pair_complete=self.e_pair_complete,
            pair_ids=self.pair_ids,
            original_indices=self.original_indices,
            # Legacy aliases consumed by the supercomputer VASP scripts:
            #   scripts/get_n_modes.py reads 'idx' (0-based, printed +1 as
            #   VASP folder numbers) and 'freqs';
            #   scripts/create_combined_phonon_struct.py reads 'eigs' and
            #   'masses'. Keep these keys in sync with the modern ones.
            eigs=self.eigenvectors,
            masses=self.atomic_masses,
            idx=self.original_indices if self.original_indices is not None else np.arange(self.n_modes),
            freqs=self.frequencies_mev,
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
        e_pair_complete = list(bool(x) for x in data["e_pair_complete"]) if "e_pair_complete" in data and data["e_pair_complete"] is not None else None
        pair_ids = np.asarray(data["pair_ids"], dtype=int) if "pair_ids" in data and data["pair_ids"] is not None else None
        original_indices = np.asarray(data["original_indices"], dtype=int) if "original_indices" in data and data["original_indices"] is not None else None

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
            e_pair_complete=e_pair_complete,
            pair_ids=pair_ids,
            original_indices=original_indices,
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

    @property
    def has_second_order(self) -> bool:
        """Returns True if second-order perturbation data is present and non-empty."""
        return len(self.second_order) > 0

    @property
    def has_first_order(self) -> bool:
        """Returns True if first-order perturbation data is present and non-empty."""
        return len(self.first_order) > 0

    @property
    def combined_order(self) -> int:
        """Effective order: 2 if second-order data is present, else order or 1."""
        return 2 if self.has_second_order else (self.order or 1)

    def combine(self, other: RawZFSData) -> RawZFSData:
        """
        Combines two RawZFSData containers (e.g. 1D and 2D datasets).

        Verifies that metadata (defect, cell_size, pert_scale, calc_method) matches.
        Raises ValueError if there is any metadata mismatch.
        """
        if not isinstance(other, RawZFSData):
            raise TypeError(f"Cannot combine RawZFSData with object of type {type(other)}")

        # 1. Verify defect
        d1 = self.defect if self.defect not in (None, "None", "") else None
        d2 = other.defect if other.defect not in (None, "None", "") else None
        if d1 is not None and d2 is not None and d1 != d2:
            raise ValueError(f"Cannot combine RawZFSData: defect mismatch ('{d1}' != '{d2}').")

        # 2. Verify cell_size
        c1 = self.cell_size if self.cell_size not in (None, 0) else None
        c2 = other.cell_size if other.cell_size not in (None, 0) else None
        if c1 is not None and c2 is not None and c1 != c2:
            raise ValueError(f"Cannot combine RawZFSData: cell_size mismatch ({c1} != {c2}).")

        # 3. Verify pert_scale
        p1 = self.pert_scale if self.pert_scale not in (None, 0.0) else None
        p2 = other.pert_scale if other.pert_scale not in (None, 0.0) else None
        if p1 is not None and p2 is not None and not np.isclose(p1, p2, rtol=1e-3, atol=1e-5):
            raise ValueError(f"Cannot combine RawZFSData: pert_scale mismatch ({p1} != {p2}).")

        # 4. Verify calc_method
        def _norm_method(m: Optional[str]) -> Optional[str]:
            if not m or m in ("None", ""):
                return None
            if m in ("all", "all_bands"):
                return "all_bands"
            if m in ("approx", "defect_band_approx"):
                return "defect_band_approx"
            return str(m)

        m1 = _norm_method(self.calc_method)
        m2 = _norm_method(other.calc_method)
        if m1 is not None and m2 is not None and m1 != m2:
            raise ValueError(f"Cannot combine RawZFSData: calc_method mismatch ('{m1}' != '{m2}').")

        # 5. Verify common metadata dictionary keys
        if self.metadata and other.metadata:
            for k in set(self.metadata.keys()) & set(other.metadata.keys()):
                if k in ("sim_path", "source_file", "source_files", "order", "calc_method"):
                    continue
                v1, v2 = self.metadata[k], other.metadata[k]
                if v1 is not None and v2 is not None and v1 != v2:
                    raise ValueError(f"Cannot combine RawZFSData: metadata mismatch for '{k}' ({v1} != {v2}).")

        # 6. Verify ground state ZFS if both have it
        gs = self.ground_state_zfs
        if gs is None and other.ground_state_zfs is not None:
            gs = other.ground_state_zfs
        elif gs is not None and other.ground_state_zfs is not None:
            gs_self_mhz = gs.to_unit("MHz")
            gs_other_mhz = other.ground_state_zfs.to_unit("MHz")
            if not np.allclose(gs_self_mhz.matrix, gs_other_mhz.matrix, rtol=1e-2, atol=1e-1):
                raise ValueError("Cannot combine RawZFSData: ground_state_zfs tensor mismatch.")

        # 7. Merge perturbation entries
        merged_first = dict(self.first_order)
        merged_first.update(other.first_order)

        merged_second = dict(self.second_order)
        merged_second.update(other.second_order)

        eff_order = 2 if len(merged_second) > 0 else (self.order or other.order or 1)
        merged_defect = d1 or d2 or "defect"
        merged_cell = c1 or c2 or 0
        merged_scale = p1 or p2 or 0.0
        merged_method = m1 or m2 or self.calc_method or other.calc_method
        merged_rotation = self.eigen_rotation if self.eigen_rotation is not None else other.eigen_rotation

        merged_metadata = dict(self.metadata)
        for k, v in other.metadata.items():
            if k == "sim_path":
                p_existing = merged_metadata.get("sim_path")
                if p_existing:
                    if isinstance(p_existing, list):
                        merged_metadata["sim_path"] = p_existing + [v]
                    else:
                        merged_metadata["sim_path"] = [p_existing, v]
                else:
                    merged_metadata["sim_path"] = v
            elif k not in merged_metadata or merged_metadata[k] is None:
                merged_metadata[k] = v

        return RawZFSData(
            defect=merged_defect,
            cell_size=merged_cell,
            pert_scale=merged_scale,
            calc_method=merged_method,
            order=eff_order,
            ground_state_zfs=gs,
            eigen_rotation=merged_rotation,
            first_order=merged_first,
            second_order=merged_second,
            metadata=merged_metadata,
        )

    def __add__(self, other: RawZFSData) -> RawZFSData:
        return self.combine(other)

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

    def enrich_with_spectrum(self, spectrum: PhononSpectrum) -> None:
        """
        Enriches first_order and second_order entries with mode-dependent SI displacements,
        C3v symmetries, and IPRs from the given phonon spectrum.
        """
        pert_SI = (self.pert_scale or 1.0) * CONSTANTS["ang_amu2SI"]
        phonon_pert = spectrum.get_phonon_pert(pert_SI)
        disps = phonon_pert.get("disp", [])
        syms = phonon_pert.get("sym", [])
        iprs = phonon_pert.get("ipr", [])

        for idx, entry in list(self.first_order.items()):
            disp = disps[idx] if idx < len(disps) else None
            sym = syms[idx] if idx < len(syms) else None
            ipr = iprs[idx] if idx < len(iprs) else None
            if isinstance(entry, PerturbationEntry):
                if disp is not None:
                    entry.amplitude = disp
            elif isinstance(entry, dict):
                if disp is not None:
                    entry["pert"] = disp
                if sym is not None:
                    entry["symmetry"] = sym
                if ipr is not None:
                    entry["ipr"] = ipr

        for (i, j), entry in list(self.second_order.items()):
            disp_i = disps[i] if i < len(disps) else None
            disp_j = disps[j] if j < len(disps) else None
            sym_i = syms[i] if i < len(syms) else None
            sym_j = syms[j] if j < len(syms) else None
            ipr_i = iprs[i] if i < len(iprs) else None
            ipr_j = iprs[j] if j < len(iprs) else None
            pair_disp = (disp_i, disp_j) if (disp_i is not None and disp_j is not None) else None
            if isinstance(entry, PerturbationEntry):
                if pair_disp is not None:
                    entry.amplitude = pair_disp
            elif isinstance(entry, dict):
                if pair_disp is not None:
                    entry["pert"] = pair_disp
                if sym_i is not None and sym_j is not None:
                    entry["symmetry"] = (sym_i, sym_j)
                if ipr_i is not None and ipr_j is not None:
                    entry["ipr"] = (ipr_i, ipr_j)

    def _default_name(self):
        return f"{self.defect}_{self.cell_size}_raw_zfs_data_{self.calc_method}_{self.order}d.npz"

    def save(
        self,
        out_path: Optional[Union[str, Path]] = None,
        spectrum: Optional[PhononSpectrum] = None,
    ) -> str:
        """Saves RawZFSData to a .npz file with latest naming conventions and explicit unit metadata."""
        if spectrum is not None:
            self.enrich_with_spectrum(spectrum)

        path = self._default_name() if out_path is None else str(out_path)
        if not path.endswith(".npz"):
            path += ".npz"
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        gs_mat = self.ground_state_zfs.matrix if self.ground_state_zfs else None
        gs_unit = self.ground_state_zfs.unit if self.ground_state_zfs else "MHz"

        # Format dictionaries into clean serializable entries
        saved_1d = {}
        for idx, entry in self.first_order.items():
            if isinstance(entry, PerturbationEntry):
                amp = entry.amplitude
                # Convert amplitude from Å*sqrt(amu) to SI units if it has not yet been converted/enriched
                if isinstance(amp, (int, float)) and amp > 1e-4:
                    amp = amp * CONSTANTS["ang_amu2SI"]
                saved_1d[idx] = {
                    "tensor": entry.zfs_tensor.matrix,
                    "unit": entry.zfs_tensor.unit,
                    "pert": amp,
                }
            elif isinstance(entry, dict):
                saved_1d[idx] = entry

        saved_2d = {}
        for idx, entry in self.second_order.items():
            # For 2D keys, tuple (i, j) can be stored as "i_j" string for numpy compatibility
            key = f"{idx[0]}_{idx[1]}" if isinstance(idx, tuple) else str(idx)
            if isinstance(entry, PerturbationEntry):
                amp = entry.amplitude
                if isinstance(amp, (tuple, list)):
                    amp = tuple(a * CONSTANTS["ang_amu2SI"] if (isinstance(a, (int, float)) and a > 1e-4) else a for a in amp)
                elif isinstance(amp, (int, float)) and amp > 1e-4:
                    amp = (amp * CONSTANTS["ang_amu2SI"], amp * CONSTANTS["ang_amu2SI"])
                saved_2d[key] = {
                    "tensor": entry.zfs_tensor.matrix,
                    "unit": entry.zfs_tensor.unit,
                    "pert": amp,
                }
            elif isinstance(entry, dict):
                saved_2d[key] = entry

        eff_order = self.order
        if eff_order is None or eff_order == 1:
            if self.second_order:
                eff_order = 2
            elif self.first_order:
                eff_order = 1

        np.savez(
            path,
            order=eff_order,
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
        calc_method = None
        if "calc_method" in base and base["calc_method"] is not None:
            cm = str(base["calc_method"])
            if cm != "None" and cm != "":
                calc_method = cm

        order = None
        if "order" in base and base["order"] is not None:
            try:
                ord_str = str(base["order"])
                if ord_str != "None" and ord_str != "":
                    order = int(ord_str)
            except (ValueError, TypeError):
                order = None
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

        # Determine effective order from loaded perturbation data
        if second_order:
            order = 2
        elif order is None:
            order = 1 if first_order else None

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
        ).to_unit("J")


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
    # Second-order (2D) coupling coefficients, optional — used when the run
    # combines first- and second-order data (1d + 2d perturbation sets).
    V2_0_0: Optional[np.ndarray] = None
    V2_p_m: Optional[np.ndarray] = None
    V2_0_pm: Optional[np.ndarray] = None
    zfs_2nd_derivs: Optional[np.ndarray] = None

    @property
    def has_second_order(self) -> bool:
        """Returns True if valid 2nd-order coupling data is present (non-None and non-empty)."""
        if self.V2_0_0 is None:
            return False
        arr = np.asarray(self.V2_0_0)
        return arr.ndim >= 2 and arr.size > 0

    @property
    def combined_order(self) -> int:
        """Effective order for combined runs: 2 when 2d data present, else order."""
        return 2 if self.has_second_order else self.order

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
            # Check for numpy 0-d object array holding None
            if not (getattr(self.zfs_derivs, "shape", None) == () and self.zfs_derivs.item() is None):
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
            V2_0_0=convert_energy(self.V2_0_0, self.coupling_unit, target_unit) if self.V2_0_0 is not None else None,
            V2_p_m=convert_energy(self.V2_p_m, self.coupling_unit, target_unit) if self.V2_p_m is not None else None,
            V2_0_pm=convert_energy(self.V2_0_pm, self.coupling_unit, target_unit) if self.V2_0_pm is not None else None,
            zfs_2nd_derivs=convert_energy(self.zfs_2nd_derivs, self.derivs_unit, target_unit) if self.zfs_2nd_derivs is not None else None,
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
            V2_0_0=self.V2_0_0.copy() if self.V2_0_0 is not None else None,
            V2_p_m=self.V2_p_m.copy() if self.V2_p_m is not None else None,
            V2_0_pm=self.V2_0_pm.copy() if self.V2_0_pm is not None else None,
            zfs_2nd_derivs=self.zfs_2nd_derivs.copy() if self.zfs_2nd_derivs is not None else None,
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
            # Coupling coefficients (first order)
            V_0_0=self.V_0_0,
            V_p_m=self.V_p_m,
            V_0_pm=self.V_0_pm,
            coupling_unit=self.coupling_unit,
            # Coupling coefficients (second order, when present)
            V2_0_0=self.V2_0_0,
            V2_p_m=self.V2_p_m,
            V2_0_pm=self.V2_0_pm,
            zfs_2nd_derivs=self.zfs_2nd_derivs,
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

        derivs = data["zfs_derivs"] if "zfs_derivs" in data else None
        if derivs is not None and (getattr(derivs, "shape", None) == () and derivs.item() is None):
            derivs = None

        # Second-order coupling coefficients (optional)
        def _arr(key):
            if key not in data:
                return None
            val = data[key]
            if val is None or (getattr(val, "shape", None) == () and val.item() is None):
                return None
            arr = np.asarray(val)
            return arr if arr.ndim >= 2 and arr.size > 0 else None

        V2_0_0 = _arr("V2_0_0")
        V2_p_m = _arr("V2_p_m")
        V2_0_pm = _arr("V2_0_pm")
        zfs_2nd = _arr("zfs_2nd_derivs")

        # Backward compat: legacy 2d files saved their coefficients under the
        # first-order keys with order=2 (no V2_* present). Detect and remap.
        if V2_0_0 is None and "V2_0_0" not in data and int(data.get("order", 1)) == 2:
            V2_0_0 = V_0_0
            V2_p_m = V_p_m
            V2_0_pm = V_0_pm
            zfs_2nd = zfs_2nd or derivs

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

        derivs_unit = str(data.get("derivs_unit", coupling_unit))

        sym_arr = data["symmetries"] if "symmetries" in data else (data["sym"] if "sym" in data else None)
        syms = list(sym_arr) if sym_arr is not None and getattr(sym_arr, "shape", None) != () else None

        ipr_arr = data["iprs"] if "iprs" in data else (data["ipr"] if "ipr" in data else None)
        iprs = ipr_arr if ipr_arr is not None and getattr(ipr_arr, "shape", None) != () else None

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
            V2_0_0=V2_0_0,
            V2_p_m=V2_p_m,
            V2_0_pm=V2_0_pm,
            zfs_2nd_derivs=zfs_2nd,
        )
