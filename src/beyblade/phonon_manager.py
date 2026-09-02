from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
from scipy import constants as Cn

from beyblade.constants import CONSTANTS
from beyblade.models import PhononSpectrum
from beyblade.parsers import parse_phonopy_yaml, parse_phonon_npz, save_phonon_npz
from beyblade.utils import MathUtils


class PhononManager:
    """
    Manages phonon spectra, C3v symmetry classification, defect recentering, and IPR calculations.
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        spectrum: Optional[PhononSpectrum] = None,
    ):
        self.spectrum: Optional[PhononSpectrum] = spectrum
        self.symmetry_data: Optional[dict[str, np.ndarray]] = None
        self._defect_shift: Optional[np.ndarray] = None

        if data_path is not None:
            self.load_data(data_path)

    @property
    def nmodes(self) -> int:
        return self.spectrum.n_modes if self.spectrum else 0

    @property
    def cell_size(self) -> int:
        return self.spectrum.n_atoms if self.spectrum else 0

    @property
    def data(self) -> dict[str, Any]:
        """Provides backward-compatible dict access to underlying spectrum."""
        if self.spectrum is None:
            return {}
        return {
            "freqs": self.spectrum.frequencies_mev,
            "eigs": self.spectrum.eigenvectors,
            "atoms": self.spectrum.atom_frac_coords,
            "atom_symbols": np.array(self.spectrum.atom_symbols),
            "masses": self.spectrum.atomic_masses,
            "lattice": self.spectrum.lattice,
            "n_atoms": self.spectrum.n_atoms,
            "n_modes": self.spectrum.n_modes,
            "sym": np.array(self.spectrum.symmetries) if self.spectrum.symmetries else None,
            "ipr": self.spectrum.iprs,
            "idx": np.arange(self.spectrum.n_modes),
        }

    def load_data(self, filepath: Union[str, Path], poscar_path: Optional[Union[str, Path]] = "POSCAR"):
        """Loads phonon data from either a phonopy.yaml or an .npz archive."""
        path = Path(filepath)
        if path.suffix == ".yaml":
            self.spectrum = parse_phonopy_yaml(path, poscar_path=poscar_path if Path(str(poscar_path)).is_file() else None)
        elif path.suffix == ".npz":
            self.spectrum = parse_phonon_npz(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}. Use .yaml or .npz")

        if self.spectrum.symmetries is None:
            self.analyze_c3v_symmetry()
        else:
            self.symmetry_data = {
                "sym": np.array(self.spectrum.symmetries),
                "freqs": self.spectrum.frequencies_mev,
                "idx": np.arange(self.spectrum.n_modes),
            }

        print(f"Loaded phonon data from {path.name}: {self.nmodes} modes, {self.cell_size} atoms.")

    def save_data(self, filename: Union[str, Path] = "phonon_data.npz"):
        if self.spectrum is not None:
            save_phonon_npz(self.spectrum, filename)
            print(f"Saved phonon data to: {filename}")

    def get_freqs(self) -> np.ndarray:
        return self.spectrum.frequencies_mev if self.spectrum else np.array([])

    def get_phonon_pert(self, perturbation_scale_si: float) -> dict[str, Any]:
        if self.spectrum is None:
            raise ValueError("No phonon data loaded.")
        return self.spectrum.get_phonon_pert(perturbation_scale_si)

    def calc_ipr(self) -> np.ndarray:
        """
        Calculates and caches the Inverse Participation Ratio (IPR) for all phonon modes.
        """
        if self.spectrum is None:
            return np.array([])
        iprs = MathUtils.calc_ipr(self.spectrum.eigenvectors)
        self.spectrum.iprs = iprs
        return iprs

    def get_ipr(self) -> np.ndarray:
        if self.spectrum is None:
            return np.array([])
        if self.spectrum.iprs is None:
            return self.calc_ipr()
        return self.spectrum.iprs

    def translate_defect_to_origin(self, defect_pos: Optional[np.ndarray] = None, wrap: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """
        Translates all atomic positions so that the defect position is at the origin.
        """
        if self.spectrum is None:
            raise ValueError("No phonon data loaded.")

        frac_atoms = self.spectrum.atom_frac_coords
        lattice = self.spectrum.lattice
        inv_lat = np.linalg.inv(lattice)
        symbols = self.spectrum.atom_symbols

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

        self._defect_shift = defect_frac.copy()
        return shifted_frac, defect_frac

    def analyze_c3v_symmetry(self):
        """
        Analyzes C3v point group symmetry representations (A1, A2, Ex, Ey) for each phonon mode.
        """
        if self.spectrum is None:
            raise ValueError("No phonon data loaded.")

        frac_atoms, _ = self.translate_defect_to_origin()
        symbols = np.array(self.spectrum.atom_symbols)
        freqs = self.spectrum.frequencies_mev
        eigs = self.spectrum.eigenvectors
        lattice = self.spectrum.lattice

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

        self.symmetry_data = output_dict
        self.spectrum.symmetries = list(output_dict["sym"])

    def filter_sym_pairs(self, save: bool = True, debug: bool = False, tol: float = 0.01) -> PhononSpectrum:
        """
        Removes redundant degenerate partner modes from Ex/Ey doublets.
        """
        if self.symmetry_data is None:
            self.analyze_c3v_symmetry()

        skip_indices = set()
        for i in range(self.nmodes):
            if i in skip_indices:
                continue

            if "E" in self.symmetry_data["sym"][i]:
                for j in range(i + 1, self.nmodes):
                    if j not in skip_indices and "E" in self.symmetry_data["sym"][j]:
                        if abs(self.symmetry_data["freqs"][j] - self.symmetry_data["freqs"][i]) < tol:
                            skip_idx = j if self.symmetry_data["sym"][i] == "Ex" else i
                            skip_indices.add(skip_idx)
                            break

        mask = [i not in skip_indices for i in range(self.nmodes)]
        new_spectrum = PhononSpectrum(
            frequencies_mev=self.spectrum.frequencies_mev[mask],
            eigenvectors=self.spectrum.eigenvectors[mask],
            atom_frac_coords=self.spectrum.atom_frac_coords,
            atom_symbols=self.spectrum.atom_symbols,
            atomic_masses=self.spectrum.atomic_masses,
            lattice=self.spectrum.lattice,
            symmetries=[s for k, s in enumerate(self.spectrum.symmetries) if mask[k]] if self.spectrum.symmetries else None,
            iprs=self.spectrum.iprs[mask] if self.spectrum.iprs is not None else None,
        )

        if not save:
            self.spectrum = new_spectrum
            self.analyze_c3v_symmetry()
        else:
            filename = f"phonon_data_sym_n{new_spectrum.n_modes}.npz"
            save_phonon_npz(new_spectrum, filename)

        return new_spectrum

    def get_phonon_pert(self, perturbation_scale: float) -> dict[str, Any]:
        if self.spectrum is None:
            raise ValueError("No phonon data loaded.")
        if self.symmetry_data is None:
            self.analyze_c3v_symmetry()

        Q = [np.sqrt(np.sum(mode**2)) for mode in self.spectrum.eigenvectors]
        if not np.allclose(Q, 1.0):
            raise ValueError("Phonon modes not normalized correctly.")

        eigs_pert = np.array([
            perturbation_scale * np.sqrt(2 * CONSTANTS["meV2rads"] * freq / Cn.hbar) if freq > 0 else None
            for freq in self.spectrum.frequencies_mev
        ])

        return {
            "sym": self.symmetry_data["sym"],
            "idx": self.symmetry_data["idx"],
            "eigs": eigs_pert,
            "freqs": self.spectrum.frequencies_mev * CONSTANTS["meV2J"],
            "ipr": self.get_ipr(),
        }
