from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

from beyblade.models import PhononSpectrum
from beyblade.parsers import parse_phonopy_yaml, parse_phonon_npz, save_phonon_npz
from beyblade.utils import MathUtils


class PhononManager:
    """
    Manages phonon spectra, C3v symmetry classification, defect recentering, and IPR calculations.

    .. deprecated::
        PhononManager is legacy. Use PhononSpectrum and functions in beyblade.parsers instead.
    """

    def __init__(
        self,
        data_path: Optional[Union[str, Path]] = None,
        spectrum: Optional[PhononSpectrum] = None,
    ):
        warnings.warn(
            "PhononManager is deprecated and will be removed in a future release. "
            "Please use PhononSpectrum and parsers (parse_phonon_npz, parse_phonopy_yaml) directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
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

    def get_phonon_pert(self, perturbation_scale_si: float = 1.0, perturbation_scale: Optional[float] = None) -> dict[str, Any]:
        if self.spectrum is None:
            raise ValueError("No phonon data loaded.")
        return self.spectrum.get_phonon_pert(
            perturbation_scale_si=perturbation_scale_si,
            perturbation_scale=perturbation_scale,
        )

    def calc_ipr(self) -> np.ndarray:
        """
        Calculates and caches the Inverse Participation Ratio (IPR) for all phonon modes.
        """
        if self.spectrum is None:
            return np.array([])
        if self.spectrum.iprs is None:
            self.spectrum.iprs = MathUtils.calc_ipr(self.spectrum.eigenvectors)
        return self.spectrum.iprs

    def get_ipr(self) -> np.ndarray:
        if self.spectrum is None:
            return np.array([])
        if self.spectrum.iprs is None:
            return self.calc_ipr()
        return self.spectrum.iprs

    def translate_defect_to_origin(
        self, defect_pos: Optional[np.ndarray] = None, wrap: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Translates all atomic positions so that the defect position is at the origin.
        """
        if self.spectrum is None:
            raise ValueError("No phonon data loaded.")

        shifted_frac, defect_frac = self.spectrum.translate_defect_to_origin(defect_pos=defect_pos, wrap=wrap)
        self._defect_shift = defect_frac.copy()
        return shifted_frac, defect_frac

    def analyze_c3v_symmetry(self):
        """
        Analyzes C3v point group symmetry representations (A1, A2, Ex, Ey) for each phonon mode.
        """
        if self.spectrum is None:
            raise ValueError("No phonon data loaded.")

        syms = self.spectrum.analyze_c3v_symmetry()
        self.symmetry_data = {
            "idx": np.arange(self.spectrum.n_modes),
            "freqs": self.spectrum.frequencies_mev,
            "sym": np.array(syms),
        }
        return syms

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

        mask = np.array([i not in skip_indices for i in range(self.nmodes)])
        new_spectrum = PhononSpectrum(
            frequencies_mev=self.spectrum.frequencies_mev[mask],
            eigenvectors=self.spectrum.eigenvectors[mask],
            atom_frac_coords=self.spectrum.atom_frac_coords,
            atom_symbols=self.spectrum.atom_symbols,
            atomic_masses=self.spectrum.atomic_masses,
            lattice=self.spectrum.lattice,
            symmetries=[s for k, s in enumerate(self.spectrum.symmetries) if mask[k]] if self.spectrum.symmetries else None,
            iprs=self.spectrum.iprs[mask] if self.spectrum.iprs is not None else None,
            # 0-based indices into the ORIGINAL (full) spectrum, so downstream
            # code can map each reduced mode back to its source position.
            original_indices=np.where(mask)[0],
        )

        if not save:
            self.spectrum = new_spectrum
            self.analyze_c3v_symmetry()
        else:
            filename = f"phonon_data_sym_n{new_spectrum.n_modes}.npz"
            saved_path = save_phonon_npz(new_spectrum, filename)
            # Round-trip validation: reload and verify original_indices maps
            # every reduced mode back to the correct full-spectrum entry.
            reloaded = parse_phonon_npz(saved_path)
            assert reloaded.original_indices is not None, "Saved sym file lacks original_indices"
            full_freqs = self.spectrum.frequencies_mev
            mapped = full_freqs[np.asarray(reloaded.original_indices)]
            if not np.allclose(mapped, reloaded.frequencies_mev, rtol=0, atol=1e-9):
                raise AssertionError(
                    "Sym-file round-trip failed: original_indices do not map "
                    "reduced frequencies back to the full spectrum."
                )
            if self.spectrum.symmetries is not None:
                mapped_syms = [self.spectrum.symmetries[k] for k in reloaded.original_indices]
                if mapped_syms != list(reloaded.symmetries):
                    raise AssertionError(
                        "Sym-file round-trip failed: symmetry labels disagree "
                        "between the reduced file and the full spectrum."
                    )
            print(f"Round-trip validation passed for {filename}")

        return new_spectrum
