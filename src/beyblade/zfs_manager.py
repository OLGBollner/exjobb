from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union
import numpy as np

from beyblade.constants import CONSTANTS
from beyblade.models import ZFSTensor, PhononSpectrum, RawZFSData
from beyblade.parsers import (
    parse_zfs_simulation_dataset,
    parse_zfs_dataset_npz,
)
from beyblade.phonon_manager import PhononManager
from beyblade.utils import MathUtils


class ZFSManager:
    """
    Manages Zero-Field Splitting (ZFS) tensor calculations, coordinate transformations,
    finite-difference derivatives (1st and 2nd order), and spin-phonon coupling coefficients (V-tensors).
    Operates directly on the PhononSpectrum dataclass.
    """

    def __init__(
        self,
        spectrum: Optional[PhononSpectrum] = None,
        raw_data: Optional[RawZFSData] = None,
        phonon_manager: Optional[Any] = None,
        debug: bool = False,
    ):
        # spectrum is the primary dataclass; phonon_manager is accepted for legacy compatibility
        if spectrum is not None:
            self.spectrum = spectrum
        elif phonon_manager is not None:
            self.spectrum = getattr(phonon_manager, "spectrum", None)
        else:
            self.spectrum = None

        self.phonon_manager = phonon_manager
        self.raw_data = raw_data

        # Defect metadata
        self.defect: Optional[str] = raw_data.defect if raw_data else None
        self.cell_size: Optional[int] = raw_data.cell_size if raw_data else None
        self.pert_scale: Optional[float] = raw_data.pert_scale if raw_data else None
        self.calc_method: Optional[str] = raw_data.metadata.get("calc_method") if raw_data else None

        # Processed ZFS data in defect principal frame
        self.zfs_relaxed: Optional[np.ndarray] = None          # Shape (3, 3) in J
        self.eigen_rotation: Optional[np.ndarray] = None        # Shape (3, 3)
        self.zfs_tensors: dict[int, dict[str, Any]] = {}        # 1D perturbations
        self.zfs_tensors_2d: dict[tuple[int, int], dict[str, Any]] = {}  # 2D perturbations
        self.treated_modes: set[int] = set()

        self.debug = debug

        if raw_data is not None:
            self._ingest_raw_data(raw_data)

    @property
    def nmodes(self) -> int:
        if self.spectrum is not None:
            return self.spectrum.n_modes
        if self.phonon_manager is not None:
            return self.phonon_manager.nmodes
        return 0

    def get_phonon_frequencies(self) -> np.ndarray:
        if self.spectrum is not None:
            return self.spectrum.frequencies_mev
        if self.phonon_manager is not None:
            return self.phonon_manager.get_freqs()
        return np.array([])

    def get_phonon_pert(self, pert_scale_si: float) -> Optional[dict[str, Any]]:
        if self.spectrum is not None:
            return self.spectrum.get_phonon_pert(pert_scale_si)
        if self.phonon_manager is not None:
            return self.phonon_manager.get_phonon_pert(pert_scale_si)
        return None

    def _ingest_raw_data(self, raw: RawZFSData):
        """Populates internal arrays and applies principal frame rotation from RawZFSData."""
        self.defect = raw.defect
        self.cell_size = raw.cell_size
        self.pert_scale = raw.pert_scale
        self.calc_method = raw.metadata.get("calc_method", "calc")

        if raw.ground_state_zfs is not None:
            D_xx, D_yy, D_zz, R = raw.ground_state_zfs.principal_components()
            self.eigen_rotation = R
            self.zfs_relaxed = np.diag([D_xx, D_yy, D_zz]) * CONSTANTS["MHz2J"]
            if "approx" in self.calc_method:
                self.zfs_relaxed *= 1.5

        pert_SI = self.pert_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.get_phonon_pert(pert_SI)
        eigen_rot_t = self.eigen_rotation.T if self.eigen_rotation is not None else np.eye(3)

        if raw.first_order:
            self.zfs_tensors = {}
            for idx, entry in raw.first_order.items():
                rotated = self.eigen_rotation @ entry.zfs_tensor.matrix @ eigen_rot_t if self.eigen_rotation is not None else entry.zfs_tensor.matrix
                tensor_j = rotated * CONSTANTS["MHz2J"]
                if "approx" in self.calc_method:
                    tensor_j *= 1.5
                self.zfs_tensors[idx] = {
                    "tensor": tensor_j,
                    "symmetry": phonon_pert["sym"][idx] if phonon_pert else None,
                    "pert": phonon_pert["eigs"][idx] if phonon_pert else None,
                    "ipr": phonon_pert["ipr"][idx] if phonon_pert else None,
                }

        if raw.second_order:
            self.zfs_tensors_2d = {}
            for (i, j), entry in raw.second_order.items():
                rotated = self.eigen_rotation @ entry.zfs_tensor.matrix @ eigen_rot_t if self.eigen_rotation is not None else entry.zfs_tensor.matrix
                tensor_j = rotated * CONSTANTS["MHz2J"]
                if "approx" in self.calc_method:
                    tensor_j *= 1.5
                self.zfs_tensors_2d[(i, j)] = {
                    "tensor": tensor_j,
                    "symmetry": (phonon_pert["sym"][i], phonon_pert["sym"][j]) if phonon_pert else None,
                    "pert": (phonon_pert["eigs"][i], phonon_pert["eigs"][j]) if phonon_pert else None,
                    "ipr": (phonon_pert["ipr"][i], phonon_pert["ipr"][j]) if phonon_pert else None,
                }

        self.treated_modes = self._get_symmetry_factor()

    def load_outcar_zfs_data(self, **kwargs):
        """
        Loads ZFS data by delegating directly to parsers.
        """
        if kwargs.get("sim_folder") is not None and kwargs.get("calc_method") is not None:
            sim_folder = kwargs["sim_folder"]
            zfs_folder = kwargs.get("zfs_folder", "relaxed")
            calc_method = kwargs["calc_method"]

            raw_data = parse_zfs_simulation_dataset(sim_folder, zfs_folder=zfs_folder, calc_method=calc_method)
            self._ingest_raw_data(raw_data)
            return self.zfs_relaxed, self.zfs_tensors, self.zfs_tensors_2d, self.eigen_rotation

        elif kwargs.get("raw_data_path") is not None:
            raw_paths = kwargs["raw_data_path"]
            raw_obj, legacy_dict = parse_zfs_dataset_npz(raw_paths)

            self.defect = legacy_dict["defect"]
            self.cell_size = legacy_dict["cell_size"]
            self.pert_scale = legacy_dict["pert_scale"]
            self.calc_method = legacy_dict["calc_method"]
            self.zfs_relaxed = legacy_dict["zfs_relaxed"]
            self.eigen_rotation = legacy_dict["eigen_rotation"]
            self.zfs_tensors = legacy_dict["zfs_tensors"]
            self.zfs_tensors_2d = legacy_dict["zfs_tensors_2d"]
            self.treated_modes = self._get_symmetry_factor()

            print("Successfully loaded ZFS data via parser from .npz files")
            return self.zfs_relaxed, self.zfs_tensors, self.zfs_tensors_2d, self.eigen_rotation
        else:
            print("Initialized empty ZFSManager.")

    def _get_symmetry_factor(self) -> set[int]:
        total_modes = self.nmodes
        n_1d = len(self.zfs_tensors)
        n_2d = len(self.zfs_tensors_2d)
        print(f"Tensor data size: {n_1d} 1D, {n_2d} 2D out of {total_modes} total modes")

        treated = self.zfs_tensors.keys() if (0 < n_1d <= n_2d or n_2d == 0) else self.zfs_tensors_2d.keys()
        mode_set = {x for item in treated for x in (item if isinstance(item, tuple) else (item,))}
        print(f"Treated unique modes: {len(mode_set)}")
        return mode_set

    def calculate_first_order_derivatives(self, ipr_thresh: Optional[float] = None):
        """
        Calculates 1st order finite difference derivatives dD/dq and spin-phonon coupling coefficients.
        """
        print("Calculating first-order derivatives...")
        n_modes = self.nmodes
        phonon_energies = self.get_phonon_frequencies()

        zfs_deriv = np.zeros((n_modes, 3, 3))
        V_0_0 = np.zeros(n_modes)
        V_0_pm = np.zeros(n_modes)
        V_p_m = np.zeros(n_modes)

        # Symmetry axis mapping based on defect and cell size
        if self.defect == "NV" and self.cell_size == 512:
            sym_x, sym_y = "Ey", "Ex"
        elif self.defect in ("NV", "ClV") or (self.defect == "NV" and self.cell_size == 64):
            sym_x, sym_y = "Ex", "Ey"
        else:
            sym_x, sym_y = "Ex", "Ey"

        for i, item in sorted(self.zfs_tensors.items()):
            if i not in self.treated_modes:
                continue

            if ipr_thresh is not None and item.get("ipr", 0) < ipr_thresh:
                continue

            D_i = item["tensor"]
            q = item["pert"]
            if q is None:
                continue
            sym = item["symmetry"]

            dD = D_i - self.zfs_relaxed
            self._check_symmetry(dD, sym, i)

            dD_dq = dD / q
            zfs_deriv[i] = dD_dq

            if sym == "A1":
                V_0_0[i] = np.abs(dD_dq[2, 2]) / 2.0
            elif sym == sym_y:
                V_p_m[i] = np.abs(np.mean([dD_dq[0, 1], dD_dq[1, 0]]))
                V_0_pm[i] = np.abs(np.mean([dD_dq[1, 2], dD_dq[2, 1]])) / np.sqrt(2)
            elif sym == sym_x:
                V_p_m[i] = np.abs(0.5 * (dD_dq[1, 1] - dD_dq[0, 0]))
                V_0_pm[i] = np.abs(np.mean([dD_dq[2, 0], dD_dq[0, 2]])) / np.sqrt(2)

            # Degenerate mode handling
            if len(self.treated_modes) < n_modes and len(phonon_energies) == n_modes:
                if i + 1 < n_modes and np.isclose(phonon_energies[i], phonon_energies[i + 1]):
                    V_0_pm[i + 1] = V_0_pm[i]
                    V_p_m[i + 1] = V_p_m[i]
                elif i > 0 and np.isclose(phonon_energies[i], phonon_energies[i - 1]):
                    V_0_pm[i - 1] = V_0_pm[i]
                    V_p_m[i - 1] = V_p_m[i]

            if self.debug:
                self._debug_derivs(
                    dD_dq / CONSTANTS["MHz2J"],
                    q,
                    sym,
                    i + 1,
                    V_0_0[i] / CONSTANTS["MHz2J"],
                    V_0_pm[i] / CONSTANTS["MHz2J"],
                    V_p_m[i] / CONSTANTS["MHz2J"],
                )

        print("Symmetry adjusted coefficients (1st order):")
        print(f"V_00:  {np.sum(V_0_0 > 0)}")
        print(f"V_pm:  {np.sum(V_p_m > 0)}")
        print(f"V_0pm: {np.sum(V_0_pm > 0)}")

        return zfs_deriv, V_0_0, V_p_m, V_0_pm

    def calculate_second_order_derivatives(self, zfs_1d_derivs: np.ndarray, ipr_thresh: Optional[float] = None):
        """
        Calculates 2nd order finite difference derivatives d2D/dqi dqj and 2-phonon coupling coefficients.
        """
        print("Calculating second-order derivatives...")
        n_modes = self.nmodes
        phonon_energies = self.get_phonon_frequencies()

        zfs_2nd_derivs = np.zeros((n_modes, n_modes, 3, 3))
        V_0_0_2nd = np.zeros((n_modes, n_modes))
        V_p_m_2nd = np.zeros((n_modes, n_modes))
        V_0_pm_2nd = np.zeros((n_modes, n_modes))

        for (i, j), item in sorted(self.zfs_tensors_2d.items()):
            if i not in self.treated_modes and j not in self.treated_modes:
                continue

            if ipr_thresh is not None:
                ipr_i, ipr_j = item["ipr"]
                if ipr_i < ipr_thresh and ipr_j < ipr_thresh:
                    continue

            (q_i, q_j) = item["pert"]
            if q_i is None or q_j is None or q_i != q_j:
                continue

            dD_qi = zfs_1d_derivs[i]
            dD_qj = zfs_1d_derivs[j]
            sym_i, sym_j = item["symmetry"]
            sym = MathUtils.calc_symmetry(sym_i, sym_j)

            D_qi_qj = item["tensor"]
            d2D_dqidqj = (D_qi_qj - self.zfs_relaxed) / (q_i * q_j) - dD_qi / q_j - dD_qj / q_i

            self._check_symmetry(d2D_dqidqj, (sym_i, sym_j), (i, j))

            zfs_2nd_derivs[i, j] = d2D_dqidqj
            zfs_2nd_derivs[j, i] = d2D_dqidqj

            if sym == ["A1"]:
                V_0_0_2nd[i, j] = np.abs(d2D_dqidqj[2, 2]) / 4.0
            elif {sym_i, sym_j} == {"Ex"}:
                V_0_0_2nd[i, j] = np.abs(d2D_dqidqj[2, 2]) / 4.0
                V_0_pm_2nd[i, j] = 0.5 * np.abs(np.mean([d2D_dqidqj[2, 0], d2D_dqidqj[0, 2]])) / np.sqrt(2)
                V_p_m_2nd[i, j] = 0.25 * np.abs(d2D_dqidqj[0, 0] - d2D_dqidqj[1, 1])
            elif {sym_i, sym_j} == {"Ey"}:
                V_0_0_2nd[i, j] = np.abs(d2D_dqidqj[2, 2]) / 4.0
                V_0_pm_2nd[i, j] = 0.5 * np.abs(np.mean([d2D_dqidqj[2, 0], d2D_dqidqj[0, 2]])) / np.sqrt(2)
                V_p_m_2nd[i, j] = 0.25 * np.abs(d2D_dqidqj[1, 1] - d2D_dqidqj[0, 0])
            elif {sym_i, sym_j} == {"Ex", "Ey"}:
                V_0_pm_2nd[i, j] = 0.5 * np.abs(np.mean([d2D_dqidqj[1, 2], d2D_dqidqj[2, 1]])) / np.sqrt(2)
                V_p_m_2nd[i, j] = 0.5 * np.abs(np.mean([d2D_dqidqj[1, 0], d2D_dqidqj[0, 1]]))

            if len(self.treated_modes) < n_modes and len(phonon_energies) == n_modes and i == j:
                freq_i = phonon_energies[i]
                if i + 1 < n_modes and np.isclose(freq_i, phonon_energies[i + 1]):
                    V_0_0_2nd[i + 1, j + 1] = V_0_0_2nd[i, j]
                    V_0_pm_2nd[i + 1, j + 1] = V_0_pm_2nd[i, j]
                    V_p_m_2nd[i + 1, j + 1] = V_p_m_2nd[i, j]
                elif i > 0 and np.isclose(freq_i, phonon_energies[i - 1]):
                    V_0_0_2nd[i - 1, j - 1] = V_0_0_2nd[i, j]
                    V_0_pm_2nd[i - 1, j - 1] = V_0_pm_2nd[i, j]
                    V_p_m_2nd[i - 1, j - 1] = V_p_m_2nd[i, j]

            V_0_0_2nd[j, i] = V_0_0_2nd[i, j]
            V_p_m_2nd[j, i] = V_p_m_2nd[i, j]
            V_0_pm_2nd[j, i] = V_0_pm_2nd[i, j]

            if self.debug:
                self._debug_derivs(
                    d2D_dqidqj / CONSTANTS["MHz2J"],
                    item["pert"],
                    item["symmetry"],
                    (i + 1, j + 1),
                    V_0_0_2nd[i, j] / CONSTANTS["MHz2J"],
                    V_0_pm_2nd[i, j] / CONSTANTS["MHz2J"],
                    V_p_m_2nd[i, j] / CONSTANTS["MHz2J"],
                )

        print("Symmetry adjusted coefficients (2nd order):")
        print(f"V_00:  {np.sum(V_0_0_2nd > 0)}")
        print(f"V_pm:  {np.sum(V_p_m_2nd > 0)}")
        print(f"V_0pm: {np.sum(V_0_pm_2nd > 0)}")

        return zfs_2nd_derivs, V_0_0_2nd, V_p_m_2nd, V_0_pm_2nd

    def process_first_order_perturbations(self, output_filename: Optional[str] = None) -> list[str]:
        if not self.zfs_tensors:
            raise ValueError("First order ZFS data not loaded.")

        pert_SI = self.pert_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.get_phonon_pert(pert_SI)
        zfs_derivs, V_0_0, V_p_m, V_0_pm = self.calculate_first_order_derivatives()

        save_name = (
            f"{output_filename}.npz"
            if output_filename
            else f"derivatives/{self.defect}_{self.cell_size}_zfs_coefficients_{self.calc_method}_{self.pert_scale}_.npz"
        )

        save_path = self.save_data(
            save_name,
            zfs=1.5 * self.zfs_relaxed[2, 2],
            zfs_derivs=zfs_derivs,
            V_0_0=V_0_0,
            V_p_m=V_p_m,
            V_0_pm=V_0_pm,
            freqs=phonon_pert["freqs"],
            sym=phonon_pert["sym"],
            ipr=phonon_pert["ipr"],
        )
        return [save_path]

    def process_second_order_perturbations(
        self, zfs_1d_derivs_file: Optional[str] = None, output_filename: Optional[str] = None
    ) -> list[str]:
        if not self.zfs_tensors_2d:
            raise ValueError("Second order ZFS data not loaded.")

        results = []
        if not self.zfs_tensors and zfs_1d_derivs_file:
            first_order_data = np.load(zfs_1d_derivs_file)
            zfs_1d_derivs = first_order_data["zfs_derivs"]
        else:
            first_order_file = self.process_first_order_perturbations(output_filename=output_filename)[0]
            results.append(first_order_file)
            zfs_1d_derivs = np.load(first_order_file, allow_pickle=True)["zfs_derivs"]

        zfs_2nd_derivs, V_0_0_2nd, V_p_m_2nd, V_0_pm_2nd = self.calculate_second_order_derivatives(zfs_1d_derivs)

        pert_SI = self.pert_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.get_phonon_pert(pert_SI)

        save_name = (
            f"{output_filename}.npz"
            if output_filename
            else f"derivatives/{self.defect}_{self.cell_size}_zfs2d_coefficients_{self.calc_method}_{self.pert_scale}_.npz"
        )
        save_path = self.save_data(
            save_name,
            second_order=True,
            zfs=1.5 * self.zfs_relaxed[2, 2],
            zfs_derivs=zfs_2nd_derivs,
            V_0_0=V_0_0_2nd,
            V_p_m=V_p_m_2nd,
            V_0_pm=V_0_pm_2nd,
            freqs=phonon_pert["freqs"],
            sym=phonon_pert["sym"],
            ipr=phonon_pert["ipr"],
        )
        results.append(save_path)
        return results

    def save_data(self, save_name: str, **kwargs) -> str:
        if not save_name.endswith(".npz"):
            save_name += ".npz"

        metadata = {
            "defect": self.defect,
            "cell_size": self.cell_size,
            "calc_method": self.calc_method,
            "pert_scale": self.pert_scale,
        }
        Path(save_name).parent.mkdir(parents=True, exist_ok=True)
        np.savez(save_name, **metadata, **kwargs)
        print(f"Saved ZFS data to: {save_name}")
        return save_name

    @staticmethod
    def _check_symmetry(d_tensor: np.ndarray, symmetry: Union[str, tuple[str, str]], idx: Any) -> None:
        sym_prod = (
            MathUtils.calc_symmetry(*symmetry) if isinstance(symmetry, tuple) else [symmetry]
        )
        if sym_prod == ["A1"]:
            diag = np.diag(d_tensor)
            d_xx, d_yy, d_zz = diag
            tensor_scale = np.max(np.abs(diag))
            dyn_tol = 1.0 * tensor_scale + 1e-4

            is_axial = np.abs(d_xx - d_yy) <= dyn_tol
            is_traceless_axial = np.abs(2 * d_xx + d_zz) <= dyn_tol
            off_diag_mask = ~np.eye(3, dtype=bool)
            off_diags_zero = np.all(np.abs(d_tensor[off_diag_mask]) <= dyn_tol)

            if not (is_axial and is_traceless_axial and off_diags_zero):
                print(f"\nWarning: Symmetry mismatch at index {idx} with symmetry {sym_prod}")
                print(f"Threshold: {dyn_tol:.2e} | Scale: {tensor_scale:.2f}")

    def _debug_derivs(self, dD, q, symmetry, idx, V_0_0, V_0_pm, V_p_m):
        max_val = np.max(np.abs(dD))
        col_width = 12
        total_width = (col_width * 3) + 10
        double_line = "=" * total_width

        freqs = self.get_phonon_frequencies()
        if isinstance(idx, tuple):
            energy = (freqs[idx[0] - 1], freqs[idx[1] - 1]) if len(freqs) >= max(idx) else (0, 0)
        else:
            energy = freqs[idx - 1] if len(freqs) >= idx else 0

        print(f"\n{double_line}\n DEBUG DERIVATIVES\n{'-' * total_width}")
        print(
            f" Mode: {idx}\n"
            f" Symmetry: {symmetry}\n"
            f" Energy: {energy}\n"
            f" Perturbation: {MathUtils.fmt(q)}\n"
            f"{double_line}\n"
            f" Max Value: {max_val:<10.6f}\n"
            f" V_0_0: {V_0_0}\n"
            f" V_0_pm: {V_0_pm}\n"
            f" V_p_m: {V_p_m}"
        )
        print(" Tensor Structure (3x3):\n")
        for row in dD:
            formatted_row = "  ".join(f"{val:>{col_width}.6f}" for val in row)
            print(f"  [ {formatted_row} ]")
        print(f"{double_line}\n")
