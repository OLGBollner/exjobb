from concurrent.futures import ProcessPoolExecutor
from functools import partial
import re
import numpy as np
from scipy import constants as Cn
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
from datetime import datetime

from beyblade.constants import CONSTANTS
from beyblade.utils import MathUtils
from beyblade.phonon_manager import PhononManager

def read_zfs_tensor(outcar_file: str) -> Optional[dict[str, Any]]:
    """
    Read the zero-field splitting (ZFS) tensor from a VASP OUTCAR file.

    Args:
        outcar_file: Path to the OUTCAR file

    Returns:
        Dictionary containing:
            - 'D_tensor': 3x3 matrix of ZFS tensor components (MHz)
            - 'D_diag': List of diagonal eigenvalues (MHz)
            - 'eigenvectors': List of eigenvectors
            - 'raw_values': Dictionary with D_xx, D_yy, D_zz, D_xy, D_xz, D_yz
        Returns None if ZFS tensor not found
    """
    try:
        with open(outcar_file, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File {outcar_file} not found")
        return None

    # Search for the ZFS tensor section
    zfs_pattern = r'Spin-spin contribution to zero-field splitting tensor \(MHz\)\s*-+\s*D_xx\s+D_yy\s+D_zz\s+D_xy\s+D_xz\s+D_yz\s*-+\s*([\s\d\.\-]+?)(?=\s*-{3,})'

    match = re.search(zfs_pattern, content)
    if not match:
        print("Warning: ZFS tensor section not found in OUTCAR: ", outcar_file)
        return None

    # Parse the tensor values
    values_str = match.group(1).strip()
    values = [float(x) for x in values_str.split()]
    if len(values) != 6:
        print(f"Error: Expected 6 ZFS tensor values, got {len(values)}")
        return None

    D_xx, D_yy, D_zz, D_xy, D_xz, D_yz = values

    # Construct the symmetric 3x3 tensor
    D_tensor = np.array([
        [D_xx, D_xy, D_xz],
        [D_xy, D_yy, D_yz],
        [D_xz, D_yz, D_zz]
    ])

    # Parse diagonalized values and eigenvectors
    diag_pattern = r'after diagonalization\s*-+\s*D_diag\s+eigenvector \(x,y,z\)\s*-+\s*((?:[\d\.\s\-]+\s+[\d\.\-]+\s+[\d\.\-]+\s+[\d\.\-]+\n?)+)'

    diag_match = re.search(diag_pattern, content)
    D_diag: list[float] = []
    eigenvectors: list[list[float]] = []

    if diag_match:
        diag_lines = diag_match.group(1).strip().split('\n')
        for line in diag_lines:
            if line.strip():
                parts = line.split()
                if len(parts) == 4:
                    D_diag.append(float(parts[0]))
                    eigenvectors.append([float(parts[1]), float(parts[2]), float(parts[3])])

    return {
        'D_tensor': D_tensor,
        'D_diag': D_diag,
        'eigenvectors': eigenvectors,
        'raw_values': {
            'D_xx': D_xx, 'D_yy': D_yy, 'D_zz': D_zz,
            'D_xy': D_xy, 'D_xz': D_xz, 'D_yz': D_yz
        }
    }

class ZFSCalculator:
    def __init__(self, sim_folder, sub_folder, zfs_folder, phonon_manager: PhononManager, debug=False):
        self.sim_folder = Path(sim_folder)
        if "pert" not in self.sim_folder.name:
            raise ValueError("Perturbation scale not found in folder name. Ensure the folder name contains the perturbation scale (e.g., 'pert_0.01').")
        self.perturbation_scale: float = float(self.sim_folder.name.split("_")[1])
        self.cell_size: int = int(self.sim_folder.parent.parent.name.split("_")[-1])
        print(f"Initialized ZFSCalculator with perturbation scale: {self.perturbation_scale}, cell size: {self.cell_size}")

        self.sub_folder: str = sub_folder
        self.zfs_folder: str = zfs_folder

        self.phonon_manager = phonon_manager
        self.debug: bool = debug

    def _save_derivative_data(self, save_name, **kwargs):
        if not save_name.endswith(".npz"):
            save_name += datetime.now().strftime('%Y-%m-%d') + ".npz"

        metadata = {
            "cell_size": self.cell_size,
            "sub_folder": self.sub_folder,
            "pert_scale": self.perturbation_scale
        }

        print(f"Saved ZFS data in meV to: {save_name}")
        
        np.savez(save_name, **metadata, **kwargs)


    def process_first_order_perturbations(self, output_filename=None):
        results = []

        pert_SI = abs(self.perturbation_scale) *  CONSTANTS["ang_amu2SI"]
        zfs_relaxed, phonon_pert, eigen_rotation = self._get_zfs_data(self.sim_folder, pert_SI)

        search_path = self.sim_folder / self.sub_folder
        zfs_tensor = self._load_zfs_perts(search_path, eigen_rotation, phonon_pert)

        if "approx" in self.sub_folder:
            zfs_tensor *= 3/2
            zfs_relaxed *= 3/2

        zfs_derivs, V_0_0, V_p_m, V_0_pm = self._calc_derivative(
            zfs_tensor, zfs_relaxed, phonon_pert["eigs"], phonon_pert.get("sym"), phonon_pert.get("idx")
        )

        save_name = f"{output_filename}.npz" if output_filename else f"derivatives/zfs_coefficients_{self.cell_size}_{self.sub_folder}_{self.perturbation_scale}_.npz"

        self._save_derivative_data(save_name, zfs_derivs=zfs_derivs*CONSTANTS["MHz2meV"], V_0_0=V_0_0*CONSTANTS["MHz2meV"], V_p_m=V_p_m*CONSTANTS["MHz2meV"],
                                    V_0_pm=V_0_pm*CONSTANTS["MHz2meV"], freqs=phonon_pert["freqs"], sym=phonon_pert["sym"], ipr=phonon_pert["ipr"])

        results.append(save_name)
        return results
    

    def process_second_order_perturbations(self, zfs_1d_derivs_file, output_filename=None):
        results = []
            
        pert_SI = abs(self.perturbation_scale) * CONSTANTS["ang_amu2SI"]

        save_name = f"{output_filename}.npz" if output_filename else f"derivatives/zfs2d_coefficients_{self.cell_size}_{self.sub_folder}_{self.perturbation_scale}_.npz"

        """if Path(save_name).exists:
            zfs_2d = np.load(save_name)
            results.append(save_name)
            print("ZFS derivatives already calculated: ", save_name)
            return results
        """
        # Load the pre-calculated first order derivatives to optimize compute
        first_order_data = np.load(zfs_1d_derivs_file)
        zfs_1d_derivs = first_order_data["zfs_derivs"] / CONSTANTS["MHz2meV"]

        zfs_relaxed, phonon_pert, eigen_rotation = self._get_zfs_data(self.sim_folder, pert_SI)

        search_path = self.sim_folder / self.sub_folder

        zfs_2d = self._load_zfs_perts_2d(search_path, eigen_rotation, phonon_pert)

        if "approx" in self.sub_folder:
            for key in zfs_2d:
                zfs_2d[key] *= 3/2
            zfs_relaxed *= 3/2

        zfs_2nd_derivs, V_0_0_2nd, V_p_m_2nd, V_0_pm_2nd = self._calc_second_order_derivatives(
            zfs_2d, zfs_1d_derivs, zfs_relaxed, phonon_pert["eigs"], phonon_pert["sym"], phonon_pert["idx"]
        )
        
        self._save_derivative_data(save_name, second_order=True, zfs_derivs=zfs_2nd_derivs*CONSTANTS["MHz2meV"], V_0_0=V_0_0_2nd*CONSTANTS["MHz2meV"], V_p_m=V_p_m_2nd*CONSTANTS["MHz2meV"],
                                    V_0_pm=V_0_pm_2nd*CONSTANTS["MHz2meV"], freqs=phonon_pert["freqs"], sym=phonon_pert["sym"], ipr=phonon_pert["ipr"])

        results.append(save_name)
        return results

    def _load_zfs_perts(self, search_path, eigen_rotation, phonon_pert):
        print("Reading ZFS tensors from: ", search_path)
        outcars = list(search_path.glob("**/OUTCAR"))
        zfs_tensor = {int(outcar.parent.name): val for outcar in outcars if (val := read_zfs_tensor(str(outcar)))}
        zfs_tensor = [val for key, val in sorted(zfs_tensor.items(), key=lambda item: item[0]) if (key-1) in phonon_pert["idx"]]

        zfs_tensor = np.array([eigen_rotation @ mode["D_tensor"] @ np.transpose(eigen_rotation) for mode in zfs_tensor])

        print("ZFS shape: ", zfs_tensor.shape)
        return zfs_tensor

    def _load_zfs_perts_2d(self, search_path, eigen_rotation, phonon_pert):
        # Implementation to read the 2D grid of OUTCAR files
        # It should return zfs_2d_dict, zfs_relaxed, and phonon_pert
        # zfs_2d_dict maps (i, j) -> 3x3 numpy array
        print("Reading ZFS tensors from: ", search_path)

        zfs_tensor = {}
        total_modes = int(len(phonon_pert["idx"])**2/2)
        print(f"Expecting up to {total_modes} OUTCAR files for 2D perturbations.")
        outcars = search_path.glob("**/OUTCAR")
        eigen_rotation_t = np.transpose(eigen_rotation)

        worker_task = partial(self._process_outcar_worker, eigen_rotation=eigen_rotation, eigen_rotation_t=eigen_rotation_t)

        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(worker_task, outcars), total=total_modes, desc="Processing OUTCAR files"))

        # Filter out any None results
        zfs_tensor = {r[0]: r[1] for r in results if r is not None}

        #zfs_tensor = {(int(outcar.parent.name.split("_")[0]), int(outcar.parent.name.split("_")[1])): val for outcar in outcars if (val := read_zfs_tensor(str(outcar)))}
        #zfs_tensor = {key: eigen_rotation @ mode["D_tensor"] @ np.transpose(eigen_rotation) for key, mode in zfs_tensor.items()}
                
        num_entries = len(zfs_tensor)

        print(f"Total stored pairs: {num_entries}")

        return zfs_tensor

    def _process_outcar_worker(self, outcar, eigen_rotation, eigen_rotation_t) -> Optional[tuple[tuple[int, int], np.ndarray]] | None:
        zfs = read_zfs_tensor(str(outcar))
        if zfs is None:
            return None

        index_parts = outcar.parent.name.split("_")
        indices = tuple(int(part) for part in index_parts if part.isdigit())
        if len(indices) != 2:
            print(f"Warning: Could not extract valid indices from folder name {outcar.parent.name}. Skipping this file.")
            return None

        transformed_zfs = eigen_rotation @ zfs["D_tensor"] @ eigen_rotation_t
        return indices, transformed_zfs

    def _get_zfs_data(self, path_to_pert, pert_SI):
        main_path = path_to_pert.parent.parent

        phonons = self.phonon_manager.data
        sym_data = self.phonon_manager.symmetry_data
        if phonons is None:
            raise ValueError("No phonon data loaded. Ensure the phonon data file is present and correctly formatted.")
        
        if sym_data is None:
            raise ValueError("No symmetry data loaded. Ensure the phonon data file is present and correctly formatted.")
        phonon_pert = {}
        mask = phonons["freqs"] > 0

        if phonons.get("sym") is not None:
            phonon_pert["sym"] = phonons["sym"][mask]
            phonon_pert["idx"] = phonons["idx"][mask]
        else:
            print("No symmetry data, find symmetries.")
            phonon_pert["sym"] = sym_data["sym"][mask]
            phonon_pert["idx"] = sym_data["idx"][mask]

        Q = [np.sqrt(np.sum(mode**2)) for mode in phonons["eigs"]]

        phonon_pert["eigs"] = np.array([
            pert_SI * mode * np.sqrt(2 * CONSTANTS["meV2rads"] * freq / Cn.hbar) if freq > 0 else None
            for mode, freq in zip(Q, phonons["freqs"])
        ])[mask]

        phonon_pert["freqs"] = phonons["freqs"][mask]
        phonon_pert["ipr"] = MathUtils.calc_ipr(phonons)[mask]

        zfs_relaxed = read_zfs_tensor(main_path / self.zfs_folder / "OUTCAR")
        if zfs_relaxed is None:
            raise ValueError("Relaxed ZFS tensor not found. Ensure the OUTCAR file exists and contains the ZFS tensor data.")
        eigen_rotation = zfs_relaxed["eigenvectors"]
        zfs_relaxed = np.diag(zfs_relaxed["D_diag"])

        return zfs_relaxed, phonon_pert, np.array(eigen_rotation)

    def _debug_derivs(self, dD, symmetry, idx):
        max_val = np.max(np.abs(dD))

        col_width = 12
        total_width = (col_width * 3) + 10
        double_line = "=" * total_width

        print(f"\n{double_line}\n DEBUG DERIVATIVES\n{'-' * total_width}")
        print(f" Mode: {idx+1}\n Symmetry: {symmetry}\n{double_line}\n Max Value: {max_val:<10.6f}")
        print(" Tensor Structure (3x3):\n")
        for row in dD:
            formatted_row = "  ".join(f"{val:>{col_width}.6f}" for val in row)
            print(f"  [ {formatted_row} ]")
        print(f"{double_line}\n")

    def _calc_derivative(self, zfs_tensor, zfs_relaxed, phonon_eigs, symmetry, phonon_idx):
        print("Calculating derivatives...")

        zfs_deriv = np.zeros(shape=zfs_tensor.shape)
        V_0_0 = np.zeros(shape=phonon_eigs.shape[0])
        V_0_pm = np.zeros(shape=phonon_eigs.shape[0])
        V_p_m = np.zeros(shape=phonon_eigs.shape[0])

        for i, q in enumerate(phonon_eigs):
            dD = (zfs_tensor[i] - zfs_relaxed)
            zfs_deriv[i] = dD / q

            trace_in_plane = dD[0, 0] + dD[1, 1]
            diff_in_plane = dD[0, 0] - dD[1, 1]
            off_diag_in_plane = dD[0, 1]

            sym = symmetry[i]

            if self.debug:
                self._debug_derivs(dD, sym, phonon_idx[i])

            if sym == "A1":
                V_0_0[i] = np.abs(dD[2, 2] - 0.5 * trace_in_plane) / q
            elif sym in ["Ex"]: # HUR FAN SKA DETTA SE UT?
                V_p_m[i] = 2*(0.5 * np.sqrt(diff_in_plane**2 + 2 * off_diag_in_plane**2) / q)
                V_0_pm[i] = 2*(np.sqrt(dD[0, 2]**2 + dD[1, 2]**2) / q)

        print("Symmetry adjusted coefficients: ")
        print("V_00: ", np.sum(V_0_0 > 0))
        print("V_pm: ", np.sum(V_p_m > 0))
        print("V_0pm: ", np.sum(V_0_pm > 0))

        return zfs_deriv, V_0_0 / 3, V_p_m, V_0_pm / np.sqrt(2)

    def _calc_second_order_derivatives(self, zfs_2d_dict, zfs_1d_derivs, zfs_relaxed, phonon_eigs, symmetry, phonon_idx):
        print("Calculating derivatives...")
        n_modes = len(phonon_eigs)
        
        zfs_2nd_derivs = np.zeros((n_modes, n_modes, 3, 3))
        V_0_0_2nd = np.zeros((n_modes, n_modes))
        V_p_m_2nd = np.zeros((n_modes, n_modes))
        V_0_pm_2nd = np.zeros((n_modes, n_modes))

        for i in range(n_modes):
            q_i = phonon_eigs[i]
            dD_qi = zfs_1d_derivs[i]

            for j in range(i, n_modes):
                q_j = phonon_eigs[j]
                dD_qj = zfs_1d_derivs[j]

                if (i, j) not in zfs_2d_dict:
                    continue

                D_qi_qj = zfs_2d_dict[(i, j)]

                d2D_dqidqj = (D_qi_qj - zfs_relaxed) / (q_i * q_j) - (dD_qi / q_j) - (dD_qj / q_i)

                zfs_2nd_derivs[i, j] = d2D_dqidqj
                zfs_2nd_derivs[j, i] = d2D_dqidqj

                trace_in_plane = d2D_dqidqj[0, 0] + d2D_dqidqj[1, 1]
                diff_in_plane = d2D_dqidqj[0, 0] - d2D_dqidqj[1, 1]
                off_diag_in_plane = d2D_dqidqj[0, 1]

                if symmetry[i] == symmetry[j]: # Måste ta hänsyn till att E moder bidrar till både 00 och +- i andra ordningens fononer
                    V_0_0_2nd[i, j] = np.abs(d2D_dqidqj[2, 2] - 0.5 * trace_in_plane)

                if symmetry[i] in ["Ex"] and symmetry[j] in ["A1", "A2", "Ex"]:
                    V_p_m_2nd[i, j] = 2 * (0.5 * np.sqrt(diff_in_plane**2 + 2 * off_diag_in_plane**2))
                    V_0_pm_2nd[i, j] = 2 * (np.sqrt(d2D_dqidqj[0, 2]**2 + d2D_dqidqj[1, 2]**2))

                V_0_0_2nd[j, i] = V_0_0_2nd[i, j]
                V_p_m_2nd[j, i] = V_p_m_2nd[i, j]
                V_0_pm_2nd[j, i] = V_0_pm_2nd[i, j]

        print("Symmetry adjusted coefficients: ")
        print("V_00: ", np.sum(V_0_0_2nd > 0))
        print("V_pm: ", np.sum(V_p_m_2nd > 0))
        print("V_0pm: ", np.sum(V_0_pm_2nd > 0))

        return zfs_2nd_derivs, V_0_0_2nd / 3, V_p_m_2nd, V_0_pm_2nd / np.sqrt(2)
