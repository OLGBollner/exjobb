from concurrent.futures import ProcessPoolExecutor
from functools import partial
import re
import numpy as np
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
from datetime import datetime

from beyblade.constants import CONSTANTS
from beyblade.phonon_manager import PhononManager


class ZFSManager:
    def __init__(self, phonon_manager: PhononManager, debug=False):
        # Phonons
        self.phonon_manager = phonon_manager

        # Crystal data
        self.defect: str =               None
        self.cell_size: int =            None

        # Simulation metadata
        self.perturbation_scale: float = None
        self.sub_folder: str =           None

        # ZFS Data
        self.zfs_relaxed: np.ndarray =    None
        self.zfs_tensors: np.ndarray =    None
        self.zfs_tensors_2d: np.ndarray = None
        self.eigen_rotation: np.ndarray = None

        # Other flags
        self.debug: bool = debug


    def load_outcar_zfs_data(self, **kwargs):
        if (kwargs.get("sim_folder", None) is not None
            and kwargs.get("sub_folder", None) is not None 
            and kwargs.get("zfs_folder", None) is not None):

            sim_folder = Path(kwargs["sim_folder"])
            zfs_folder = kwargs["zfs_folder"]

            self.sub_folder = kwargs["sub_folder"]

            if "pert" not in sim_folder.name:
                raise ValueError("Perturbation scale not found in folder name. Ensure the folder name contains the perturbation scale (e.g., 'pert_0.01').")

            self.perturbation_scale: float = float(sim_folder.name.split("_")[1])
            self.defect: str =               sim_folder.parent.parent.name.split("_")[0]
            self.cell_size: int =            int(sim_folder.parent.parent.name.split("_")[-1])

            print(f"Initialized ZFSManager with perturbation scale: {self.perturbation_scale}, cell size: {self.cell_size}")

            self._get_zfs_data(sim_folder, zfs_folder)

            pert_SI = self.perturbation_scale * CONSTANTS["ang_amu2SI"]
            phonon_pert = self.phonon_manager.get_phonon_pert(pert_SI)

            search_path = sim_folder / self.sub_folder
            
            if "first" in sim_folder.parent.name:
                self.zfs_tensors = self._load_zfs_perts(search_path, phonon_pert)
            if "second" in sim_folder.parent.name:
                self.zfs_tensors_2d = self._load_zfs_perts_2d(search_path, phonon_pert)
            else:
                raise ValueError(f"No valid order was specified: {sim_folder.parent.name}")
  

            print("Succesfully loaded ZFS data from OUTCARs")
            return self.zfs_relaxed, self.zfs_tensors, self.zfs_tensors_2d, self.eigen_rotation

        elif kwargs.get("raw_data_path", None) is not None:
            raw_data = np.load(kwargs["raw_data_path"])

            self.sub_folder = raw_data["sub_folder"]
            self.perturbation_scale = raw_data["perturbation_scale"]
            self.defect = raw_data["defect"]
            self.cell_size = raw_data["cell_size"]

            self.zfs_relaxed = raw_data["zfs_relaxed"]
            self.eigen_rotation = raw_data["eigen_rotation"]

            if raw_data["order"] == 1:
                self.zfs_tensors = raw_data["zfs_tensors"]
            if raw_data["order"] == 2:
                self.zfs_tensors_2d = raw_data["zfs_tensors"]
            print("Succesfully loaded ZFS data from .npz file")

        else:
            print("Initialized empty phonon_manager, no data was given.")


    def save_data(self, save_name, **kwargs):
        if not save_name.endswith(".npz"):
            save_name += ".npz"
        #save_name = save_name.replace(".npz", datetime.now().strftime('%Y-%m-%d') + ".npz")

        metadata = {
            "defect": self.defect,
            "cell_size": self.cell_size,
            "sub_folder": self.sub_folder,
            "pert_scale": self.perturbation_scale
        }

        print(f"Saved ZFS data to: {save_name}")
        
        np.savez(save_name, **metadata, **kwargs)
        return save_name


    def read_zfs_tensor(self, outcar_file: str) -> Optional[dict[str, Any]]:
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


    def process_first_order_perturbations(self, output_filename=None):
        results = []

        if "approx" in self.sub_folder:
            self.zfs_tensors *= 3/2
            self.zfs_relaxed *= 3/2

        pert_SI = self.perturbation_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.phonon_manager.get_phonon_pert(pert_SI)

        zfs_derivs, V_0_0, V_p_m, V_0_pm = self._calc_derivative(
            phonon_pert["eigs"], phonon_pert.get("sym"), phonon_pert.get("idx")
        )

        save_name = f"{output_filename}.npz" if output_filename else f"derivatives/{self.defect}_{self.cell_size}_zfs_coefficients_{self.sub_folder}_{self.perturbation_scale}_.npz"

        save_name = self.save_data(save_name, zfs_derivs=zfs_derivs*CONSTANTS["MHz2meV"], V_0_0=V_0_0*CONSTANTS["MHz2meV"], V_p_m=V_p_m*CONSTANTS["MHz2meV"],
                                    V_0_pm=V_0_pm*CONSTANTS["MHz2meV"], freqs=phonon_pert["freqs"], sym=phonon_pert["sym"], ipr=phonon_pert["ipr"])

        results.append(save_name)
        return results
    

    def process_second_order_perturbations(self, zfs_1d_derivs_file, output_filename=None):
        results = []
            
        save_name = f"{output_filename}.npz" if output_filename else f"derivatives/{self.defect}_{self.cell_size}_zfs2d_coefficients_{self.sub_folder}_{self.perturbation_scale}_.npz"

        """if Path(save_name).exists:
            zfs_2d = np.load(save_name)
            results.append(save_name)
            print("ZFS derivatives already calculated: ", save_name)
            return results
        """
        # Load the pre-calculated first order derivatives to optimize compute
        first_order_data = np.load(zfs_1d_derivs_file)
        zfs_1d_derivs = first_order_data["zfs_derivs"] / CONSTANTS["MHz2meV"]

        if "approx" in self.sub_folder:
            for key in self.zfs_tensors_2d:
                self.zfs_tensors_2d[key] *= 3/2
            self.zfs_relaxed *= 3/2

        pert_SI = self.perturbation_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.phonon_manager.get_phonon_pert(pert_SI)

        zfs_2nd_derivs, V_0_0_2nd, V_p_m_2nd, V_0_pm_2nd = self._calc_second_order_derivatives(
            self.zfs_tensors_2d, zfs_1d_derivs, self.zfs_relaxed, phonon_pert["eigs"], phonon_pert["sym"], phonon_pert["idx"]
        )
        
        save_name = self.save_data(save_name, second_order=True, zfs_derivs=zfs_2nd_derivs*CONSTANTS["MHz2meV"], V_0_0=V_0_0_2nd*CONSTANTS["MHz2meV"], V_p_m=V_p_m_2nd*CONSTANTS["MHz2meV"],
                                    V_0_pm=V_0_pm_2nd*CONSTANTS["MHz2meV"], freqs=phonon_pert["freqs"], sym=phonon_pert["sym"], ipr=phonon_pert["ipr"])

        results.append(save_name)
        return results


    def _load_zfs_perts(self, search_path, phonon_pert):
        print("Reading ZFS tensors from: ", search_path)
        outcars = list(search_path.glob("**/OUTCAR"))
        zfs_tensors = {int(outcar.parent.name): val for outcar in outcars if (val := self.read_zfs_tensor(str(outcar)))}
        zfs_tensors = [val for key, val in sorted(zfs_tensors.items(), key=lambda item: item[0]) if (key-1) in phonon_pert["idx"]]

        eigen_rotation_t = self.eigen_rotation.T
        zfs_tensors = np.array([self.eigen_rotation @ mode["D_tensor"] @ eigen_rotation_t for mode in zfs_tensors])

        print("ZFS shape: ", zfs_tensors.shape)
        return zfs_tensors


    #TODO: Never use phonon index?
    def _load_zfs_perts_2d(self, search_path, phonon_pert):
        # Implementation to read the 2D grid of OUTCAR files
        print("Reading ZFS tensors from: ", search_path)

        zfs_tensors = {}
        total_modes = int(len(phonon_pert["idx"])**2 / 2)
        print(f"Expecting up to {total_modes} OUTCAR files for 2D perturbations.")
        outcars = search_path.glob("**/OUTCAR")
        eigen_rotation_t = self.eigen_rotation.T

        worker_task = partial(self._process_outcar_worker, eigen_rotation=self.eigen_rotation, eigen_rotation_t=eigen_rotation_t)

        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(worker_task, outcars), total=total_modes, desc="Processing OUTCAR files"))

        # Filter out any None results
        zfs_tensors = {r[0]: r[1] for r in results if r is not None}

        num_entries = len(zfs_tensors)

        print(f"Total stored pairs: {num_entries}")

        return zfs_tensors


    def _process_outcar_worker(self, outcar, eigen_rotation, eigen_rotation_t) -> Optional[tuple[tuple[int, int], np.ndarray]] | None:
        zfs = self.read_zfs_tensor(str(outcar))
        if zfs is None:
            return None

        index_parts = outcar.parent.name.split("_")
        indices = tuple(int(part)-1 for part in index_parts if part.isdigit())
        if len(indices) != 2:
            print(f"Warning: Could not extract valid indices from folder name {outcar.parent.name}. Skipping this file.")
            return None

        transformed_zfs = eigen_rotation @ zfs["D_tensor"] @ eigen_rotation_t
        return indices, transformed_zfs


    def _get_zfs_data(self, sim_folder, zfs_folder):
        if self.zfs_relaxed is None and self.eigen_rotation is None:
            main_path = sim_folder.parent.parent

            zfs_relaxed = self.read_zfs_tensor(str(main_path / zfs_folder / "OUTCAR"))
            if zfs_relaxed is None:
                raise ValueError("Relaxed ZFS tensor not found. Ensure the OUTCAR file exists and contains the ZFS tensor data.")
            eigen_rotation = zfs_relaxed["eigenvectors"]
            self.zfs_relaxed = np.diag(zfs_relaxed["D_diag"])
            self.eigen_rotation = np.array(eigen_rotation)

        return self.zfs_relaxed, self.eigen_rotation


    def _debug_derivs(self, dD, symmetry, idx):
        max_val = np.max(np.abs(dD))

        col_width = 12
        total_width = (col_width * 3) + 10
        double_line = "=" * total_width

        print(f"\n{double_line}\n DEBUG DERIVATIVES\n{'-' * total_width}")
        print(f" Mode: {idx}\n Symmetry: {symmetry}\n{double_line}\n Max Value: {max_val:<10.6f}")
        print(" Tensor Structure (3x3):\n")
        for row in dD:
            formatted_row = "  ".join(f"{val:>{col_width}.6f}" for val in row)
            print(f"  [ {formatted_row} ]")
        print(f"{double_line}\n")


    def _calc_derivative(self, phonon_eigs, symmetry, phonon_idx):
        print("Calculating derivatives...")

        zfs_deriv = np.zeros(shape=self.zfs_tensors.shape)
        V_0_0 = np.zeros(shape=phonon_eigs.shape[0])
        V_0_pm = np.zeros(shape=phonon_eigs.shape[0])
        V_p_m = np.zeros(shape=phonon_eigs.shape[0])

        for i, q in enumerate(phonon_eigs):
            dD = (self.zfs_tensors[i] - self.zfs_relaxed)
            zfs_deriv[i] = dD / q

            trace_in_plane = dD[0, 0] + dD[1, 1]
            diff_in_plane = dD[0, 0] - dD[1, 1]
            off_diag_in_plane = dD[0, 1]

            sym = symmetry[i]

            if self.debug:
                self._debug_derivs(dD, sym, phonon_idx[i]+1)

            if sym == "A1":
                V_0_0[i] = np.abs(dD[2, 2] - 0.5 * trace_in_plane) / q
            elif sym in ["Ex"]:
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

                # d2D_dqidqj = (D_qi_qj - zfs_relaxed) / (q_i * q_j) - (dD_qi / q_j) - (dD_qj / q_i)

                d2D_dqidqj = (D_qi_qj - zfs_relaxed - dD_qi*q_i - dD_qj*q_j)
                if self.debug:
                    self._debug_derivs(d2D_dqidqj, (symmetry[i], symmetry[j]), (phonon_idx[i]+1, phonon_idx[j]+1))

                zfs_2nd_derivs[i, j] = d2D_dqidqj
                zfs_2nd_derivs[j, i] = d2D_dqidqj

                trace_in_plane = d2D_dqidqj[0, 0] + d2D_dqidqj[1, 1]
                diff_in_plane = d2D_dqidqj[0, 0] - d2D_dqidqj[1, 1]
                off_diag_in_plane = d2D_dqidqj[0, 1]

                if symmetry[i] == symmetry[j]: # Måste ta hänsyn till att E moder bidrar till både 00 och +- i andra ordningens fononer
                    V_0_0_2nd[i, j] = np.abs(d2D_dqidqj[2, 2] - 0.5 * trace_in_plane) / (q_i*q_j)

                if symmetry[i] in ["Ex"] and symmetry[j] in ["A1", "A2", "Ex"]:
                    V_p_m_2nd[i, j] = 2 * (0.5 * np.sqrt(diff_in_plane**2 + 2 * off_diag_in_plane**2)) / (q_i*q_j)
                    V_0_pm_2nd[i, j] = 2 * (np.sqrt(d2D_dqidqj[0, 2]**2 + d2D_dqidqj[1, 2]**2)) / (q_i*q_j)

                V_0_0_2nd[j, i] = V_0_0_2nd[i, j]
                V_p_m_2nd[j, i] = V_p_m_2nd[i, j]
                V_0_pm_2nd[j, i] = V_0_pm_2nd[i, j]

        print("Symmetry adjusted coefficients: ")
        print("V_00: ", np.sum(V_0_0_2nd > 0))
        print("V_pm: ", np.sum(V_p_m_2nd > 0))
        print("V_0pm: ", np.sum(V_0_pm_2nd > 0))

        return zfs_2nd_derivs, V_0_0_2nd / 3, V_p_m_2nd, V_0_pm_2nd / np.sqrt(2)
