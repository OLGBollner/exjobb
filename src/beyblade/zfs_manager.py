from concurrent.futures import ProcessPoolExecutor
from functools import partial
import re
import numpy as np
from pathlib import Path
from typing import Optional, Any
from tqdm import tqdm
from datetime import datetime
from matplotlib import pyplot as plt

from beyblade.constants import CONSTANTS
from beyblade.phonon_manager import PhononManager
from beyblade.utils import MathUtils


def read_zfs_tensor(outcar_file: str, debug: bool = False) -> Optional[dict[str, Any]]:
    """
    Read the zero-field splitting (ZFS) tensor from a VASP OUTCAR file.

    Args:
        outcar_file: Path to the OUTCAR file
        debug: If True, print additional debug information

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

    if debug:
        print(f"Debug: Extracted ZFS tensor values string:\n{values_str}\n")
        print(f"Debug: Extracted diagonal values: {D_diag}")
        print(f"Debug: Extracted eigenvectors: {eigenvectors}")

    return {
        'D_tensor': D_tensor,
        'D_diag': D_diag,
        'eigenvectors': eigenvectors,
        'raw_values': {
            'D_xx': D_xx, 'D_yy': D_yy, 'D_zz': D_zz,
            'D_xy': D_xy, 'D_xz': D_xz, 'D_yz': D_yz
        }
    }

def _process_outcar_worker_1d(outcar, eigen_rotation, eigen_rotation_t):
    zfs = read_zfs_tensor(str(outcar))
    if zfs is None:
        return None

    index = int(outcar.parent.name)-1
    transformed_zfs = eigen_rotation @ zfs["D_tensor"] @ eigen_rotation_t
    return index, transformed_zfs

def _process_outcar_worker_2d(outcar, eigen_rotation, eigen_rotation_t) -> Optional[tuple[tuple[int, int], np.ndarray]] | None:
    zfs = read_zfs_tensor(str(outcar))
    if zfs is None:
        return None

    index_parts = outcar.parent.name.split("_")
    indices = tuple(int(part)-1 for part in index_parts if part.isdigit())
    if len(indices) != 2:
        print(f"Warning: Could not extract valid indices from folder name {outcar.parent.name}. Skipping this file.")
        return None

    transformed_zfs = eigen_rotation @ zfs["D_tensor"] @ eigen_rotation_t
    return indices, transformed_zfs



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

            if "pert" not in sim_folder.name:
                raise ValueError("Perturbation scale not found in folder name. Ensure the folder name contains the perturbation scale (e.g., 'pert_0.01').")

            self.defect =    sim_folder.parent.parent.name.split("_")[0]
            self.cell_size = int(sim_folder.parent.parent.name.split("_")[-1])

            self.perturbation_scale = float(sim_folder.name.split("_")[1])
            self.sub_folder =                kwargs["sub_folder"]

            print(f"Initialized ZFSManager with perturbation scale: {self.perturbation_scale}, cell size: {self.cell_size}")

            self._get_zfs_data(sim_folder, zfs_folder)

            pert_SI = self.perturbation_scale * CONSTANTS["ang_amu2SI"]
            phonon_pert = self.phonon_manager.get_phonon_pert(pert_SI)

            search_path = sim_folder / self.sub_folder

            found_valid_order = False

            if "first" in sim_folder.parent.name:
                self.zfs_tensors = self._load_zfs_perts(search_path, phonon_pert)
                found_valid_order = True

            if "second" in sim_folder.parent.name:
                self.zfs_tensors_2d = self._load_zfs_perts_2d(search_path, phonon_pert)
                found_valid_order = True

            if not found_valid_order:
                raise ValueError(f"No valid order was specified: {sim_folder.parent.name}")

            if "approx" in self.sub_folder:
                self.zfs_relaxed *= 3/2
                if "first" in sim_folder.parent.name:
                    for key in self.zfs_tensors:
                        self.zfs_tensors[key]["tensor"] *= 3/2
                else:
                    for key in self.zfs_tensors_2d:
                        self.zfs_tensors_2d[key]["tensor"] *= 3/2

            print("Succesfully loaded ZFS data from OUTCARs")
            return self.zfs_relaxed, self.zfs_tensors, self.zfs_tensors_2d, self.eigen_rotation

        elif kwargs.get("raw_data_path", None) is not None:
            raw_data = np.load(kwargs["raw_data_path"], allow_pickle=True)

            self.sub_folder = raw_data["sub_folder"]
            self.perturbation_scale = raw_data["pert_scale"]
            self.defect = raw_data["defect"]
            self.cell_size = raw_data["cell_size"]

            self.zfs_relaxed = raw_data["zfs_relaxed"]
            self.eigen_rotation = raw_data["eigen_rotation"]

            self.zfs_tensors = raw_data["zfs_tensors"][()]
            self.zfs_tensors_2d = raw_data["zfs_tensors_2d"][()]

            print("\nRelaxed ZFS tensor:")
            print(self.zfs_relaxed,"\n")

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



    def process_first_order_perturbations(self, output_filename=None):
        if None in self.zfs_tensors:
            raise ValueError("First order ZFS data not available. Make sure to load the data first with load_outcar_zfs_data()")
        results = []

        pert_SI = self.perturbation_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.phonon_manager.get_phonon_pert(pert_SI)

        zfs_derivs, V_0_0, V_p_m, V_0_pm = self._calc_derivative()

        save_name = f"{output_filename}.npz" if output_filename else f"derivatives/{self.defect}_{self.cell_size}_zfs_coefficients_{self.sub_folder}_{self.perturbation_scale}_.npz"

        save_name = self.save_data(save_name, zfs_derivs=zfs_derivs, V_0_0=V_0_0, V_p_m=V_p_m,
                                    V_0_pm=V_0_pm, freqs=phonon_pert["freqs"], sym=phonon_pert["sym"], ipr=phonon_pert["ipr"])

        results.append(save_name)
        return results


    def process_second_order_perturbations(self, zfs_1d_derivs_file, output_filename=None):
        results = []
        if None in self.zfs_tensors_2d:
           raise ValueError("Second order ZFS data not available. Make sure to load the data first with load_outcar_zfs_data()")

        save_name = f"{output_filename}.npz" if output_filename else f"derivatives/{self.defect}_{self.cell_size}_zfs2d_coefficients_{self.sub_folder}_{self.perturbation_scale}_.npz"

        # Load the pre-calculated first order derivatives to optimize compute
        first_order_data = np.load(zfs_1d_derivs_file)
        zfs_1d_derivs = first_order_data["zfs_derivs"]

        pert_SI = self.perturbation_scale * CONSTANTS["ang_amu2SI"]
        phonon_pert = self.phonon_manager.get_phonon_pert(pert_SI)

        zfs_2nd_derivs, V_0_0_2nd, V_p_m_2nd, V_0_pm_2nd = self._calc_second_order_derivatives(zfs_1d_derivs)

        save_name = self.save_data(save_name, second_order=True, zfs_derivs=zfs_2nd_derivs, V_0_0=V_0_0_2nd, V_p_m=V_p_m_2nd,
                                    V_0_pm=V_0_pm_2nd, freqs=phonon_pert["freqs"], sym=phonon_pert["sym"], ipr=phonon_pert["ipr"])

        results.append(save_name)
        return results


    def _load_zfs_perts(self, search_path, phonon_pert):
        print("Reading ZFS tensors from: ", search_path)
        outcars = search_path.glob("**/OUTCAR")
        eigen_rotation_t = self.eigen_rotation.T

        total_modes = len(phonon_pert["idx"])
        print(f"Expecting up to {total_modes} OUTCAR files.")
        
        worker_task = partial(_process_outcar_worker_1d, eigen_rotation=self.eigen_rotation, eigen_rotation_t=eigen_rotation_t)

        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(worker_task, outcars), total=total_modes, desc="Processing OUTCAR files"))

        # Filter out any None results
        zfs_tensors = {r[0]: {"tensor": r[1]*CONSTANTS["MHz2J"], "symmetry": phonon_pert["sym"][r[0]], "pert": phonon_pert["q"][r[0]] } for r in results if r is not None}

        print("Number of tensors: ", len(zfs_tensors.keys()))
        return zfs_tensors


    def _load_zfs_perts_2d(self, search_path, phonon_pert):
        # Implementation to read the 2D grid of OUTCAR files
        print("Reading ZFS tensors from: ", search_path)

        zfs_tensors = {}
        total_modes = int((len(phonon_pert["idx"])**2) / 2)
        print(f"Expecting up to {total_modes} OUTCAR files for 2D perturbations.")
        outcars = search_path.glob("**/OUTCAR")
        eigen_rotation_t = self.eigen_rotation.T

        worker_task = partial(_process_outcar_worker_2d, eigen_rotation=self.eigen_rotation, eigen_rotation_t=eigen_rotation_t)

        with ProcessPoolExecutor(max_workers=4) as executor:
            results = list(tqdm(executor.map(worker_task, outcars), total=total_modes, desc="Processing OUTCAR files"))

        # Filter out any None results
        zfs_tensors = {r[0]: {"tensor": r[1]*CONSTANTS["MHz2J"], "symmetry": (phonon_pert["sym"][r[0][0]], phonon_pert["sym"][r[0][1]]), "pert": (phonon_pert["q"][r[0][0]], phonon_pert["q"][r[0][1]]) } for r in results if r is not None}

        num_entries = len(zfs_tensors.keys())

        print(f"Total stored pairs: {num_entries}")

        return zfs_tensors


    def _get_zfs_data(self, sim_folder, zfs_folder):
        if self.zfs_relaxed is None and self.eigen_rotation is None:
            main_path = sim_folder.parent.parent

            zfs_relaxed = read_zfs_tensor(str(main_path / zfs_folder / "OUTCAR"))
            if zfs_relaxed is None:
                raise ValueError("Relaxed ZFS tensor not found. Ensure the OUTCAR file exists and contains the ZFS tensor data.")
            eigen_rotation = zfs_relaxed["eigenvectors"]
            self.zfs_relaxed = np.diag(zfs_relaxed["D_diag"])*CONSTANTS["MHz2J"]
            self.eigen_rotation = np.array(eigen_rotation)

            print("\nRelaxed ZFS tensor:")
            print(self.zfs_relaxed,"\n")

        return self.zfs_relaxed, self.eigen_rotation


    def _debug_derivs(self, dD, q, symmetry, idx, V_0_0, V_0_pm, V_p_m):
        max_val = np.max(np.abs(dD))

        col_width = 12
        total_width = (col_width * 3) + 10
        double_line = "=" * total_width

        print(f"\n{double_line}\n DEBUG DERIVATIVES\n{'-' * total_width}")
        print((f" Mode: {idx}\n"
               f" Symmetry: {symmetry}\n"
               f" Perturbation: {MathUtils.fmt(q)}\n"
               f"{double_line}\n"
               f" Max Value: {max_val/CONSTANTS['MHz2J']:<10.6f}\n"
               f" V_0_0: {V_0_0/CONSTANTS['MHz2J']}\n"
               f" V_0_pm: {V_0_pm/CONSTANTS['MHz2J']}\n"
               f" V_p_m: {V_p_m/CONSTANTS['MHz2J']}"))
        print(" Tensor Structure (3x3):\n")
        for row in dD:
            formatted_row = "  ".join(f"{val/CONSTANTS['MHz2J']:>{col_width}.6f}" for val in row)
            print(f"  [ {formatted_row} ]")
        print(f"{double_line}\n")


    def _calc_derivative(self):
        print("Calculating derivatives...")
        n_modes = self.phonon_manager.nmodes
        symmetry_factor = self._get_symmetry_factor(n_modes, len(self.zfs_tensors.keys()))

        zfs_deriv = np.zeros(shape=(n_modes, 3, 3))
        V_0_0 = np.zeros(shape=n_modes)
        V_0_pm = np.zeros(shape=n_modes)
        V_p_m = np.zeros(shape=n_modes)

        for i, item in self.zfs_tensors.items():
            D_i = item["tensor"]
            q = item["pert"]
            if q is None:
                continue
            sym = item["symmetry"]

            dD = (D_i - self.zfs_relaxed)
            
            self._check_symmetry(dD, sym, i)

            zfs_deriv[i] = dD / q

            trace_in_plane = dD[0, 0] + dD[1, 1]
            diff_in_plane = dD[0, 0] - dD[1, 1]
            off_diag_in_plane = dD[0, 1]

            if sym == "A1":
                V_0_0[i] = (np.abs(dD[2, 2] - 0.5 * trace_in_plane) / q) / 3
            elif sym in ["Ex", "Ey"]:
                V_p_m[i] = symmetry_factor*(0.5 * np.sqrt(diff_in_plane**2 + 2 * off_diag_in_plane**2) / q)
                V_0_pm[i] = symmetry_factor*(np.sqrt(dD[0, 2]**2 + dD[1, 2]**2) / q) / np.sqrt(2)

            if self.debug:
                self._debug_derivs(dD, q, sym, i+1, V_0_0, V_0_pm, V_p_m)

        print("Symmetry adjusted coefficients: ")
        print("V_00: ", np.sum(V_0_0 > 0))
        print("V_pm: ", np.sum(V_p_m > 0))
        print("V_0pm: ", np.sum(V_0_pm > 0))
        print("Number of tensors: ", zfs_deriv.shape)

        return zfs_deriv, V_0_0, V_p_m, V_0_pm

    @staticmethod
    def _check_symmetry(d_tensor: np.ndarray, symmetry: str | tuple[str, str], idx: int) -> None:
        """
        Validate ZFS tensor against C3v constraints using scale-aware tolerances.
        """
        sym_prod = (
            MathUtils.calc_symmetry(*symmetry) 
            if isinstance(symmetry, tuple) 
            else [symmetry]
        )

        if sym_prod == ["A1"]:
            diag = np.diag(d_tensor)
            d_xx, d_yy, d_zz = diag
            
            tensor_scale = np.max(np.abs(diag))
            
            rt, at = 1e0, 1e-4
            dyn_tol = rt * tensor_scale + at

            is_axial = np.abs(d_xx - d_yy) <= dyn_tol
            is_traceless_axial = np.abs(2 * d_xx + d_zz) <= dyn_tol
            
            off_diag_mask = ~np.eye(3, dtype=bool)
            off_diags_zero = np.all(np.abs(d_tensor[off_diag_mask]) <= dyn_tol)

            if not (is_axial and is_traceless_axial and off_diags_zero):
                print(f"\nWarning: Symmetry mismatch at index {idx} with symmetry {sym_prod}")
                print(f"Threshold: {dyn_tol:.2e} | Scale: {tensor_scale:.2f}")
                print(d_tensor)


    def _calc_second_order_derivatives(self, zfs_1d_derivs):
        print("Calculating derivatives...")
        n_modes = self.phonon_manager.nmodes
        symmetry_factor = 1 # self._get_symmetry_factor(n_modes, len(self.zfs_tensors_2d.keys()))

        zfs_2nd_derivs = np.zeros((n_modes, n_modes, 3, 3))
        V_0_0_2nd = np.zeros((n_modes, n_modes))
        V_p_m_2nd = np.zeros((n_modes, n_modes))
        V_0_pm_2nd = np.zeros((n_modes, n_modes))

        for (i, j), item in self.zfs_tensors_2d.items():
            (q_i, q_j) = item["pert"]

            if q_i is None or q_j is None:
                continue

            if q_i != q_j:
                continue

            dD_qi = zfs_1d_derivs[i]

            dD_qj = zfs_1d_derivs[j]

            (sym_i, sym_j) = item["symmetry"]

            sym = MathUtils.calc_symmetry(sym_i, sym_j)

            D_qi_qj = item["tensor"]

            d2D_dqidqj = (D_qi_qj - self.zfs_relaxed)/(q_i*q_j) - dD_qi/q_j - dD_qj/q_i

            self._check_symmetry(d2D_dqidqj, (sym_i, sym_j), (i, j))

            zfs_2nd_derivs[i, j] = d2D_dqidqj
            zfs_2nd_derivs[j, i] = d2D_dqidqj

            trace_in_plane = d2D_dqidqj[0, 0] + d2D_dqidqj[1, 1]
            diff_in_plane = d2D_dqidqj[0, 0] - d2D_dqidqj[1, 1]
            off_diag_in_plane = d2D_dqidqj[0, 1]


            if "A1" in sym:
                V_0_0_2nd[i, j] = np.abs(d2D_dqidqj[2, 2] - 0.5 * trace_in_plane) / 3
                if "E" in sym:
                    V_0_0_2nd[i, j] *= symmetry_factor

            if "E" in sym:
                V_p_m_2nd[i, j] = symmetry_factor * (0.5 * np.sqrt(diff_in_plane**2 + 2 * off_diag_in_plane**2))
                V_0_pm_2nd[i, j] = symmetry_factor * (np.sqrt(d2D_dqidqj[0, 2]**2 + d2D_dqidqj[1, 2]**2)) / np.sqrt(2)

            V_0_0_2nd[j, i] = V_0_0_2nd[i, j]
            V_p_m_2nd[j, i] = V_p_m_2nd[i, j]
            V_0_pm_2nd[j, i] = V_0_pm_2nd[i, j]

            if self.debug:
                self._debug_derivs(d2D_dqidqj,
                                item["pert"],
                                item["symmetry"],
                                (i, j),
                                V_0_0_2nd[i, j],
                                V_0_pm_2nd[i, j],
                                V_p_m_2nd[i, j])


        print("Symmetry adjusted coefficients: ")
        print("V_00: ", np.sum(V_0_0_2nd > 0))
        print("V_pm: ", np.sum(V_p_m_2nd > 0))
        print("V_0pm: ", np.sum(V_0_pm_2nd > 0))
        print("Number of tensors: ", zfs_2nd_derivs.shape)

        plt.plot([item["pert"] for (i, j), item in self.zfs_tensors_2d.items() if i == j])
        #plt.plot(V_0_0_2nd.diagonal()/CONSTANTS["MHz2J"], color="red", label=r"$V_{00}^{(2)}$")
        #plt.plot(V_p_m_2nd.diagonal()/CONSTANTS["MHz2J"], color="blue", label=r"$V_{00}^{(2)}$")
        plt.xlabel("Mode")
        plt.ylabel("Perturbation")
        plt.title("ZFS Perturbations")
        plt.show()
        
        return zfs_2nd_derivs, V_0_0_2nd, V_p_m_2nd, V_0_pm_2nd

    def _get_symmetry_factor(self, n_modes, nr_zfs_tensors):
        total_modes = n_modes
        print("Tensor data size: ", nr_zfs_tensors, " out of ", total_modes, " total modes")
        if nr_zfs_tensors == total_modes:
            print("Using all phonon modes")
            symmetry_factor = 1
        else:
            print("Excluding degenerate E modes")
            symmetry_factor = 2

        return symmetry_factor
