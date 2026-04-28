from dataclasses import dataclass

from scipy import constants as Cn
import numpy as np
from pymatgen.core import Structure
from pathlib import Path
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from beyblade.constants import CONSTANTS
from beyblade.utils import MathUtils

@dataclass
class PhononManager:
    def __init__(self, data_path=None):
        self.data: dict | None = None
        self.symmetry_data: dict | None = None
        self.nmodes: int = 0
        self.cell_size: int = 0

        if data_path:
            self.load_data(data_path)

    def read_yaml(self, yaml_file):
        with open(yaml_file, "r") as f:
            raw_data = yaml.load(f, Loader=Loader)

        phonon_data = raw_data["phonon"][0]
        nphonon = len(phonon_data["band"])
        nlattice = len(phonon_data["band"][0]["eigenvector"])
        mode_freqs = np.array([d["frequency"] for d in phonon_data["band"]]) * CONSTANTS["THz2meV"]

        mode_eigenvectors = np.zeros((nphonon, nlattice, 3))
        for i in range(nphonon):
            mode_eigenvectors[i] = np.reshape(np.array([np.array(d)[:, 0] for d in phonon_data["band"][i]["eigenvector"]]), (nlattice, 3))

        phonon_info: dict = {
            "freqs": np.array(mode_freqs),
            "eigs": np.array(mode_eigenvectors),
            "n_atoms": nlattice,
            "n_modes": nphonon
        }

        if "points" in raw_data.keys():
            atom_types = np.array([d["symbol"] for d in raw_data["points"]])
            lattice_points = np.array([d["coordinates"] for d in raw_data["points"]])
            lattice_vecs = [np.array(d) for d in raw_data["lattice"]]

            phonon_info += {
                "atom_types": atom_types,
                "atoms": lattice_points,
                "lattice": np.array(lattice_vecs),
            }
        
        return phonon_info

    def read_structure_data(self, poscar_file):
        structure = Structure.from_file(poscar_file)
        
        structure_info = {
            "atoms": structure.frac_coords,
            "atom_symbols": np.array([site.species_string for site in structure]),
            "lattice": structure.lattice.matrix,
            "n_atoms": len(structure)
        }
        
        return structure_info

    def load_all_data(self, yaml_path, poscar_path="POSCAR"):
        phonon_data = self.read_yaml(yaml_path)
        if "atoms" not in phonon_data.keys():
            print("No atom data loaded from yaml")
            print("Trying to read data from: ", poscar_path)
            try:
                struct_data = self.read_structure_data(poscar_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"POSCAR file not found at: {poscar_path}")
        
            if struct_data["n_atoms"] != phonon_data["n_atoms"]:
                raise ValueError(f"Geometry mismatch: {struct_data['n_atoms']} atoms in POSCAR "
                                f"vs {phonon_data['n_atoms']} atoms in YAML.")
            
            self.data = {
                "atoms": struct_data["atoms"],
                "atom_symbols": struct_data["atom_symbols"],
                "lattice": struct_data["lattice"],
                "freqs": phonon_data["freqs"],
                "eigs": phonon_data["eigs"]
            }
        else:
            self.data = phonon_data

    def load_data(self, filepath):
        path = Path(filepath)
        if path.suffix == ".yaml":
            self.load_all_data(path)
        elif path.suffix == ".npz":
            self.data = dict(np.load(path, allow_pickle=True))
            if self.data is None:
                raise ValueError("Failed to load data from .npz file.")
        else:
            raise ValueError("Unsupported file format. Use .yaml or .npz")

        self.nmodes = self.data["freqs"].shape[0]
        self.cell_size = self.data["atoms"].shape[0]
        self.analyze_c3v_symmetry()
        if self.symmetry_data is None:
            raise ValueError("No symmetry data available.")
        print("Loaded phonon data from .npz file.")
        print("n Phonons: ", self.nmodes)
        print("n Atoms: ", self.cell_size)

    def save_data(self, filename="phonon_data.npz"):
        if self.data is not None:
            np.savez(filename, **self.data)

    def analyze_c3v_symmetry(self):
        if self.data is None:
            raise ValueError("No phonon data loaded.")

        frac_atoms = self.data['atoms']
        symbols = self.data['atom_symbols']
        freqs = self.data['freqs']
        eigs = self.data['eigs']
        lattice = self.data['lattice']

        R_C3 = MathUtils.get_c3_111_matrix()
        R_sv = MathUtils.get_sv_matrix([1, -1, 0])

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

        results_dict = {'idx': [], 'freqs': [], 'sym': [], 'char_C3': [], 'char_sv': []}

        for m in range(num_modes):
            eig = eigs[m]
            char_C3 = np.trace(np.dot(eig[map_C3], R_C3 @ eig.T))
            char_sv = np.trace(np.dot(eig[map_sv], R_sv @ eig.T))
            
            if char_C3 > 0.8:
                sym = "A1" if char_sv > 0.0 else "A2"
            else:
                sym = "Ex" if char_sv > 0.0 else "Ey"

            results_dict['idx'].append(m)
            results_dict['freqs'].append(freqs[m])
            results_dict['sym'].append(sym)
            results_dict['char_C3'].append(char_C3)
            results_dict['char_sv'].append(char_sv)

        sort_indices = np.argsort(results_dict['idx'])

        output_dict = {key: np.array(value)[sort_indices] for key, value in results_dict.items()}

        self.symmetry_data = output_dict

    def filter_sym_pairs(self, save=True, debug=False, tol=0.01):
        if self.symmetry_data is None:
            raise ValueError("No symmetry data available. Call analyze_c3v_symmetry() first.")
        if self.data is None:
            raise ValueError("No phonon data loaded.")
        
        skip_indices = set()
        num_modes = len(self.symmetry_data["freqs"])

        for i in range(num_modes):
            if debug:
                print("Symmetry: ", self.symmetry_data["sym"][i], "\nIndex: ", i+1)

            if i in skip_indices:
                continue

            if "E" in self.symmetry_data['sym'][i]:
                # Check for nearly degenerate partner
                for j in range(i + 1, num_modes):
                    if j not in skip_indices and "E" in self.symmetry_data['sym'][j]:
                        if abs(self.symmetry_data['freqs'][j] - self.symmetry_data['freqs'][i]) < tol:
                            skip_idx = j if self.symmetry_data["sym"][i] == "Ex" else i
                            skip_indices.add(skip_idx)
                            break

        skip_indices = sorted(list(skip_indices))

        mask = [i not in skip_indices for i in range(self.nmodes)]

        phonons = {key: value[mask] if value.shape[0] == self.nmodes else value for key, value in self.data.items()}
        phonon_symmetries = {key: value[mask] for key, value in self.symmetry_data.items()}

        nmodes = phonon_symmetries["sym"].shape[0]

        if not save:
            self.nmodes = nmodes
            self.data = phonons
            self.symmetry_data = phonon_symmetries
        else:
            print(f"Filtered out {len(skip_indices)} modes, remaining modes: {nmodes}")
            filename = f"phonon_data_sym_n{nmodes}.npz"

            print(f"Saving file in: {filename}")

            np.savez(filename, **phonons, sym=phonon_symmetries["sym"], ipr=MathUtils.calc_ipr(phonons), idx=phonon_symmetries["idx"])

    def get_phonon_pert(self, perturbation_scale):
        phonon_pert = {}
        if self.data is None:
            raise ValueError("No phonon data loaded.")
        if self.symmetry_data is None:
            raise ValueError("No symmetry data loaded.")

        mask = self.data["freqs"] > 0

        if self.data.get("sym") is not None:
            phonon_pert["sym"] = self.data["sym"][mask]
            phonon_pert["idx"] = self.data["idx"][mask]
        else:
            print("No symmetry data, find symmetries.")
            phonon_pert["sym"] = self.symmetry_data["sym"][mask]
            phonon_pert["idx"] = self.symmetry_data["idx"][mask]

        Q = [np.sqrt(np.sum(mode**2)) for mode in self.data["eigs"]]

        if not np.isclose(Q, 1).all():
          raise ValueError("Phonon modes not normalized correctly.")

        phonon_pert["eigs"] = np.array([
            perturbation_scale * mode * np.sqrt(2 * CONSTANTS["meV2rads"] * freq / Cn.hbar) if freq > 0 else None
            for mode, freq in zip(Q, self.data["freqs"])
        ])[mask]

        phonon_pert["freqs"] = self.data["freqs"][mask]
        phonon_pert["ipr"] = MathUtils.calc_ipr(self.data)[mask]

        return phonon_pert
