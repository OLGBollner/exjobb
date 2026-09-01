from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from pymatgen.core import Structure

from beyblade.constants import CONSTANTS
from beyblade.models import ZFSTensor, PhononSpectrum, PerturbationEntry, RawZFSData


ZFS_REGEX = re.compile(
    r"Spin-spin contribution to zero-field splitting tensor \(MHz\)\s*-+\s*D_xx\s+D_yy\s+D_zz\s+D_xy\s+D_xz\s+D_yz\s*-+\s*([\s\d\.\-]+?)(?=\s*-{3,})",
    re.MULTILINE | re.DOTALL,
)

ENERGY_REGEX = re.compile(
    r"free  energy   TOTEN  =\s+([\-\d\.]+)\s+eV",
    re.MULTILINE,
)


def parse_outcar_zfs(outcar_path: Union[str, Path]) -> Optional[ZFSTensor]:
    """
    Parses the dipole-dipole spin-spin ZFS tensor from a VASP OUTCAR file.
    Returns ZFSTensor in MHz, or None if not found.
    """
    path = Path(outcar_path)
    if not path.is_file():
        return None

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    match = ZFS_REGEX.search(content)
    if not match:
        return None

    try:
        raw_values = match.group(1).split()
        values = [float(v) for v in raw_values]
        if len(values) < 6:
            return None

        # VASP outputs: D_xx, D_yy, D_zz, D_xy, D_xz, D_yz
        D_xx, D_yy, D_zz, D_xy, D_xz, D_yz = values[:6]
        matrix = np.array([
            [D_xx, D_xy, D_xz],
            [D_xy, D_yy, D_yz],
            [D_xz, D_yz, D_zz],
        ], dtype=float)

        return ZFSTensor(matrix=matrix, unit="MHz")
    except (ValueError, IndexError):
        return None


def parse_outcar_energy(outcar_path: Union[str, Path]) -> Optional[float]:
    """
    Parses the final free energy TOTEN (in eV) from a VASP OUTCAR file.
    """
    path = Path(outcar_path)
    if not path.is_file():
        return None

    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    matches = ENERGY_REGEX.findall(content)
    if not matches:
        return None

    try:
        return float(matches[-1])
    except ValueError:
        return None


def parse_phonopy_yaml(yaml_path: Union[str, Path], poscar_path: Optional[Union[str, Path]] = None) -> PhononSpectrum:
    """
    Parses phonon vibrational frequencies, eigenvectors, and structure from a phonopy.yaml file.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"phonopy file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw_data = yaml.load(f, Loader=Loader)

    # Phonon modes
    phonon_data = raw_data["phonon"][0]
    n_phonon = len(phonon_data["band"])
    n_lattice = len(phonon_data["band"][0]["eigenvector"])

    # Frequencies converted to meV
    mode_freqs_mev = np.array([d["frequency"] for d in phonon_data["band"]]) * CONSTANTS["THz2meV"]

    # Eigenvectors: real part at Gamma
    mode_eigenvectors = np.zeros((n_phonon, n_lattice, 3), dtype=float)
    for i in range(n_phonon):
        mode_eigenvectors[i] = np.array(
            [[comp[0] for comp in atom] for atom in phonon_data["band"][i]["eigenvector"]]
        )

    # Structure data
    if poscar_path and Path(poscar_path).is_file():
        struct = Structure.from_file(str(poscar_path))
        frac_coords = struct.frac_coords
        symbols = [site.specie.symbol for site in struct]
        masses = np.array([site.specie.atomic_mass for site in struct], dtype=float)
        lattice = struct.lattice.matrix
    else:
        points = raw_data.get("points", [])
        if points:
            frac_coords = np.array([p["coordinates"] for p in points], dtype=float)
            symbols = [p.get("symbol", "X") for p in points]
            masses = np.array([p.get("mass", 1.0) for p in points], dtype=float)
        else:
            frac_coords = np.zeros((n_lattice, 3), dtype=float)
            symbols = ["X"] * n_lattice
            masses = np.ones(n_lattice, dtype=float)

        lattice = np.array(raw_data.get("lattice", np.eye(3)), dtype=float)

    return PhononSpectrum(
        frequencies_mev=mode_freqs_mev,
        eigenvectors=mode_eigenvectors,
        atom_frac_coords=frac_coords,
        atom_symbols=symbols,
        atomic_masses=masses,
        lattice=lattice,
    )


def parse_phonon_npz(npz_path: Union[str, Path]) -> PhononSpectrum:
    """Loads a precomputed PhononSpectrum from a .npz file."""
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Phonon npz file not found: {path}")

    data = np.load(path, allow_pickle=True)
    
    freqs = data["freqs"] if "freqs" in data else data["frequencies_mev"]
    eigs = data["eigs"] if "eigs" in data else data["eigenvectors"]
    atoms = data["atoms"] if "atoms" in data else data["atom_frac_coords"]
    symbols = list(data["atom_symbols"])
    masses = data["masses"] if "masses" in data else data["atomic_masses"]
    lattice = data["lattice"]

    symmetries = list(data["symmetries"]) if "symmetries" in data else None
    iprs = data["iprs"] if "iprs" in data else None

    return PhononSpectrum(
        frequencies_mev=np.asarray(freqs, dtype=float),
        eigenvectors=np.asarray(eigs, dtype=float),
        atom_frac_coords=np.asarray(atoms, dtype=float),
        atom_symbols=symbols,
        atomic_masses=np.asarray(masses, dtype=float),
        lattice=np.asarray(lattice, dtype=float),
        symmetries=symmetries,
        iprs=np.asarray(iprs, dtype=float) if iprs is not None else None,
    )


def save_phonon_npz(spectrum: PhononSpectrum, out_path: Union[str, Path]) -> None:
    """Saves a PhononSpectrum object to a .npz archive."""
    np.savez(
        str(out_path),
        frequencies_mev=spectrum.frequencies_mev,
        eigenvectors=spectrum.eigenvectors,
        atom_frac_coords=spectrum.atom_frac_coords,
        atom_symbols=spectrum.atom_symbols,
        atomic_masses=spectrum.atomic_masses,
        lattice=spectrum.lattice,
        symmetries=spectrum.symmetries,
        iprs=spectrum.iprs,
    )


def _worker_parse_outcar_1d(outcar_file: Path) -> Optional[tuple[int, ZFSTensor, Optional[float]]]:
    zfs = parse_outcar_zfs(outcar_file)
    if zfs is None:
        return None
    try:
        folder_name = outcar_file.parent.name
        digits = re.findall(r"\d+", folder_name)
        if not digits:
            return None
        index = int(digits[0]) - 1
        energy = parse_outcar_energy(outcar_file)
        return index, zfs, energy
    except Exception:
        return None


def _worker_parse_outcar_2d(outcar_file: Path) -> Optional[tuple[tuple[int, int], ZFSTensor, Optional[float]]]:
    zfs = parse_outcar_zfs(outcar_file)
    if zfs is None:
        return None
    try:
        folder_name = outcar_file.parent.name
        digits = re.findall(r"\d+", folder_name)
        if len(digits) < 2:
            return None
        indices = (int(digits[0]) - 1, int(digits[1]) - 1)
        energy = parse_outcar_energy(outcar_file)
        return indices, zfs, energy
    except Exception:
        return None


def parse_perturbation_directory(
    directory: Union[str, Path],
    order: int = 1,
    amplitude: Union[float, tuple[float, float]] = 1.0,
    max_workers: int = 4,
) -> dict[Any, PerturbationEntry]:
    """
    Parses all perturbed OUTCARs under a directory for 1D or 2D displacements in parallel.
    """
    dir_path = Path(directory)
    outcar_files = list(dir_path.glob("**/OUTCAR"))
    if not outcar_files:
        return {}

    worker_fn = _worker_parse_outcar_1d if order == 1 else _worker_parse_outcar_2d
    results = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(worker_fn, outcar_files):
            if res is not None:
                key, zfs, energy = res
                mode_indices = (key,) if isinstance(key, int) else key
                results[key] = PerturbationEntry(
                    order=order,
                    mode_indices=mode_indices,
                    amplitude=amplitude,
                    zfs_tensor=zfs,
                    energy=energy,
                )

    return results


def parse_zfs_simulation_dataset(
    sim_folder: Union[str, Path],
    zfs_folder: str = "relaxed",
    calc_method: str = "calc",
    max_workers: int = 4,
) -> RawZFSData:
    """
    Parses an entire simulation directory structure:
    - Extracts defect name, cell size, and perturbation scale from directory paths
    - Reads the unperturbed ground-state ZFS tensor
    - Reads all 1D or 2D perturbed calculations
    """
    sim_path = Path(sim_folder)
    if not sim_path.is_dir():
        raise FileNotFoundError(f"Simulation folder not found: {sim_path}")

    if "pert" not in sim_path.name:
        raise ValueError("Perturbation scale not found in folder name (expected e.g. 'pert_0.01').")

    defect = sim_path.parent.parent.name.split("_")[0]
    cell_size = int(sim_path.parent.parent.name.split("_")[-1])
    pert_scale = float(sim_path.name.split("_")[1])

    # Unperturbed relaxed ground state
    relaxed_outcar = sim_path.parent.parent / zfs_folder / "OUTCAR"
    ground_state_zfs = parse_outcar_zfs(relaxed_outcar)
    if ground_state_zfs is None:
        raise ValueError(f"Relaxed ZFS tensor not found in: {relaxed_outcar}")

    search_path = sim_path / calc_method
    first_order = {}
    second_order = {}

    if "first" in sim_path.parent.name:
        first_order = parse_perturbation_directory(search_path, order=1, amplitude=pert_scale, max_workers=max_workers)
    elif "second" in sim_path.parent.name:
        second_order = parse_perturbation_directory(search_path, order=2, amplitude=(pert_scale, pert_scale), max_workers=max_workers)

    return RawZFSData(
        defect=defect,
        cell_size=cell_size,
        pert_scale=pert_scale,
        ground_state_zfs=ground_state_zfs,
        first_order=first_order,
        second_order=second_order,
        metadata={"calc_method": calc_method, "sim_path": str(sim_path)},
    )


def parse_zfs_dataset_npz(raw_paths: Union[list[Union[str, Path]], tuple[Union[str, Path], ...]]) -> tuple[RawZFSData, dict[str, Any]]:
    """
    Loads raw 1D and 2D perturbation data from precomputed .npz files.
    """
    if not isinstance(raw_paths, (list, tuple)) or len(raw_paths) != 2:
        raise ValueError("raw_paths requires a list/tuple of two .npz files [1D, 2D].")

    raw_data = [np.load(f, allow_pickle=True) for f in raw_paths]
    meta_keys = ["calc_method", "pert_scale", "defect", "cell_size"]
    for key in meta_keys:
        if raw_data[0][key] != raw_data[1][key]:
            raise ValueError(f"Metadata mismatch for {key}: {raw_data[0][key]} vs {raw_data[1][key]}")

    defect = str(raw_data[0]["defect"])
    cell_size = int(raw_data[0]["cell_size"])
    pert_scale = float(raw_data[0]["pert_scale"])
    calc_method = str(raw_data[0]["calc_method"])

    zfs_relaxed_mat = raw_data[0]["zfs_relaxed"]
    eigen_rotation = raw_data[0]["eigen_rotation"]
    tensors_1d = raw_data[0]["zfs_tensors"][()]
    tensors_2d = raw_data[1]["zfs_tensors_2d"][()]

    legacy_dict = {
        "zfs_relaxed": zfs_relaxed_mat,
        "eigen_rotation": eigen_rotation,
        "zfs_tensors": tensors_1d,
        "zfs_tensors_2d": tensors_2d,
        "calc_method": calc_method,
        "pert_scale": pert_scale,
        "defect": defect,
        "cell_size": cell_size,
    }

    raw_obj = RawZFSData(
        defect=defect,
        cell_size=cell_size,
        pert_scale=pert_scale,
        ground_state_zfs=ZFSTensor(matrix=zfs_relaxed_mat / CONSTANTS["MHz2J"], unit="MHz"),
        metadata={"calc_method": calc_method},
    )

    return raw_obj, legacy_dict
