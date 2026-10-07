from __future__ import annotations

import re
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from pymatgen.core import Structure

from beyblade.constants import CONSTANTS
from beyblade.models import PerturbationEntry, PhononSpectrum, RawZFSData, ZFSTensor

ZFS_REGEX = re.compile(
    r"Spin-spin contribution to zero-field splitting tensor \(MHz\)\s*-+\s*D_xx\s+D_yy\s+D_zz\s+D_xy\s+D_xz\s+D_yz\s*-+\s*([\s\d\.\-]+?)(?=\s*-{3,})",
    re.MULTILINE | re.DOTALL,
)

ENERGY_REGEX = re.compile(
    r"free  energy   TOTEN  =\s+([\-\d\.]+)\s+eV",
    re.MULTILINE,
)


def parse_outcar_zfs(outcar_path: str | Path) -> ZFSTensor | None:
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
        matrix = np.array(
            [
                [D_xx, D_xy, D_xz],
                [D_xy, D_yy, D_yz],
                [D_xz, D_yz, D_zz],
            ],
            dtype=float,
        )

        return ZFSTensor(matrix=matrix, unit="MHz")
    except (ValueError, IndexError):
        return None


def parse_outcar_energy(outcar_path: str | Path) -> float | None:
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


def parse_phonopy_yaml(yaml_path: str | Path, poscar_path: str | Path | None = None) -> PhononSpectrum:
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
        mode_eigenvectors[i] = np.array([[comp[0] for comp in atom] for atom in phonon_data["band"][i]["eigenvector"]])

    # Structure data
    if poscar_path and Path(poscar_path).is_file():
        struct = Structure.from_file(str(poscar_path))
        frac_coords = struct.frac_coords
        symbols = [site.specie.symbol for site in struct]
        masses = np.array([site.specie.atomic_mass for site in struct], dtype=float)
        lattice = struct.lattice.matrix
    else:
        points = raw_data.get("points", [])
        if not points:
            # Fall back to a POSCAR next to the phonopy file, like the old
            # PhononManager.load_all_data did.
            sibling = path.parent / "POSCAR"
            if sibling.is_file():
                struct = Structure.from_file(str(sibling))
                if len(struct) != n_lattice:
                    raise ValueError(
                        f"Geometry mismatch: {len(struct)} atoms in POSCAR "
                        f"({sibling}) vs {n_lattice} atoms in eigenvectors "
                        f"of {path}"
                    )
                print(f"Trying to read data from: {sibling}")
                return parse_phonopy_yaml(yaml_path, poscar_path=sibling)

        if points:
            frac_coords = np.array([p["coordinates"] for p in points], dtype=float)
            symbols = [p.get("symbol", "X") for p in points]
            masses = np.array([p.get("mass", 1.0) for p in points], dtype=float)
        else:
            frac_coords = np.zeros((n_lattice, 3), dtype=float)
            symbols = ["X"] * n_lattice
            masses = np.ones(n_lattice, dtype=float)
            warnings.warn(
                f"No structure data in {path} and no POSCAR found in "
                f"{path.parent}: falling back to degenerate defaults "
                f"(identity lattice, all atoms at origin, symbol X, mass 1.0). "
                f"Symmetry analysis will fail; place a matching POSCAR next "
                f"to the file or run write_full_phonopy_yaml().",
                stacklevel=2,
            )

        lattice = np.array(raw_data.get("lattice", np.eye(3)), dtype=float)

    return PhononSpectrum(
        frequencies_mev=mode_freqs_mev,
        eigenvectors=mode_eigenvectors,
        atom_frac_coords=frac_coords,
        atom_symbols=symbols,
        atomic_masses=masses,
        lattice=lattice,
    )


def write_full_phonopy_yaml(
    yaml_path: str | Path,
    poscar_path: str | Path | None = None,
    out_path: str | Path | None = None,
) -> Path | None:
    """Write a phonopy yaml with structure data filled in from a POSCAR.

    If ``yaml_path`` already contains structure data (``points``), nothing is
    written and ``None`` is returned. Otherwise the structure is taken from
    ``poscar_path`` (default: a ``POSCAR`` next to the yaml) — the POSCAR atom
    count must match the eigenvector dimension or ``ValueError`` is raised —
    injected verbatim as top-level ``lattice`` and ``points`` sections —
    the original file text is copied unchanged and the structure block is
    appended at the end, so the phonon data keeps its original formatting —
    and the result saved to ``out_path`` (default: ``<yaml stem>_full.yaml``
    next to the input). Returns the output path.
    """
    path = Path(yaml_path)
    with path.open("r", encoding="utf-8") as f:
        raw_data = yaml.load(f, Loader=Loader)

    if raw_data.get("points"):
        return None

    if poscar_path is None:
        sibling = path.parent / "POSCAR"
        poscar_path = sibling if sibling.is_file() else None
    if poscar_path is None or not Path(poscar_path).is_file():
        return None

    struct = Structure.from_file(str(poscar_path))
    phonon_block = raw_data["phonon"][0]
    n_atoms = len(phonon_block["band"][0]["eigenvector"])
    if len(struct) != n_atoms:
        raise ValueError(
            f"Geometry mismatch: {len(struct)} atoms in POSCAR "
            f"({poscar_path}) vs {n_atoms} atoms in eigenvectors of {path}"
        )

    lines = ["lattice:"]
    for row in struct.lattice.matrix:
        lines.append("- [" + ", ".join(f"{v:.12f}" for v in row) + "]")
    lines.append("points:")
    for site in struct:
        c = ", ".join(f"{v:.12f}" for v in site.frac_coords)
        lines.append(f"- coordinates: [{c}]")
        lines.append(f"  symbol: {site.specie.symbol}")
        lines.append(f"  mass: {site.specie.atomic_mass:.6f}")

    out = Path(out_path) if out_path else path.with_name(f"{path.stem}_full.yaml")
    with out.open("w", encoding="utf-8") as f:
        shutil.copyfileobj(path.open("r", encoding="utf-8"), f)
        f.write("\n")
        f.write("\n".join(lines))
        f.write("\n")
    print(f"wrote {out} (structure data from {poscar_path})")
    return out


def parse_phonon_data(
    path: str | Path,
    poscar_path: str | Path | None = None,
) -> PhononSpectrum:
    """Load a PhononSpectrum from any supported phonon file format.

    Dispatches on the file extension: ``.yaml``/``.yml`` goes to
    :func:`parse_phonopy_yaml`, everything else (``.npz``) to
    :func:`parse_phonon_npz`. Raises ``ValueError`` for unsupported
    extensions.
    """
    p = Path(path)
    if p.suffix in (".yaml", ".yml"):
        return parse_phonopy_yaml(p, poscar_path=poscar_path)
    if p.suffix == ".npz":
        return parse_phonon_npz(p)
    raise ValueError(f"Unsupported phonon file format: '{p.suffix}' (path: {p}). Expected .yaml, .yml or .npz")


def parse_phonon_npz(npz_path: str | Path) -> PhononSpectrum:
    """Loads a precomputed PhononSpectrum from a .npz file using PhononSpectrum.load."""
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"Phonon npz file not found: {path}")

    # Fall back to legacy keys if needed
    data = np.load(path, allow_pickle=True)
    if "frequencies" in data or "frequencies_mev" in data:
        try:
            return PhononSpectrum.load(path)
        except Exception:
            pass

    freqs = data["freqs"] if "freqs" in data else data["frequencies_mev"]
    eigs = data["eigs"] if "eigs" in data else data["eigenvectors"]
    atoms = data["atoms"] if "atoms" in data else data["atom_frac_coords"]
    symbols = list(data["atom_symbols"])
    if "masses" in data:
        masses = data["masses"]
    elif "atomic_masses" in data:
        masses = data["atomic_masses"]
    else:
        try:
            from pymatgen.core import Element

            masses = np.array([Element(s).atomic_mass for s in symbols], dtype=float)
        except Exception:
            masses = np.ones(len(symbols), dtype=float)
    lattice = data["lattice"]

    symmetries = list(data["symmetries"]) if "symmetries" in data else (list(data["sym"]) if "sym" in data else None)
    iprs = data["iprs"] if "iprs" in data else (data["ipr"] if "ipr" in data else None)
    # np.savez(..., iprs=None) stores a 0-d object array: not None but not a
    # usable per-mode array either. Normalize to None when scalar/empty.
    if iprs is not None:
        iprs = np.asarray(iprs)
        if iprs.ndim == 0 or iprs.size == 0:
            iprs = None
    e_pair_complete = (
        list(bool(x) for x in data["e_pair_complete"])
        if "e_pair_complete" in data and data["e_pair_complete"].ndim > 0
        else None
    )
    # Legacy files store the original DFT-run mode indices under "idx"; the spectrum
    # owns mode identity, so this must not be dropped on load.
    original_indices = None
    if "original_indices" in data and data["original_indices"] is not None:
        original_indices = np.asarray(data["original_indices"], dtype=int)
    elif "idx" in data and data["idx"] is not None:
        # idx is 0-based into the full 3N-mode run (verified: full_freqs[idx] ==
        # sym_freqs and all symmetry labels agree at these indices for NV_512).
        original_indices = np.asarray(data["idx"], dtype=int)

    return PhononSpectrum(
        frequencies_mev=np.asarray(freqs, dtype=float),
        eigenvectors=np.asarray(eigs, dtype=float),
        atom_frac_coords=np.asarray(atoms, dtype=float),
        atom_symbols=symbols,
        atomic_masses=np.asarray(masses, dtype=float),
        lattice=np.asarray(lattice, dtype=float),
        symmetries=symmetries,
        iprs=np.asarray(iprs, dtype=float) if iprs is not None else None,
        e_pair_complete=e_pair_complete,
        original_indices=original_indices,
    )


def save_phonon_npz(spectrum: PhononSpectrum, out_path: str | Path) -> str:
    """Saves a PhononSpectrum object to a .npz archive with explicit unit tracking."""
    return spectrum.save(out_path)


def _worker_parse_outcar_1d(outcar_file: Path) -> tuple[int, ZFSTensor, float | None] | None:
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


def _worker_parse_outcar_2d(outcar_file: Path) -> tuple[tuple[int, int], ZFSTensor, float | None] | None:
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
    directory: str | Path,
    order: int = 1,
    amplitude: float | tuple[float, float] = 1.0,
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
    sim_folder: str | Path,
    max_workers: int = 4,
    order: int | None = None,
    pert_scale: float | None = None,
    defect: str | None = None,
    cell_size: int | None = None,
    *,
    calc_method: str | None = None,
    zfs_folder: str | None = None,
) -> RawZFSData:
    """
    Parses an entire simulation directory structure:
    - Extracts defect name, cell size, and perturbation scale from directory paths
    - Reads the unperturbed ground-state ZFS tensor
    - Reads all 1D or 2D perturbed calculations

    Explicit overrides (order, pert_scale, defect, cell_size) take precedence over
    values inferred from the directory path.
    """
    sim_path = Path(sim_folder)
    if not sim_path.is_dir():
        raise FileNotFoundError(f"Simulation folder not found: {sim_path}")

    if calc_method in ("all", "all_bands"):
        calc_method, default_zfs = ("all_bands", "ZFS_hyp")
    elif calc_method in ("approx", "defect_band_approx"):
        calc_method, default_zfs = ("defect_band_approx", "ZFS_occup")
    else:
        raise ValueError(f"{calc_method} is mot a valid zfs calculation method.")

    final_zfs_folder = zfs_folder or default_zfs

    if "pert" not in sim_path.name:
        raise ValueError("Perturbation scale not found in folder name (expected e.g. 'pert_0.01').")

    defect = defect or sim_path.parent.parent.name.split("_")[0]
    cell_size = cell_size or int(sim_path.parent.parent.name.split("_")[-1])
    pert_scale = pert_scale if pert_scale is not None else float(sim_path.name.split("_")[1])

    # Unperturbed relaxed ground state
    relaxed_outcar = sim_path.parent.parent / final_zfs_folder / "OUTCAR"
    ground_state_zfs = parse_outcar_zfs(relaxed_outcar)
    if ground_state_zfs is None:
        raise ValueError(f"Relaxed ZFS tensor not found in: {relaxed_outcar}")

    search_path = sim_path / calc_method
    first_order = {}
    second_order = {}

    if order is None:
        # Infer from parent folder (e.g. 'first_order', 'second_order')
        order = 1 if "first" in sim_path.parent.name else 2 if "second" in sim_path.parent.name else None

    if order == 1:
        first_order = parse_perturbation_directory(search_path, order=1, amplitude=pert_scale, max_workers=max_workers)
    elif order == 2:
        second_order = parse_perturbation_directory(
            search_path, order=2, amplitude=(pert_scale, pert_scale), max_workers=max_workers
        )

    return RawZFSData(
        defect=defect,
        cell_size=cell_size,
        pert_scale=pert_scale,
        calc_method=calc_method,
        ground_state_zfs=ground_state_zfs,
        order=order,
        first_order=first_order,
        second_order=second_order,
        metadata={"calc_method": calc_method, "sim_path": str(sim_path)},
    )


def parse_zfs_dataset_npz(raw_paths: list[str | Path] | tuple[str | Path, ...]) -> RawZFSData:
    """
    Loads raw 1D and 2D perturbation data from precomputed .npz files using RawZFSData.load.
    """
    return RawZFSData.load(raw_paths)
