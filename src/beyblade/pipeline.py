from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Sequence, Union
import numpy as np

from beyblade.constants import CONSTANTS
from beyblade.models import RawZFSData, SpinPhononCouplingData, PhononSpectrum
from beyblade.parsers import (
    parse_phonon_npz,
    parse_phonopy_yaml,
    parse_zfs_dataset_npz,
    parse_zfs_simulation_dataset,
)
from beyblade.plotter import (
    plot_1d_spectral_functions,
    plot_t1_relaxation,
    plot_transition_rates_stacked,
)
from beyblade.relaxation_dynamics import RelaxationDynamics
from beyblade.transition_rate import TransitionRate
from beyblade.zfs_manager import ZFSManager


def get_unique_run_dir(
    output_root: Union[str, Path] = "runs",
    defect: str = "defect",
    cell_size: Union[str, int] = "",
    calc_method: str = "all_bands",
    order: int = 1,
    run_name: Optional[str] = None,
) -> Path:
    """
    Creates a unique run directory so prior results are never overwritten.
    """
    base = Path(output_root)
    base.mkdir(parents=True, exist_ok=True)

    if run_name:
        candidate = base / run_name
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        idx = 1
        while (base / f"{run_name}_{idx}").exists():
            idx += 1
        final_dir = base / f"{run_name}_{idx}"
        final_dir.mkdir(parents=True, exist_ok=True)
        return final_dir

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{defect}_{cell_size}_{calc_method}_{order}d_{now_str}"
    candidate = base / dir_name
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate

    idx = 1
    while (base / f"{dir_name}_{idx}").exists():
        idx += 1
    final_dir = base / f"{dir_name}_{idx}"
    final_dir.mkdir(parents=True, exist_ok=True)
    return final_dir


def find_default_phonon_file(sim_folder: Optional[Path]) -> Optional[Path]:
    """Search conventional defect locations for phonopy.yaml or phonon_data.npz."""
    if sim_folder is None:
        return None
    candidates = [
        sim_folder / "phonopy.yaml",
        sim_folder / "phonon_data.npz",
        sim_folder.parent / "phonopy.yaml",
        sim_folder.parent / "phonon_data.npz",
        sim_folder.parent.parent / "data" / "phonon_data.npz",
        sim_folder.parent.parent / "phonon_data.npz",
        sim_folder.parent.parent.parent / "data" / "phonon_data.npz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def find_default_phonon_file_for_raw(raw_zfs_file: Union[str, Path, Sequence[Union[str, Path]]]) -> Optional[Path]:
    """Search next to the raw ZFS .npz file(s) for phonon_data.npz / phonopy.yaml."""
    if raw_zfs_file is None:
        return None
    if isinstance(raw_zfs_file, (list, tuple)):
        if not raw_zfs_file:
            return None
        anchor = Path(raw_zfs_file[0])
    else:
        anchor = Path(raw_zfs_file)
    anchor_dir = anchor.parent
    candidates = [
        anchor_dir / "phonon_data.npz",
        anchor_dir / "phonopy.yaml",
        anchor_dir.parent / "phonon_data.npz",
        anchor_dir.parent / "phonopy.yaml",
        anchor_dir.parent.parent / "data" / "phonon_data.npz",
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def run_full_pipeline(
    *,
    sim_folder: Optional[Union[str, Path]] = None,
    raw_zfs_file: Optional[Union[str, Path, Sequence[Union[str, Path]]]] = None,
    raw_zfs_file_1d: Optional[Union[str, Path]] = None,
    raw_zfs_file_2d: Optional[Union[str, Path]] = None,
    coupling_file: Optional[Union[str, Path]] = None,
    phonon_file: Optional[Union[str, Path]] = None,
    two_phonon_file: Optional[Union[str, Path]] = None,
    calc_method: str = "all_bands",
    zfs_folder: Optional[str] = None,
    order: Optional[int] = None,
    pert_scale: Optional[float] = None,
    defect: Optional[str] = None,
    cell_size: Optional[int] = None,
    t_start: float = 0.0,
    t_end: float = 300.0,
    t_step: float = 10.0,
    temperatures: Optional[Sequence[float]] = None,
    init_state: str = "ms_0",
    output_root: Union[str, Path] = "runs",
    run_name: Optional[str] = None,
    save_plots: bool = True,
    show_plots: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """
    Executes the entire end-to-end pipeline:
      1. Ingest raw simulation (OUTCARs or raw .npz)
      2. Calculate ZFS spin-phonon couplings and derivatives
      3. Compute 1-phonon, Raman and two-phonon transition rates vs temperature
      4. Compute population relaxation dynamics and T_1 times
      5. Save all results bunched together in a dedicated unique run folder
      6. (Optional) Generate and save publication-ready figures
    """
    # ── 1. Resolve order and method ──────────────────────────────────────────
    combine_orders = raw_zfs_file_1d is not None and raw_zfs_file_2d is not None
    if combine_orders:
        order = 2 if order is None else order  # combined run is handled below
    elif order is None:
        if sim_folder is not None:
            s = str(sim_folder).lower()
            if "second_order" in s or "2d" in s or "2nd" in s:
                order = 2
            else:
                order = 1
        else:
            order = 1

    if zfs_folder is None:
        zfs_folder = "ZFS_hyp" if calc_method == "all_bands" else "ZFS_occup"

    raw_data: Optional[RawZFSData] = None
    spectrum: Optional[PhononSpectrum] = None
    coupling_data: Optional[SpinPhononCouplingData] = None

    # ── 2. Ingest Input Data ─────────────────────────────────────────────────
    if coupling_file is not None:
        coupling_data = SpinPhononCouplingData.load(coupling_file)
        defect = coupling_data.defect or defect or "defect"
        cell_size = coupling_data.cell_size or cell_size or 0
        calc_method = coupling_data.calc_method or calc_method
        order = coupling_data.order or order
    else:
        # Load phonons
        ph_from_sim = find_default_phonon_file(Path(sim_folder)) if sim_folder else None
        ph_from_raw = find_default_phonon_file_for_raw(raw_zfs_file) if raw_zfs_file is not None else None
        ph_path = Path(phonon_file) if phonon_file else (ph_from_sim or ph_from_raw)
        if ph_path is not None and ph_path.exists():
            if ph_path.suffix in [".yaml", ".yml"]:
                spectrum = parse_phonopy_yaml(ph_path)
            else:
                spectrum = parse_phonon_npz(ph_path)
        elif phonon_file is not None:
            raise FileNotFoundError(f"Phonon file not found: {phonon_file}")
        elif ph_path is not None and not ph_path.exists():
            print(f"Warning: phonon file {ph_path} not found, running without phonon data.")

        # Load raw ZFS
        if sim_folder is not None:
            raw_data = parse_zfs_simulation_dataset(
                sim_folder=sim_folder,
                calc_method=calc_method,
                zfs_folder=zfs_folder,
                order=order,
                pert_scale=pert_scale,
                defect=defect,
                cell_size=cell_size,
            )
        elif combine_orders:
            # Combine 1d + 2d raw datasets into one RawZFSData with both
            # first_order and second_order perturbation sets.
            raw_data = RawZFSData.load([str(raw_zfs_file_1d), str(raw_zfs_file_2d)])
            order = 2  # both orders present; run combined analysis
        elif raw_zfs_file is not None:
            raw_data = parse_zfs_dataset_npz(raw_zfs_file)
        else:
            raise ValueError("Must provide either sim_folder, raw_zfs_file, raw_zfs_file_1d+2d, or coupling_file.")

        defect = raw_data.defect or defect or "defect"
        cell_size = raw_data.cell_size or cell_size or 0
        calc_method = raw_data.calc_method or calc_method
        if not combine_orders:
            order = raw_data.order or order

    # ── 3. Create Unique Run Folder ──────────────────────────────────────────
    run_dir = get_unique_run_dir(
        output_root=output_root,
        defect=str(defect),
        cell_size=cell_size,
        calc_method=calc_method,
        order=order,
        run_name=run_name,
    )
    print(f"\n========================================================")
    print(f"  Initialized run in: {run_dir}")
    print(f"  Defect: {defect} | Cell: {cell_size} | Method: {calc_method} | Order: {order}")
    print(f"========================================================\n")

    # ── 4. Save Raw ZFS Data (if generated/loaded) ───────────────────────────
    raw_path = run_dir / "raw_zfs_data.npz"
    if raw_data is not None:
        raw_data.save(raw_path)
        print(f"[1/5] Saved raw ZFS data -> {raw_path.name}")

    # ── 5. Calculate Spin-Phonon Couplings ───────────────────────────────────
    coupling_path = run_dir / "spin_phonon_coupling.npz"
    if coupling_data is not None:
        coupling_data.save(coupling_path)
        print(f"[2/5] Using provided coupling data -> {coupling_path.name}")
    else:
        manager = ZFSManager(raw_data=raw_data, spectrum=spectrum, debug=debug)
        out_base = str(run_dir / "spin_phonon_coupling")
        if order == 2 and manager.zfs_tensors_2d and manager.zfs_tensors:
            # Combined 1d + 2d analysis: compute both and save in one file.
            manager.process_both_orders(output_filename=out_base)
        elif order == 1:
            manager.process_first_order_perturbations(output_filename=out_base)
        else:
            manager.process_second_order_perturbations(output_filename=out_base)
        coupling_data = SpinPhononCouplingData.load(coupling_path)
        print(f"[2/5] Computed and saved spin-phonon coupling -> {coupling_path.name}")

    # ── 6. Transition Rates Calculation ──────────────────────────────────────
    if temperatures is None:
        if t_start < 1.0:
            low_range = np.arange(t_start, 1.0, max(0.01, t_step / 100))
            high_range = np.arange(1.0 + t_step, t_end + t_step, t_step)
            temps = np.unique(np.concatenate([low_range, high_range]))
        else:
            temps = np.arange(t_start, t_end + t_step, t_step)
    else:
        temps = np.asarray(temperatures, dtype=float)

    two_ph_p = Path(two_phonon_file) if two_phonon_file else None
    calculator = TransitionRate(str(coupling_path), two_phonon_data_file=two_ph_p)

    # For combined 1d+2d runs the coupling file itself carries the second-order
    # coefficients (V2_*); make sure the calculator sees them as the 2-phonon source.
    if two_ph_p is None and coupling_data is not None and coupling_data.has_second_order:
        calculator.data_2ph = calculator.data

    omega, J_0_pm, J_p_m, J_0_0 = calculator.get_spectral_density(res=0.02, sigma=7.5)
    if two_ph_p is not None or (coupling_data is not None and coupling_data.has_second_order):
        omega_x, omega_y, J2_0_pm, J2_p_m, J2_0_0 = calculator.get_2d_spectral_density(res=0.5, sigma=7.5)
    else:
        omega_x = omega_y = J2_0_pm = J2_p_m = J2_0_0 = None

    omega_zfs = calculator.data["zfs"] / CONSTANTS["meV2J"]

    rate_results = {
        "first_order":  {"0_1": [], "1_-1": []},
        "second_order": {"0_1": [], "1_-1": []},
        "two_phonon":   {"0_1": [], "1_-1": []},
    }
    directional_results = {
        "0_to_1": [], "0_to_-1": [],
        "1_to_0": [], "-1_to_0": [],
        "1_to_-1": [], "-1_to_1": [],
    }
    valid_temps = []

    print(f"[3/5] Computing transition rates for {len(temps)} temperature points...")
    for T in temps:
        calculator.compute_transition_rates(T, omega, J_0_pm, J_p_m, J_0_0, omega_zfs)
        if two_ph_p is not None or (coupling_data is not None and coupling_data.has_second_order):
            calculator.compute_two_phonon_rates(T, omega_x, omega_y, J2_0_pm, J2_p_m, J2_0_0, omega_zfs)

        total_rates = calculator.get_total_rates()
        dir_rates = calculator.get_directional_rates()

        if total_rates and dir_rates:
            for k, val in dir_rates.items():
                directional_results[k].append(val)
            for ord_k in ["first_order", "second_order", "two_phonon"]:
                for trans_k in ["0_1", "1_-1"]:
                    rate_results[ord_k][trans_k].append(total_rates[ord_k][trans_k])
            valid_temps.append(float(T))

    valid_temps_arr = np.asarray(valid_temps, dtype=float)

    # Save rates
    rates_path = run_dir / "transition_rates.npz"
    np.savez(
        rates_path,
        defect=defect,
        cell_size=cell_size,
        calc_method=calc_method,
        pert_scale=raw_data.pert_scale if raw_data else coupling_data.pert_scale,
        temperatures=valid_temps_arr,
        **rate_results,
    )

    dir_rates_path = run_dir / "directional_rates.npz"
    np.savez(
        dir_rates_path,
        defect=defect,
        cell_size=cell_size,
        calc_method=calc_method,
        pert_scale=raw_data.pert_scale if raw_data else coupling_data.pert_scale,
        temperatures=valid_temps_arr,
        **directional_results,
    )
    print(f"      Saved transition rates -> {rates_path.name}")
    print(f"      Saved directional rates -> {dir_rates_path.name}")

    # ── 7. Population Relaxation Dynamics & T1 Times ─────────────────────────
    ms_mapping = {
        "ms_1": [1.0, 0.0, 0.0],
        "ms_0": [0.0, 1.0, 0.0],
        "ms_-1": [0.0, 0.0, 1.0],
    }
    init_pop = ms_mapping.get(init_state, [0.0, 1.0, 0.0])

    print(f"[4/5] Solving population relaxation dynamics (initial state: {init_state})...")
    dynamics = RelaxationDynamics(init_state=init_pop, rates_data=str(dir_rates_path))

    t1_fit_list = dynamics.compute_T1_range()
    t1_eig_list = [dynamics.get_T1_eigenval(temp_index=i)[0] for i in range(len(valid_temps_arr))]

    t1_fit_arr = np.asarray(t1_fit_list, dtype=float)
    t1_eig_arr = np.asarray(t1_eig_list, dtype=float)
    rates_fit = np.where(np.isfinite(t1_fit_arr) & (t1_fit_arr > 0), 1.0 / t1_fit_arr, np.nan)
    rates_eig = np.where(np.isfinite(t1_eig_arr) & (t1_eig_arr > 0), 1.0 / t1_eig_arr, np.nan)

    t1_path = run_dir / "t1_relaxation.npz"
    np.savez(
        t1_path,
        defect=defect,
        cell_size=cell_size,
        calc_method=calc_method,
        order=order,
        init_state=init_state,
        temperatures=valid_temps_arr,
        t1_fit=t1_fit_arr,
        t1_eigenval=t1_eig_arr,
        rates_fit=rates_fit,
        rates_eigenval=rates_eig,
    )
    print(f"      Saved T1 relaxation -> {t1_path.name}")

    # ── 8. Summary and Metadata ──────────────────────────────────────────────
    run_info = {
        "defect": defect,
        "cell_size": cell_size,
        "calc_method": calc_method,
        "order": order,
        "timestamp": datetime.now().isoformat(),
        "inputs": {
            "sim_folder": str(sim_folder) if sim_folder else None,
            "raw_zfs_file": str(raw_zfs_file) if raw_zfs_file else None,
            "coupling_file": str(coupling_file) if coupling_file else None,
            "phonon_file": str(phonon_file) if phonon_file else None,
            "two_phonon_file": str(two_phonon_file) if two_phonon_file else None,
        },
        "outputs": {
            "raw_zfs_data": str(raw_path) if raw_data else None,
            "spin_phonon_coupling": str(coupling_path),
            "transition_rates": str(rates_path),
            "directional_rates": str(dir_rates_path),
            "t1_relaxation": str(t1_path),
        },
        "temperatures": {
            "min": float(np.min(valid_temps_arr)),
            "max": float(np.max(valid_temps_arr)),
            "count": len(valid_temps_arr),
            "points": len(valid_temps_arr),
        },
        "init_state": init_state,
    }

    info_path = run_dir / "run_info.json"
    with open(info_path, "w") as f:
        json.dump(run_info, f, indent=2)

    # ── 9. Optional Plotting ─────────────────────────────────────────────────
    figures_saved = []
    if save_plots:
        fig_dir = run_dir / "figures"
        fig_dir.mkdir(exist_ok=True)
        print(f"[5/5] Generating figures in {fig_dir.name}/ ...")

        # Coupling plots (1D and 2D diagonal)
        try:
            coupling_display = coupling_data.to_unit("MHz").frequencies_to_unit("meV")
            
            # 1D coupling plot
            fig_c1, _, _ = plot_1d_spectral_functions(
                frequencies_mev=coupling_display.frequencies,
                V_0_0=coupling_display.V_0_0,
                V_p_m=coupling_display.V_p_m,
                V_0_pm=coupling_display.V_0_pm,
                order=1,
            )
            fc1_path = fig_dir / "coupling_spectral_1d.png"
            fig_c1.savefig(fc1_path, dpi=300)
            figures_saved.append(fc1_path)

            # Also save alias coupling_spectral.png for backward compatibility
            fc_path = fig_dir / "coupling_spectral.png"
            fig_c1.savefig(fc_path, dpi=300)
            figures_saved.append(fc_path)

            # 2D coupling plot (diagonal phonons i == j)
            if coupling_display.V2_0_0 is not None:
                v2_00 = coupling_display.V2_0_0
                v2_pm = coupling_display.V2_p_m
                v2_0pm = coupling_display.V2_0_pm

                v2_00_diag = np.diag(v2_00) if v2_00.ndim == 2 else v2_00
                v2_pm_diag = np.diag(v2_pm) if (v2_pm is not None and v2_pm.ndim == 2) else (v2_pm if v2_pm is not None else np.zeros_like(v2_00_diag))
                v2_0pm_diag = np.diag(v2_0pm) if (v2_0pm is not None and v2_0pm.ndim == 2) else (v2_0pm if v2_0pm is not None else np.zeros_like(v2_00_diag))

                fig_c2, _, _ = plot_1d_spectral_functions(
                    frequencies_mev=coupling_display.frequencies,
                    V_0_0=v2_00_diag,
                    V_p_m=v2_pm_diag,
                    V_0_pm=v2_0pm_diag,
                    order=2,
                )
                fc2_path = fig_dir / "coupling_spectral_2d.png"
                fig_c2.savefig(fc2_path, dpi=300)
                figures_saved.append(fc2_path)
        except Exception as e:
            print(f"Warning: could not generate coupling plot: {e}")

        # Transition rate stacked plots
        try:
            fr_path = fig_dir / "transition_rates.png"
            plot_transition_rates_stacked(rates_path, output_path=fr_path)
            figures_saved.append(fig_dir / "transition_rates_0_1.png")
            figures_saved.append(fig_dir / "transition_rates_1_-1.png")
        except Exception as e:
            print(f"Warning: could not generate rate plots: {e}")

        # T1 plot
        try:
            ft1_path = fig_dir / "t1_vs_temperature.png"
            plot_t1_relaxation(t1_path, output_path=ft1_path)
            figures_saved.append(ft1_path)
        except Exception as e:
            print(f"Warning: could not generate T1 plot: {e}")

    print(f"\n========================================================")
    print(f"  Run successfully completed!")
    print(f"  All results stored in: {run_dir}")
    print(f"========================================================\n")

    return {
        "run_dir": run_dir,
        "run_info": run_info,
        "raw_zfs_path": raw_path if raw_data else None,
        "coupling_path": coupling_path,
        "rates_path": rates_path,
        "directional_rates_path": dir_rates_path,
        "t1_path": t1_path,
        "figures": figures_saved,
    }
