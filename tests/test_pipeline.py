import json
import tempfile
from pathlib import Path
import numpy as np
import pytest

from beyblade.models import PhononMode, PhononSpectrum, RawZFSData, SpinPhononCouplingData, ZFSTensor
from beyblade.pipeline import get_unique_run_dir, run_full_pipeline


@pytest.fixture
def dummy_coupling_file(tmp_path):
    n_modes = 5
    coupling = SpinPhononCouplingData(
        order=1,
        defect="NV",
        cell_size=64,
        pert_scale=0.025,
        calc_method="all_bands",
        frequencies=np.linspace(10.0, 50.0, n_modes) * 1.60218e-22,  # in J
        frequency_unit="J",
        ground_state_zfs=ZFSTensor(matrix=np.diag([-1e-25, -1e-25, 2e-25]), unit="J"),
        V_0_0=np.array([1e-25, 2e-25, 1.5e-25, 3e-25, 2.2e-25]),
        V_p_m=np.array([2e-25, 1e-25, 2.5e-25, 1.8e-25, 3.1e-25]),
        V_0_pm=np.array([1.2e-25, 2.1e-25, 1.9e-25, 2.7e-25, 1.4e-25]),
        coupling_unit="J",
        symmetries=["A1", "E", "E", "A1", "E"],
        iprs=np.full(n_modes, 0.2),
    )
    save_file = tmp_path / "test_coupling.npz"
    coupling.save(save_file)
    return save_file


def test_unique_run_dir_increments_when_existing(tmp_path):
    root = tmp_path / "runs"
    dir1 = get_unique_run_dir(output_root=root, run_name="my_experiment")
    assert dir1.exists()
    assert dir1.name == "my_experiment"

    dir2 = get_unique_run_dir(output_root=root, run_name="my_experiment")
    assert dir2.exists()
    assert dir2.name == "my_experiment_1"

    dir3 = get_unique_run_dir(output_root=root, run_name="my_experiment")
    assert dir3.exists()
    assert dir3.name == "my_experiment_2"


def test_run_full_pipeline_from_coupling_data(tmp_path, dummy_coupling_file):
    run_root = tmp_path / "test_runs"

    res = run_full_pipeline(
        coupling_file=dummy_coupling_file,
        output_root=run_root,
        run_name="pipeline_test_run",
        t_start=50.0,
        t_end=150.0,
        t_step=50.0,
        save_plots=True,
    )

    run_dir = Path(res["run_dir"])
    assert run_dir.exists()

    # Verify all expected artifacts exist in run_dir
    assert (run_dir / "spin_phonon_coupling.npz").exists()
    assert (run_dir / "transition_rates.npz").exists()
    assert (run_dir / "directional_rates.npz").exists()
    assert (run_dir / "t1_relaxation.npz").exists()
    assert (run_dir / "run_info.json").exists()

    # Check run_info.json contents
    with open(run_dir / "run_info.json") as f:
        info = json.load(f)
    assert info["defect"] == "NV"
    assert info["cell_size"] == 64
    assert info["temperatures"]["count"] == 3

    # Check t1_relaxation.npz contents
    t1_data = np.load(run_dir / "t1_relaxation.npz")
    assert len(t1_data["temperatures"]) == 3
    assert len(t1_data["t1_eigenval"]) == 3
    assert len(t1_data["t1_fit"]) == 3
    assert np.all(np.isfinite(t1_data["t1_eigenval"]))

    # Check figures folder
    fig_dir = run_dir / "figures"
    assert fig_dir.exists()
    assert (fig_dir / "coupling_spectral.png").exists()
    assert (fig_dir / "t1_vs_temperature.png").exists()

    # Test plot_results script functions on this run directory
    import sys
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.plot_results import plot_run_coupling, plot_run_rates, plot_run_t1
    custom_fig_dir = tmp_path / "custom_figures"
    custom_fig_dir.mkdir()
    plot_run_coupling(run_dir, custom_fig_dir, "png", 150, False)
    plot_run_rates(run_dir, custom_fig_dir, "png", 150, False)
    plot_run_t1(run_dir, custom_fig_dir, "png", 150, False)

    assert (custom_fig_dir / "coupling_spectral.png").exists()
    assert (custom_fig_dir / "t1_vs_temperature.png").exists()
