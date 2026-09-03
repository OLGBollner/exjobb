import pytest
import numpy as np
import matplotlib.pyplot as plt
from beyblade.plotter import (
    plot_1d_spectral_functions,
    plot_ipr_spectrum,
    plot_2d_spectral_density_map,
    ZFSPlotter,
)

class TestPlotter:
    def test_plot_1d_spectral_functions(self):
        freqs = np.linspace(10, 100, 10)
        V_00 = np.random.rand(10) * 10
        V_pm = np.random.rand(10) * 10
        V_0pm = np.random.rand(10) * 10

        fig, ax, ax2 = plot_1d_spectral_functions(freqs, V_00, V_pm, V_0pm, zfs_mev=1.2, order=1)
        assert fig is not None
        assert ax is not None
        assert ax2 is not None
        plt.close(fig)

        # Test order=2 (diagonal of 2-phonon coupling)
        fig2, ax_2, ax2_2 = plot_1d_spectral_functions(freqs, V_00, V_pm, V_0pm, order=2)
        assert fig2 is not None
        plt.close(fig2)

    def test_plot_ipr_spectrum(self):
        freqs = np.linspace(10, 100, 10)
        ipr = np.random.rand(10)

        fig, ax = plot_ipr_spectrum(freqs, ipr, as_bar=False)
        assert fig is not None
        plt.close(fig)

        fig_bar, ax_bar = plot_ipr_spectrum(freqs, ipr, as_bar=True)
        assert fig_bar is not None
        plt.close(fig_bar)

    def test_plot_2d_spectral_density_map(self):
        freqs = np.linspace(10, 100, 8)
        zfs_2nd = np.random.rand(8, 8, 3, 3)

        fig, ax = plot_2d_spectral_density_map(freqs, zfs_2nd)
        assert fig is not None
        plt.close(fig)

    def test_zfs_plotter_with_spin_phonon_coupling_data(self):
        """Tests that ZFSPlotter accepts SpinPhononCouplingData directly and converts units seamlessly."""
        from pathlib import Path
        from types import SimpleNamespace
        from beyblade.models import SpinPhononCouplingData, ZFSTensor
        from beyblade.plotter import ZFSPlotter
        from beyblade.constants import CONSTANTS

        # Data in Joules
        freqs_j = np.array([20.0, 40.0, 60.0]) * CONSTANTS["meV2J"]
        v_00_j = np.array([1.5, 3.0, 4.5]) * CONSTANTS["MHz2J"]
        v_pm_j = np.array([0.5, 1.0, 1.5]) * CONSTANTS["MHz2J"]
        v_0pm_j = np.array([0.2, 0.4, 0.6]) * CONSTANTS["MHz2J"]

        data = SpinPhononCouplingData(
            order=1,
            defect="NV",
            cell_size=64,
            pert_scale=0.025,
            calc_method="all_bands",
            frequencies=freqs_j,
            frequency_unit="J",
            V_0_0=v_00_j,
            V_p_m=v_pm_j,
            V_0_pm=v_0pm_j,
            coupling_unit="J",
            ground_state_zfs=ZFSTensor(matrix=np.diag([-1000.0, -1000.0, 2000.0]), unit="MHz"),
            symmetries=["A1", "Ex", "Ey"],
            iprs=np.array([0.1, 0.2, 0.3]),
        )

        plotter = ZFSPlotter()
        args = SimpleNamespace(plot=False, ipr=False, output="test_output", format=".png")

        # Plot directly without throwing errors or mutating units
        plotter.plot_data([data], args)
        out_file = Path("figures/test_output.png")
        assert out_file.exists()
        out_file.unlink()
