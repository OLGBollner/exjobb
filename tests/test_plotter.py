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

        fig, ax, ax2 = plot_1d_spectral_functions(freqs, V_00, V_pm, V_0pm, zfs_mev=1.2)
        assert fig is not None
        assert ax is not None
        assert ax2 is not None
        plt.close(fig)

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
