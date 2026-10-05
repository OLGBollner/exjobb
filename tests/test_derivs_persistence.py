"""Seam 1: derivative tensors survive a save/load round-trip exactly.

Covers the spec item "persist derivative tensors (defect frame) alongside
channel scalars".  zfs_derivs must round-trip densely; zfs_2nd_derivs is
packed on disk (canonical i<=j entries over computed pairs only) and must
come back as the identical symmetric dense array.
"""
import numpy as np

from beyblade.models import SpinPhononCouplingData


RNG = np.random.default_rng(42)


def _make_coupling(n_modes=6, n_pairs=4):
    """Synthetic first+second order coupling data with a symmetric 2d tensor."""
    rng = np.random.default_rng(n_modes)
    d1 = rng.normal(size=(n_modes, 3, 3)) * 1e-3
    d2 = np.zeros((n_modes, n_modes, 3, 3))
    pairs = []
    for k in range(n_pairs):
        i = k % n_modes
        j = (i + 1 + k) % n_modes
        i, j = min(i, j), max(i, j)
        if (i, j) in pairs:
            continue
        pairs.append((i, j))
        t = rng.normal(size=(3, 3)) * 1e-4
        t = 0.5 * (t + t.T)
        d2[i, j] = t
        d2[j, i] = t
    return SpinPhononCouplingData(
        order=2,
        defect="Test",
        cell_size=2,
        pert_scale=0.01,
        calc_method="test",
        frequencies=np.arange(n_modes) * 1e-21,
        V_0_0=np.abs(rng.normal(size=n_modes)),
        V_p_m=np.abs(rng.normal(size=n_modes)),
        V_0_pm=np.abs(rng.normal(size=n_modes)),
        zfs_derivs=d1,
        zfs_2nd_derivs=d2,
    ), set(pairs)


class TestFirstOrderRoundTrip:
    def test_zfs_derivs_roundtrip_exact(self, tmp_path):
        obj, _ = _make_coupling()
        path = obj.save(tmp_path / "coupling.npz")
        loaded = SpinPhononCouplingData.load(path)
        assert loaded.zfs_derivs is not None
        np.testing.assert_array_equal(loaded.zfs_derivs, obj.zfs_derivs)


class TestSecondOrderRoundTrip:
    def test_roundtrip_identical_dense(self, tmp_path):
        obj, pairs = _make_coupling()
        path = obj.save(tmp_path / "coupling2d.npz")
        loaded = SpinPhononCouplingData.load(path)
        np.testing.assert_array_equal(loaded.zfs_2nd_derivs, obj.zfs_2nd_derivs)

    def test_uncomputed_pairs_are_zero_after_load(self, tmp_path):
        obj, pairs = _make_coupling()
        path = obj.save(tmp_path / "coupling2d.npz")
        loaded = SpinPhononCouplingData.load(path)
        n = obj.zfs_2nd_derivs.shape[0]
        for i in range(n):
            for j in range(n):
                if (min(i, j), max(i, j)) not in pairs:
                    assert not loaded.zfs_2nd_derivs[i, j].any()

    def test_file_is_packed_not_dense(self, tmp_path):
        """The npz must hold only canonical i<=j entries, not the full square."""
        obj, pairs = _make_coupling()
        path = obj.save(tmp_path / "coupling2d.npz")
        with np.load(path) as data:
            stored = data["zfs_2nd_derivs"]
            # packed: structured (K,) with i/j/tensor fields, never the dense (n, n, 3, 3)
            assert stored.dtype.names is not None, "expected packed structured array"
            assert set(stored.dtype.names) == {"i", "j", "tensor"}
            assert stored.shape[0] == len(pairs)
            assert stored["tensor"].shape == (len(pairs), 3, 3)
