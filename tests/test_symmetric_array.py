"""SymmetricArray: memory-frugal in-memory holder for symmetric (n, n, 3, 3) tensors.

Stores only canonical i <= j entries; [j, i] reads redirect to [i, j].
Materializes to a dense ndarray on demand (np.asarray) for numpy consumers.
"""
import numpy as np

from beyblade.models import SymmetricArray


def _tensor(scale=1.0):
    t = np.arange(9, dtype=float).reshape(3, 3) * scale
    return 0.5 * (t + t.T)  # symmetric tensor itself


class TestConstruction:
    def test_starts_empty(self):
        arr = SymmetricArray(4)
        assert arr.shape == (4, 4, 3, 3)
        assert not arr.any_entry()

    def test_n_modes_roundtrip_size(self):
        arr = SymmetricArray(1150)
        assert arr.shape == (1150, 1150, 3, 3)
        # memory must stay tiny while empty
        assert arr.nbytes_stored < 10_000


class TestAccess:
    def test_set_get_canonical_order(self):
        arr = SymmetricArray(4)
        arr[1, 2] = _tensor(3.0)
        np.testing.assert_array_equal(arr[1, 2], _tensor(3.0))

    def test_get_redirected_order(self):
        arr = SymmetricArray(4)
        arr[1, 2] = _tensor(3.0)
        np.testing.assert_array_equal(arr[2, 1], _tensor(3.0))

    def test_set_redirected_order_stores_canonical(self):
        arr = SymmetricArray(4)
        arr[2, 1] = _tensor(3.0)
        np.testing.assert_array_equal(arr[1, 2], _tensor(3.0))
        assert arr.n_packed == 1

    def test_overwrite_same_canonical_pair(self):
        arr = SymmetricArray(4)
        arr[1, 2] = _tensor(1.0)
        arr[2, 1] = _tensor(5.0)
        assert arr.n_packed == 1
        np.testing.assert_array_equal(arr[1, 2], _tensor(5.0))


class TestArrayConversion:
    def test_asarray_dense_symmetric(self):
        arr = SymmetricArray(3)
        arr[0, 2] = _tensor(2.0)
        arr[1, 1] = _tensor(7.0)
        dense = np.asarray(arr)
        assert dense.shape == (3, 3, 3, 3)
        np.testing.assert_array_equal(dense[0, 2], _tensor(2.0))
        np.testing.assert_array_equal(dense[2, 0], _tensor(2.0))
        np.testing.assert_array_equal(dense[1, 1], _tensor(7.0))
        assert not dense[0, 1].any()

    def test_norm_over_tensor_axes(self):
        """Plotter calls np.linalg.norm(arr, axis=(2, 3))."""
        arr = SymmetricArray(3)
        arr[0, 1] = _tensor(2.0)
        Z = np.linalg.norm(np.asarray(arr), axis=(2, 3))
        assert Z[0, 1] == Z[1, 0] > 0

    def test_scalar_multiplication_stays_symmetric(self):
        """Unit conversion multiplies by a factor."""
        arr = SymmetricArray(3)
        arr[0, 1] = _tensor(2.0)
        scaled = arr * 1e6
        np.testing.assert_allclose(scaled[1, 0], np.asarray(arr)[0, 1] * 1e6)

    def test_copy_independent(self):
        arr = SymmetricArray(3)
        arr[0, 1] = _tensor(2.0)
        dup = arr.copy()
        dup[0, 1] = _tensor(9.0)
        np.testing.assert_array_equal(arr[0, 1], _tensor(2.0))


class TestPackedIO:
    def test_packed_rows_roundtrip(self):
        arr = SymmetricArray(4)
        arr[2, 0] = _tensor(1.0)
        arr[3, 3] = _tensor(4.0)
        packed = arr.to_packed()
        assert packed.shape[0] == 2
        rebuilt = SymmetricArray.from_packed(packed)
        np.testing.assert_array_equal(np.asarray(rebuilt), np.asarray(arr))
