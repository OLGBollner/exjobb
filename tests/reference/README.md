# Frozen T1 / transition-rate reference values

These JSON files pin what the pipeline computes for `NV_64`, `NV_512` and `ClV_128`. They
back `tests/test_t1_regression.py`, which compares numbers; the rest of the suite only
checks shapes and finiteness.

They record current behaviour, not correct behaviour. See [Observations](#observations) for
things that look wrong and were left alone.

## Provenance

| | |
|---|---|
| Frozen at commit | `1db0004b46079d7daa444b1fd544fee83234b4a2` (`dev`) |
| Generated with | CPython 3.12.15, numpy 2.5.4, scipy 1.18.1 |
| Host | Intel Core i5-8250U, scipy-openblas 0.3.34 |
| Temperature grid (K) | 1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300 |
| Initial state | `ms_0` |
| Method | `all_bands` (`--method all`), perturbation order 2 |

The grid is passed explicitly via `--temperatures` so it does not depend on the pipeline's
internal `arange`. Only `all_bands` is frozen: the `defect_band_approx` datasets differ
only in band selection baked into the raw ZFS `.npz` files and re-enter the same
`ZFSManager` / `TransitionRate` / `RelaxationDynamics` code.

## Regenerating

```sh
python scripts/regenerate_t1_reference.py
git diff tests/reference    # any change here is a change to published numbers
```

Each fixture also carries its own `invocation.argv`, which is what the test replays. For
`NV_64`, from the repository root:

```sh
git lfs pull   # the inputs are LFS payloads
spinlabs run \
  --raw-zfs-file-1d NV_64/NV_64_raw_zfs_data_all_bands_1d.npz \
  --raw-zfs-file-2d NV_64/NV_64_raw_zfs_data_all_bands_2d.npz \
  -ph NV_64/phonon_data.npz \
  --method all --init-state ms_0 \
  --temperatures 1 5 10 20 30 50 75 100 150 200 250 300 \
  -o <run dir> --run-name reference
```

`-ph` is required: with `--raw-zfs-file-1d/2d`, phonon auto-discovery does not run and the
combined 1d+2d analysis aborts.

## Tolerances

| Quantity | `rtol` | `atol` | Max deviation seen across Python 3.10/3.12/3.13 |
|---|---|---|---|
| Transition rates (all orders) | `1e-9` | `1e-30` s⁻¹ | `4.7e-16` |
| `t1_eigenval` | `1e-9` | 0 | `8.5e-16` |
| `t1_fit` | `1e-6` | 0 | `6.9e-10` |

Repeated runs on one interpreter are bit-identical, so the spread is entirely
cross-version. `t1_fit` is looser because it goes through `odeint` plus `curve_fit`,
whereas `t1_eigenval` is a closed-form 3×3 eigendecomposition. The remaining headroom
covers a different BLAS kernel or CPU on a GitHub runner.

The `1e-30` s⁻¹ floor on rates exists because the `NV_64` one-phonon channel emits values
around `1e-171` s⁻¹ (see Observations), which no relative tolerance can bound. Rates of
interest are ≥ `1e-8` s⁻¹.

The inf/NaN pattern is compared exactly — it is part of the frozen result.

## Frozen T1 values

`t1_eigenval`, in seconds. Full precision and the rate breakdown are in the JSON files.

| T (K) | NV_64 | NV_512 | ClV_128 |
|---:|---:|---:|---:|
| 1 | inf | 1.2113e+07 | 2.7801e+00 |
| 5 | inf | 1.2113e+07 | 1.5252e+00 |
| 10 | inf | 1.1974e+07 | 8.3708e-01 |
| 20 | inf | 1.0067e+07 | 4.0343e-01 |
| 30 | 1.0112e+08 | 6.0727e+06 | 2.3200e-01 |
| 50 | 9.7802e+03 | 2.7116e+05 | 6.7794e-02 |
| 75 | 7.0436e+01 | 1.2170e+04 | 1.6735e-02 |
| 100 | 5.4665e+00 | 1.6068e+03 | 6.2627e-03 |
| 150 | 3.8914e-01 | 1.2810e+02 | 1.8365e-03 |
| 200 | 9.7783e-02 | 2.5747e+01 | 8.5819e-04 |
| 250 | 4.0919e-02 | 8.3625e+00 | 4.9857e-04 |
| 300 | 2.2124e-02 | 3.6564e+00 | 3.2720e-04 |

## CI

No workflow change was needed: the existing job checks out with `lfs: true` and runs
`pytest -q`, and `ruff check src tests scripts` already covers the new files.

The test adds about 27 s warm (suite 50.6 s → 77.2 s; `NV_512` 7–9 s, `NV_64` and
`ClV_128` ~3 s each) and closer to a minute on the cold cache a fresh runner gets, most of
it reading LFS payloads. Peak RSS is 0.70 GB for `NV_512`. Cost is dominated by the fixed
coupling calculation rather than the temperature loop, so a shorter grid would save little;
running `ClV_128` alone would save roughly 10 s warm. To trim it, narrow
`EXPECTED_SYSTEMS` in `tests/test_t1_regression.py` — CI then stops guarding the systems
you drop.

## Observations

Left as they are, per the task constraints; flagged for the physics audit:

1. **`NV_64` one-phonon rates are ~`1e-171` s⁻¹ at every temperature** — numerically zero
   and essentially constant in T, where `NV_512` gives `1e-7` and `ClV_128` gives `1e0`.
   Consistent with a Gaussian-broadening tail artefact: the spectral density is evaluated
   at the ZFS frequency (~0.0119 meV), far below `NV_64`'s lowest phonon mode, so
   `exp(-x²/2σ²)` underflows.

2. **`NV_64` T1 is `inf` for T ≤ 20 K.** `get_T1_eigenval` discards eigenvalues with
   `|λ| < 1e-9` (a hard absolute cutoff in s⁻¹) and returns `inf` when none survive, so
   genuinely slow relaxation (T1 > 10⁹ s) is reported as infinite.

3. **That cutoff is masking a sign error, not just slowness.** Lowering it to `1e-30`
   turns `NV_64` at 20 K into T1 = `-7.7e+27` s. The near-zero eigenvalues there are noise
   straddling zero.

4. **`get_time_estimate` computes `t_end = 10 * T1_eigenval + 1/T`**, adding a reciprocal
   temperature (K⁻¹) to a time (s). Harmless at these magnitudes, but dimensionally wrong.

5. **`t1_fit` is the less stable of the two T1 estimators.** It agrees with `t1_eigenval`
   to ~5e-8 in the well-behaved regime but diverges to 1.6e-4 relative for `NV_512` at
   5 K, where `curve_fit` also emits `OptimizeWarning: Covariance of the parameters could
   not be estimated`. Prefer `t1_eigenval`.

6. **Phonon auto-discovery does not cover the `--raw-zfs-file-1d/2d` input mode.**
   `find_default_phonon_file_for_raw` is only consulted for `--raw-zfs-file`, so a combined
   1d+2d run without an explicit `-ph` fails in `process_both_orders` even when
   `phonon_data.npz` sits next to the inputs.

7. **CI pulls 530 MB of LFS payloads to use about 27 MB.** `scripts/derivatives/` holds
   452 MB of `NV_512` 2D coefficient files that no test touches. Pre-existing and unchanged
   here, but it is most of the checkout time.
