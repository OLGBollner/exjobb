# Frozen T1 / transition-rate reference values

These JSON files pin **what the pipeline computes today** for `NV_64`, `NV_512` and
`ClV_128`. They are the value-level guard behind `tests/test_t1_regression.py`: the rest
of the suite checks shapes and finiteness, this checks numbers.

They record behaviour; they are not a statement that the behaviour is correct. See
[Observations](#observations-recorded-not-fixed) at the end for things that look wrong and
were deliberately left alone.

## Provenance

| | |
|---|---|
| Frozen at commit | `1db0004b46079d7daa444b1fd544fee83234b4a2` (`dev`, "Merge pull request #47 from OLGBollner/feat/d-table-base-crystal") |
| Generated with | CPython 3.12.15, numpy 2.5.4, scipy 1.18.1 |
| BLAS | scipy-openblas 0.3.34 (`DYNAMIC_ARCH`, SkylakeX kernel) |
| Host CPU | Intel Core i5-8250U |
| Temperature grid (K) | 1, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300 |
| Initial state | `ms_0` |
| Method | `all_bands` (`--method all`) |
| Perturbation order | 2 (combined 1st + 2nd order) |

The grid is explicit (`--temperatures`) rather than `--t-start/--t-end/--t-step`, so it is
exactly reproducible and does not depend on the pipeline's internal `arange` construction.
Twelve points were chosen to straddle every regime the pipeline shows: the low-temperature
corner where `NV_64` underflows to an infinite T1, the steep crossover between 30 K and
100 K, and room temperature.

Only `all_bands` is frozen. The `defect_band_approx` datasets differ upstream of the
computational core — they are a different band selection baked into the raw ZFS `.npz`
files, and they re-enter the same `ZFSManager` / `TransitionRate` / `RelaxationDynamics`
code these fixtures already cover.

## Exact invocation

Each fixture carries its own `invocation.argv`, which is what the test replays. For
`NV_64` that is, from the repository root:

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

`-ph` is required: when the two raw files are passed as `--raw-zfs-file-1d/2d`, the
pipeline's phonon auto-discovery does not run and the combined 1d+2d analysis aborts.

To regenerate all three:

```sh
python scripts/regenerate_t1_reference.py
git diff tests/reference    # any change here is a change to published numbers
```

## Tolerances

`tests/test_t1_regression.py` compares with:

| Quantity | `rtol` | `atol` |
|---|---|---|
| Transition rates (all three orders) | `1e-9` | `1e-30` s⁻¹ |
| `t1_eigenval` | `1e-9` | 0 |
| `t1_fit` | `1e-6` | 0 |

The inf/NaN pattern is compared **exactly** — it is part of the frozen result, not noise to
be tolerated away.

### Why these numbers

They were measured, not guessed. The same sweep was run on all three interpreters in the
CI matrix, which also spans two numpy/scipy generations:

- CPython 3.10.22, numpy 2.2.6, scipy 1.15.3
- CPython 3.12.15, numpy 2.5.4, scipy 1.18.1
- CPython 3.13.16, numpy 2.5.4, scipy 1.18.1

Worst relative deviation from the frozen values, over all three systems and all twelve
temperatures:

| Quantity | Max observed | Chosen `rtol` | Headroom |
|---|---|---|---|
| Transition rates | `4.7e-16` | `1e-9` | ~2 000 000× |
| `t1_eigenval` | `8.5e-16` | `1e-9` | ~1 000 000× |
| `t1_fit` | `6.9e-10` | `1e-6` | ~1 400× |

Repeated runs on one interpreter are bit-identical, so the spread above is entirely
cross-version. `t1_fit` is looser than `t1_eigenval` because it goes through `odeint` plus
`curve_fit` (`relaxation_dynamics.py`) and is iterative, whereas `t1_eigenval` is a
closed-form 3×3 eigendecomposition. The headroom covers a different BLAS kernel or CPU on
a GitHub runner; it is still six or more orders of magnitude tighter than any change a
physics or refactoring bug would produce.

The `1e-30` s⁻¹ absolute floor on rates is physical, not cosmetic: a rate that small is a
mean waiting time about 10¹² times the age of the universe. It exists because the
`NV_64` one-phonon channel currently emits values around `1e-171` s⁻¹ (see below), which no
relative tolerance can sensibly bound. Every rate these papers quote is ≥ `1e-8` s⁻¹, more
than twenty orders above the floor.

### The tolerances are not loose enough to pass anything

Three mutations of the real source, each reverted afterwards, all caught:

| Mutation | Result |
|---|---|
| Broadening `sigma=7.5` → `7.5000001` in `pipeline.py` (1.3e-8 relative) | caught — 9/12 `ClV_128` T1 values moved by 1.4e-9…4.0e-8 |
| T1 eigenvalue cutoff `1e-9` → `1e-30` in `relaxation_dynamics.py` | caught by the inf-pattern check at `NV_64` 10 K and 20 K |
| Two-phonon `1_-1` total scaled by `1 + 1e-8` in `transition_rate.py` | caught — all 12 `ClV_128` values moved by 1.0e-8 |

## Frozen T1 values

`t1_eigenval`, in seconds. Full precision, and the first/second/two-phonon rate breakdown,
are in the JSON files.

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

No change to `.github/workflows/ci.yml` was needed. The existing job already checks out
with `lfs: true` and runs `pytest -q`, so `tests/test_t1_regression.py` is collected
automatically, and `ruff check src tests scripts` already covers both new files.

All three systems run in CI rather than only the cheapest one, because the measured cost
does not justify giving up two thirds of the guard. Like-for-like on the host above, with
a warm page cache:

| | tests | wall clock |
|---|---:|---:|
| Suite without `test_t1_regression.py` | 156 | 50.6 s |
| Suite with it | 160 | 77.2 s |

That is about 27 s added. Per case: `NV_512` 7–9 s, `NV_64` ~3 s, `ClV_128` ~3 s. On a
cold cache — which is what a fresh runner gets — the same file takes ~41 s and the whole
suite ~290 s, so expect closer to a minute of added wall clock there, most of it spent
reading the LFS payloads rather than computing.

Peak RSS is 0.70 GB for `NV_512`, well inside a runner. Cost is dominated by the fixed
coupling calculation, not by the temperature loop, so a shorter grid would save very
little; dropping to `ClV_128` alone would save roughly 10 s warm.

If this ever does become too slow, narrow `EXPECTED_SYSTEMS` in
`tests/test_t1_regression.py` and move the dropped systems to a documented manual run of
`scripts/regenerate_t1_reference.py` — but note that CI then stops guarding them.

## Observations (recorded, not fixed)

Per the task constraints these were left exactly as they are. Flagging them for the
physics audit:

1. **`NV_64` one-phonon rates are ~`1e-171` s⁻¹ at every temperature** — numerically zero,
   and essentially constant in T. `NV_512` gives `1e-7` and `ClV_128` gives `1e0` for the
   same channel. This looks like a Gaussian-broadening tail artefact: the spectral density
   is evaluated at the ZFS frequency (~0.0119 meV), far below `NV_64`'s lowest phonon mode,
   so `exp(-x²/2σ²)` underflows. The value is not meaningful, only its smallness is.

2. **`NV_64` T1 is `inf` for T ≤ 20 K.** `get_T1_eigenval` discards eigenvalues with
   `|λ| < 1e-9` — a hard absolute cutoff in s⁻¹ — and returns `inf` when none survive. Any
   genuinely slow relaxation (T1 > 10⁹ s) is reported as infinite rather than slow.

3. **That cutoff is masking a sign error, not just slowness.** Lowering it to `1e-30`
   during mutation testing turned `NV_64` at 20 K into T1 = `-7.7e+27` s — a *negative*
   relaxation time. Whatever produces the near-zero eigenvalues there is not merely small,
   it is noise straddling zero.

4. **`get_time_estimate` computes `t_end = 10 * T1_eigenval + 1/T`**, adding a reciprocal
   temperature (K⁻¹) to a time (s). Dimensionally inconsistent. It is numerically harmless
   at these magnitudes, but it is not what was meant.

5. **`t1_fit` is the less stable of the two T1 estimators.** It agrees with `t1_eigenval` to
   ~5e-8 in the well-behaved regime, but diverges to 1.6e-4 relative for `NV_512` at 5 K,
   where `curve_fit` also emits `OptimizeWarning: Covariance of the parameters could not be
   estimated`. `t1_eigenval` is the quantity to trust.

6. **Phonon auto-discovery does not cover the `--raw-zfs-file-1d/2d` input mode.**
   `find_default_phonon_file_for_raw` is only consulted for `--raw-zfs-file`, so a combined
   1d+2d run without an explicit `-ph` fails in `process_both_orders` even when
   `phonon_data.npz` sits next to the inputs.

7. **CI pulls 530 MB of LFS payloads to use about 27 MB.** `scripts/derivatives/` holds
   452 MB of `NV_512` 2D coefficient files that no test touches. Pre-existing, unchanged
   here, but it is most of the checkout time.
