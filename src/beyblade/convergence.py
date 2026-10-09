"""D vs displacement (ZFS) with fit residuals from a packed perturbation npz.

Loads the .npz written by pack_perturbation_runs.py, rotates each ZFS
tensor into the ground-state principal frame (fixed rotation built once
from the Q=0 tensor, as in beyblade's ZFSManager), and plots the diagonal
components against the perturbation scale together with the residuals
from a linear fit.

Usage (anywhere, needs numpy + matplotlib):
    python plot_pert_convergence.py pert_runs.npz [-m MODE] [-o out.png]

With --quad, also fit y = a + b x + c x^2 per principal component and
report 2c (the second-order ZFS derivative d2D/dQ2) next to the linear
slope, so the quadratic-regression estimate can be cross-checked against
a forward-difference value computed from the two smallest-|Q| points.
When Q=0 is present (either as a pert in the npz or via the 'd0' key
written by pack_perturbation_runs.py --ground-state) it is included as
an anchored data point and a central second difference
(f(+Q) + f(-Q) - 2 f(0)) / Q^2 is reported per |Q| as well.
"""
from __future__ import annotations




import json
from pathlib import Path
import sys

import numpy as np

from beyblade.models import ZFSTensor


def principal_frame(D0: np.ndarray) -> np.ndarray:
    """Ground-state principal-axis rotation R, via beyblade.ZFSTensor.

    ZFSTensor(D0).principal_components() returns (D_xx, D_yy, D_zz, R)
    in EPR convention (|D_zz| >= |D_yy| >= |D_xx|, traceless), with R's
    columns the ground-state eigenvectors. Built ONCE; every perturbed
    tensor is then rotated into this fixed frame (R.T @ D @ R), exactly
    as ZFSManager._ingest_raw_data does.
    """
    _, _, _, R = ZFSTensor(matrix=D0).principal_components()
    return R


def linear_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    """OLS fit y = a + b x. Returns (a, b, y_fit, residuals)."""
    b, a = np.polyfit(x, y, 1)
    y_fit = a + b * x
    return a, b, y_fit, y - y_fit


def quad_fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    """OLS fit y = a + b x + c x^2. Returns (a, b, c, y_fit, residuals)."""
    c, b, a = np.polyfit(x, y, 2)
    y_fit = a + b * x + c * x**2
    return a, b, c, y_fit, y - y_fit


def q_needed(c: float, noise: float, snr: float = 3.0) -> float | None:
    """Amplitude Q at which the quadratic term c*Q^2 exceeds noise by snr.

    noise is the max |linear-fit residual|, i.e. the numerical scatter in D.
    Returns None when |c| is negligible (quadratic term stays buried at any
    reachable amplitude) — comparing c*Q^2 to noise directly.
    """
    if c == 0.0 or not np.isfinite(c):
        return None
    q = float(np.sqrt(snr * abs(noise) / abs(c)))
    return q if np.isfinite(q) else None


def one_sided_d2(p: np.ndarray, v: np.ndarray) -> float | None:
    """Second derivative from the three smallest-|Q| points (Lagrange).

    Evaluated at the smallest-|Q| point (closest to Q=0):
      f''(x0) = 2 [ f0/(h1(h1+h2)) - f1/(h1 h2) + f2/(h2(h1+h2)) ]
    with h1 = p1-p0, h2 = p2-p1. Returns None if fewer than 3 points or
    the points are collinear/not strictly ordered in |Q| spacing.
    """
    if len(p) < 3:
        return None
    srt = np.argsort(np.abs(p))
    p_s, v_s = p[srt], v[srt]
    if np.isclose(p_s[0], 0.0) and ((p_s > 0).sum() >= 2 or (p_s < 0).sum() >= 2):
        # with a Q=0 anchor: triple (0, Q1, Q2) on whichever side has points,
        # evaluated at Q=0 itself
        pos_idx = np.where(p_s > 0)[0][:2]
        neg_idx = np.where(p_s < 0)[0][:2]
        idx = np.concatenate(([0], pos_idx if len(pos_idx) == 2 else neg_idx))
    else:
        # no anchor: three consecutive same-side points, evaluated at the
        # smallest-|Q| one
        side = p_s > 0 if (p_s > 0).sum() >= 3 else p_s < 0
        idx = np.where(side)[0][:3]
    if len(idx) < 3:
        return None
    p3, v3 = p_s[idx], v_s[idx]
    h1, h2 = p3[1] - p3[0], p3[2] - p3[1]
    if h1 <= 0 or h2 <= 0:
        return None
    return 2 * (v3[0] / (h1 * (h1 + h2)) - v3[1] / (h1 * h2) + v3[2] / (h2 * (h1 + h2)))




def read_conv_metadata(npz_path: Path) -> dict:
    """Return the JSON 'metadata' key of a packed convergence npz ({} if absent)."""
    data = np.load(npz_path)
    if "metadata" not in data:
        return {}
    try:
        return json.loads(str(data["metadata"]))
    except (ValueError, TypeError):
        return {}


def plot_pert_convergence(
    npz_path: Path,
    output_path: Path | None = None,
    mode: list[int] | None = None,
    quad: bool = False,
    deviation: bool = False,
    csv_path: Path | None = None,
    show: bool = False,
    drop_anchor: bool = False,
) -> list[Path]:
    """Plot ZFS principal components vs perturbation scale from a packed npz.

    Returns the list of written output files.
    """
    try:
        import matplotlib

        if not show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("Error: matplotlib not available")

    data = np.load(npz_path)
    modes = data["modes"]
    perts = data["perts"]
    tensors = data["tensors"]

    # Anchor the Q=0 point: take it from the perts array if present, else
    # from the 'd0' key written by pack_perturbation_runs.py --ground-state.
    if 0.0 not in set(perts.tolist()):
        if "d0" in data:
            perts = np.concatenate(([0.0], perts))
            tensors = np.concatenate((np.broadcast_to(data["d0"], (tensors.shape[0], 1, 3, 3)), tensors), axis=1)
        else:
            print(
                "warning: no Q=0 point (no 0 in perts, no 'd0' key in npz); fits use the unanchored points only",
                file=sys.stderr,
            )

    # Q=0 is the relaxed structure, so the ZFS tensor there is identical for
    # every mode: if a mode lacks its own Q=0 run (absent column or NaN),
    # borrow the Q=0 tensor from whichever mode has one.
    izero_g = int(np.argmin(np.abs(perts)))
    q0_shared = None
    if np.isclose(perts[izero_g], 0.0):
        for i in range(tensors.shape[0]):
            t0 = tensors[i][izero_g]
            if not np.isnan(t0).any():
                q0_shared = t0
                break
        if q0_shared is not None:
            patched = 0
            for i in range(tensors.shape[0]):
                if np.isnan(tensors[i][izero_g]).any():
                    tensors[i][izero_g] = q0_shared
                    patched += 1
            if patched:
                print(
                    f"borrowed the Q=0 tensor for {patched} mode(s) that lack "
                    "their own pert=0 run (identical relaxed structure)",
                    file=sys.stderr,
                )

    wanted = mode if mode else [int(m) for m in modes]
    missing = [m for m in wanted if m not in set(modes.tolist())]
    if missing:
        print(f"warning: modes {missing} not in npz", file=sys.stderr)
    if output_path is None:
        output_path = npz_path.with_name(npz_path.stem + "_convergence.png")

    csv_rows: list[tuple] = []
    out_files: list[Path] = []

    for mode in wanted:
        if mode not in set(modes.tolist()):
            continue
        i = int(np.where(modes == mode)[0][0])
        # drop unfinished runs (NaN tensors) BEFORE diagonalizing: eigh on a
        # NaN slot raises LinAlgError for the whole mode
        ok0 = ~np.isnan(tensors[i]).any(axis=(1, 2))
        dropped = perts[~ok0]
        if len(dropped):
            print(
                f"mode {mode}: dropping {len(dropped)} run(s) with no ZFS tensor "
                f"(unfinished or failed): {np.array2string(dropped)}",
                file=sys.stderr,
            )
        if ok0.sum() < 2:
            print(f"mode {mode}: fewer than 2 finished runs, skipping")
            continue
        p_all, t_all = perts[ok0], tensors[i][ok0]
        # fixed principal frame from the Q=0 (relaxed) tensor; plot the
        # diagonal components of every tensor rotated into that frame
        izero = int(np.argmin(np.abs(p_all)))
        if not np.isclose(p_all[izero], 0.0):
            print(
                f"mode {mode}: warning: no Q=0 tensor; principal frame taken "
                "from the smallest-|Q| perturbation instead",
                file=sys.stderr,
            )
        R = principal_frame(t_all[izero])
        p_use = p_all
        t_use = t_all
        if drop_anchor:
            p_use = p_all[np.abs(p_all) > 0]
            t_use = t_all[np.abs(p_all) > 0]
            if len(p_use) == 0:
                print(f"mode {mode}: --drop-anchor leaves no points, skipping", file=sys.stderr)
                continue
        vals = np.stack([np.diag(R.T @ t @ R) for t in t_use])
        if deviation:
            # plot D(Q) - D(Q=0) so the response is visible on its own scale
            vals = vals - vals[izero]
        # plot order: ascending signed value, so lines connect monotonically
        # instead of zigzagging between +Q and -Q
        sgn = np.argsort(p_use)

        ncols = 2
        fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for k in range(3):
            a, b, y_lin, res = linear_fit(p_use, vals[:, k])
            qa, qb, qc, y_q, qres = quad_fit(p_use, vals[:, k])
            axes[0, 0].plot(p_use[sgn], vals[sgn, k], "o", label=f"PC {k + 1}")
            axes[0, 0].plot(p_use[sgn], y_lin[sgn], "-", alpha=0.4, color=axes[0, 0].lines[-1].get_color())
            axes[1, 0].plot(p_use[sgn], res[sgn], "o-")
            axes[1, 0].axhline(np.max(np.abs(res)), ls="--", lw=0.8, color="grey")
            axes[1, 0].axhline(-np.max(np.abs(res)), ls="--", lw=0.8, color="grey")
            axes[0, 1].plot(p_use[sgn], vals[sgn, k], "o", label=f"PC {k + 1} (b={b:.4g}, 2c={2 * qc:.4g})")
            axes[0, 1].plot(p_use[sgn], y_q[sgn], "-", alpha=0.4, color=axes[0, 1].lines[-1].get_color())
            axes[1, 1].plot(p_use[sgn], qres[sgn], "o-")
            axes[1, 1].axhline(np.max(np.abs(qres)), ls="--", lw=0.8, color="grey")
            axes[1, 1].axhline(-np.max(np.abs(qres)), ls="--", lw=0.8, color="grey")
            row: list = [mode, k + 1, f"{b:.6g}", f"{np.max(np.abs(res)):.4g}"]
            msg = (
                f"mode {mode} PC {k + 1}: slope {b:.4g} MHz/unit-pert, max |res| {np.max(np.abs(res)):.4g} MHz (signed)"
            )
            if quad:
                qa, qb, qc, _, qres = quad_fit(p_use, vals[:, k])
                d2 = 2 * qc
                fwd = one_sided_d2(p_use, vals[:, k])
                fwd_txt = f"{fwd:.4g}" if fwd is not None else "n/a"
                span = float(np.ptp(vals[:, k])) or 1e-30
                rel = (
                    f" ({abs(d2 - fwd) / max(abs(fwd), 1e-30) * 100:.2f}% from fwd)"
                    if fwd is not None and abs(fwd) > 0.05 * span
                    else ""
                )
                msg += f"\n    quad: D(Q) = {qa:.4g} + {qb:.4g} Q + {qc:.4g} Q^2 (dof {len(p_use) - 3})"
                msg += f"\n    d2D/dQ2: regression {d2:.4g}, one-sided fwd {fwd_txt} MHz/pert^2{rel}"
                qreq = q_needed(qc, np.max(np.abs(res)))
                if qreq is not None:
                    msg += f"\n    Q needed for quad term at SNR 3: |Q| ~ {qreq:.3g} (current max |Q|: {np.max(np.abs(p_use)):.3g})"
                else:
                    msg += "\n    Q needed for quad term at SNR 3: n/a (c negligible vs noise)"
                row += [f"{d2:.6g}", fwd_txt]
                # central second differences (f(+Q) + f(-Q) - 2 f(0)) / Q^2
                if 0.0 in set(p_use.tolist()):
                    f0 = vals[p_use == 0.0, k][0]
                    cents = []
                    for q in sorted({abs(x) for x in p_use if x > 0}):
                        neg, pos = np.isclose(p_use, -q), np.isclose(p_use, q)
                        if neg.any() and pos.any():
                            cent = (vals[pos, k][0] + vals[neg, k][0] - 2 * f0) / q**2
                            cents.append(f"{q:g}:{cent:.4g}")
                            row[-1] = f"{cent:.6g}"
                    if cents:
                        msg += f"\n    central d2 (|Q|:value): {', '.join(cents)} MHz/pert^2"
            print(20 * "-")
            print(msg)
            print(20 * "-", "\n")
            csv_rows.append(tuple(row))
        for j, title in enumerate(["ax + b", "ax² + bx + c"]):
            axes[0, j].set_title(title)
            axes[0, j].set_ylabel("principal component (MHz)")
            axes[1, j].set_xlabel("perturbation scale")
            axes[1, j].set_ylabel("residual (MHz)")
        if deviation:
            fig.suptitle("$D_\\lambda - D_0$")
        axes[0, 1].legend(fontsize=7)
        fig.tight_layout()
        stem, ext = output_path.stem, output_path.suffix or ".png"
        out = output_path.with_name(f"{stem}_mode{mode}{ext}")
        fig.savefig(out, dpi=150)
        out_files.append(out)
        print(f"  -> {out}")
        if show:
            plt.show()

    if csv_path:
        header = ["mode", "pc", "slope", "max_abs_res"]
        if quad:
            header += ["d2_reg", "d2_3pt"]  # 3-pt = central diff if Q=0 present, else one-sided
        with csv_path.open("w") as fh:
            fh.write(",".join(header) + "\n")
            for row in csv_rows:
                fh.write(",".join(str(v) for v in row) + "\n")
        print(f"  -> {csv_path}")
    return out_files
