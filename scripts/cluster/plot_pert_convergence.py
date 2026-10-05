#!/usr/bin/env python3
"""plot_pert_convergence.py -- D vs displacement with residuals from pack npz.

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

import argparse
import sys
from pathlib import Path

import numpy as np

# allow running from anywhere: find the beyblade package on the repo's src/
_repo = Path(__file__).resolve().parents[2]
if (_repo / "src").is_dir():
    sys.path.insert(0, str(_repo / "src"))

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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot ZFS principal components vs perturbation scale with linear-fit residuals"
    )
    ap.add_argument("npz", type=Path, help="npz from pack_perturbation_runs.py")
    ap.add_argument("-m", "--mode", type=int, action="append", help="plot only this mode (repeatable; default: all)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="output png (default: <npz stem>_convergence.png)")
    ap.add_argument(
        "--quad", action="store_true", help="also fit y = a + b x + c x^2 and report the second-order derivative 2c"
    )
    ap.add_argument(
        "--csv", type=Path, default=None, help="write slopes (and quad coefficients with --quad) to this csv"
    )
    ap.add_argument("--no-show", action="store_true", help="don't call plt.show() (useful on headless cluster)")
    args = ap.parse_args()

    try:
        import matplotlib

        if args.no_show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("Error: matplotlib not available")

    data = np.load(args.npz)
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

    wanted = args.mode if args.mode else [int(m) for m in modes]
    missing = [m for m in wanted if m not in set(modes.tolist())]
    if missing:
        print(f"warning: modes {missing} not in npz", file=sys.stderr)
    if args.output is None:
        args.output = args.npz.with_name(args.npz.stem + "_convergence.png")

    csv_rows: list[tuple] = []

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
        vals = np.stack([ZFSTensor(matrix=t).rotate(R.T).matrix.diagonal() for t in t_all])
        # plot order: ascending signed value / ascending |value|, so lines
        # connect monotonically instead of zigzagging between +Q and -Q
        sgn = np.argsort(p_use)
        mag = np.argsort(np.abs(p_use))

        ncols = 2
        fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for k in range(3):
            a, b, y_fit, res = linear_fit(p_use, vals[:, k])
            axes[0, 0].plot(p_use[sgn], vals[sgn, k], "o-", label=f"PC {k + 1}")
            axes[1, 0].plot(p_use[sgn], res[sgn], "o-")
            # twin panel: D vs |p| (magnitude, catches even-order contamination)
            a2, b2, y_fit2, res2 = linear_fit(np.abs(p_use), vals[:, k])
            axes[0, 1].plot(np.abs(p_use)[mag], vals[mag, k], "o-")
            axes[1, 1].plot(np.abs(p_use)[mag], res2[mag], "o-")
            row: list = [mode, k + 1, f"{b:.6g}", f"{np.max(np.abs(res)):.4g}"]
            msg = (
                f"mode {mode} PC {k + 1}: slope {b:.4g} MHz/unit-pert, max |res| {np.max(np.abs(res)):.4g} MHz (signed)"
            )
            if args.quad:
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
            print(msg)
            csv_rows.append(tuple(row))
        for j, title in enumerate(["signed perturbation", "|perturbation|"]):
            axes[0, j].set_title(f"mode {mode}: ZFS PCs ({title})")
            axes[0, j].set_ylabel("principal component (MHz)")
            axes[1, j].set_xlabel("perturbation scale")
            axes[1, j].set_ylabel("residual (MHz)")
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        stem, ext = args.output.stem, args.output.suffix or ".png"
        out = args.output.with_name(f"{stem}_mode{mode}{ext}")
        fig.savefig(out, dpi=150)
        print(f"  -> {out}")
        if not args.no_show:
            plt.show()

    if args.csv:
        header = ["mode", "pc", "slope", "max_abs_res"]
        if args.quad:
            header += ["d2_reg", "d2_3pt"]  # 3-pt = central diff if Q=0 present, else one-sided
        with args.csv.open("w") as fh:
            fh.write(",".join(header) + "\n")
            for row in csv_rows:
                fh.write(",".join(str(v) for v in row) + "\n")
        print(f"  -> {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
