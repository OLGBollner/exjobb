#!/usr/bin/env python3
"""plot_pert_convergence.py -- D vs displacement with residuals from pack npz.

Loads the .npz written by pack_perturbation_runs.py, diagonalises each ZFS
tensor, matches eigenvector branches across perturbations (by overlap with
the smallest-|pert| reference), and plots each principal component against
the perturbation scale together with the residuals from a linear fit.

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


def diagonalize_tensors(tensors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per (mode, pert) tensor: eigenvalues (ascending) and eigenvectors.

    Returns vals (..., 3) and vecs (..., 3, 3) with vecs[..., :, k] the
    k-th eigenvector column, sorted by eigenvalue.
    """
    vals, vecs = np.linalg.eigh(tensors)
    order = np.argsort(vals, axis=-1)
    vals = np.take_along_axis(vals, order, axis=-1)
    vecs = np.take_along_axis(vecs, order[..., None, :], axis=-1)
    # fix sign ambiguity: make the largest-|.| component positive
    lead = np.argmax(np.abs(vecs), axis=-2, keepdims=True)
    sign = np.sign(np.take_along_axis(vecs, lead, axis=-2))
    vecs = vecs * sign
    return vals, vecs


def match_branches(vals: np.ndarray, vecs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reorder branches along the pert axis so each tracks one eigenvector.

    vals/vecs: (n_pert, 3) and (n_pert, 3, 3) for ONE mode. Branch k of each
    perturbation is assigned to the reference branch it overlaps most with
    (greedy, per perturbation). Returns reordered copies.
    """
    n_pert = vals.shape[0]
    ref = 0  # smallest-|pert| index (npz perts are sorted)
    order = np.zeros((n_pert, 3), dtype=int)
    for j in range(n_pert):
        ov = np.abs(vecs[ref].T @ vecs[j])  # (3, 3): ref branch k vs j branch l
        # greedy assignment of ref branches to pert branches
        assign = [-1] * 3
        pairs = sorted(
            ((ov[k, col], k, col) for k in range(3) for col in range(3)),
            key=lambda t: (-t[0], t[1], t[2]),
        )
        used_k, used_col = set(), set()
        for score, k, col in pairs:
            if k in used_k or col in used_col:
                continue
            assign[k] = col
            used_k.add(k)
            used_col.add(col)
        order[j] = assign
    idx = np.arange(n_pert)[:, None], order
    return vals[idx], vecs[idx]


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
    order = np.argsort(np.abs(p))
    # three consecutive points on the same side of zero (monotone in |Q|);
    # fall back to the global |Q|-sorted triple only if one side is short
    side = p[p > 0] if (p > 0).sum() >= 3 else p[p < 0]
    if len(side) >= 3:
        srt = np.argsort(np.abs(side))
        p3, v3 = side[srt[:3]], v[order[np.isin(p, side[srt[:3]])]]
    else:
        p3, v3 = p[order[:3]], v[order[:3]]
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
        if ok0.sum() < 2:
            print(f"mode {mode}: fewer than 2 finished runs, skipping")
            continue
        p_all, t_all = perts[ok0], tensors[i][ok0]
        # sort perts by |pert| so the reference branch is the smallest one
        srt = np.argsort(np.abs(p_all))
        p_use = p_all[srt]
        vals, vecs = diagonalize_tensors(t_all[srt])
        vals, _ = match_branches(vals, vecs)
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
