#!/usr/bin/env python3
"""plot_pert_convergence.py -- D vs displacement with residuals from pack npz.

Loads the .npz written by pack_perturbation_runs.py, diagonalises each ZFS
tensor, matches eigenvector branches across perturbations (by overlap with
the smallest-|pert| reference), and plots each principal component against
the perturbation scale together with the residuals from a linear fit.

Usage (anywhere, needs numpy + matplotlib):
    python plot_pert_convergence.py pert_runs.npz [-m MODE] [-o out.png]
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


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Plot ZFS principal components vs perturbation scale with linear-fit residuals"
    )
    ap.add_argument("npz", type=Path, help="npz from pack_perturbation_runs.py")
    ap.add_argument("-m", "--mode", type=int, action="append", help="plot only this mode (repeatable; default: all)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="output png (default: <npz stem>_convergence.png)")
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

    wanted = args.mode if args.mode else [int(m) for m in modes]
    missing = [m for m in wanted if m not in set(modes.tolist())]
    if missing:
        print(f"warning: modes {missing} not in npz", file=sys.stderr)
    if args.output is None:
        args.output = args.npz.with_name(args.npz.stem + "_convergence.png")

    for mode in wanted:
        if mode not in set(modes.tolist()):
            continue
        i = int(np.where(modes == mode)[0][0])
        # sort perts by |pert| so the reference branch is the smallest one
        srt = np.argsort(np.abs(perts))
        p_sorted = perts[srt]
        vals, vecs = diagonalize_tensors(tensors[i][srt])
        # skip NaN slots (unfinished runs)
        ok = ~np.isnan(vals).any(axis=1)
        if ok.sum() < 2:
            print(f"mode {mode}: fewer than 2 finished runs, skipping")
            continue
        vals, vecs = vals[ok], vecs[ok]
        p_use = p_sorted[ok]
        vals, _ = match_branches(vals, vecs)

        ncols = 2
        fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
        for k in range(3):
            a, b, y_fit, res = linear_fit(p_use, vals[:, k])
            axes[0, 0].plot(p_use, vals[:, k], "o-", label=f"PC {k + 1}")
            axes[1, 0].plot(p_use, res, "o-")
            # twin panel: D vs |p| (magnitude, catches even-order contamination)
            a2, b2, y_fit2, res2 = linear_fit(np.abs(p_use), vals[:, k])
            axes[0, 1].plot(np.abs(p_use), vals[:, k], "o-")
            axes[1, 1].plot(np.abs(p_use), res2, "o-")
            print(
                f"mode {mode} PC {k + 1}: slope {b:.4g} MHz/unit-pert, max |res| {np.max(np.abs(res)):.4g} MHz (signed)"
            )
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
