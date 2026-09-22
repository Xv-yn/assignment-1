"""Convergence study: objective value versus iteration.

Why this exists.  In the first round of experiments every single run
terminated at exactly ``max_iter`` iterations, which means the relative
tolerance never fired and we had no evidence that any algorithm had
actually converged -- so the reported RREs could just as well have been
"whatever the iterate happened to be at iteration 300".  This script
produces that evidence, and it doubles as the efficiency comparison the
marking scheme rewards ("code runs within a feasible time").

Each algorithm minimises a *different* objective, so the raw cost values
are not comparable across algorithms.  We therefore plot the normalised
suboptimality

    (E_t - E_min) / (E_0 - E_min)

on a log axis, which puts every algorithm on the same 1 -> 0 scale and
makes the *rate* of convergence comparable even when the objectives are
not.  The companion panel plots RRE against wall-clock seconds, which is
the practically relevant trade-off.

Usage (from ``code/``):

    python run_convergence.py --dataset orl --data-root data
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from algorithm import ALGORITHMS  # noqa: E402
from algorithm.data_io import load  # noqa: E402
from algorithm.evaluate import relative_reconstruction_error  # noqa: E402
from algorithm.noise import occlusion_noise  # noqa: E402


def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    V, Y, height, width = load(args.dataset, args.data_root, reduce=args.reduce)
    rng = np.random.default_rng(args.seed)
    n_keep = int(round(args.sample_fraction * V.shape[1]))
    idx = rng.choice(V.shape[1], size=n_keep, replace=False)
    V_clean, Y_sub = V[:, idx], Y[idx]
    k = args.k or len(np.unique(Y_sub))

    V_noisy = occlusion_noise(V_clean, height=height, width=width,
                              block=args.block_size, n_blocks=args.n_blocks,
                              rng=np.random.default_rng(args.seed + 1),
                              fraction=args.fraction)

    rows, histories = [], {}
    for name, fn in ALGORITHMS.items():
        print(f'{name} ...', flush=True)
        # tol=0 so the loop always runs the full budget and we can see the
        # whole curve rather than stopping at the first flat stretch.
        W, H, info = fn(V_noisy, k=k, max_iter=args.max_iter, tol=0.0,
                        seed=args.seed, record_history=True)
        histories[name] = info['history']
        rre = relative_reconstruction_error(V_clean, W, H)
        for it, cost in info['history']:
            rows.append({'algorithm': name, 'iteration': it, 'cost': cost})
        print(f'  final RRE={rre:.4f}  {info["seconds"]:.1f}s  '
              f'{info["iterations"]} iters')

    with (out_dir / f'{args.dataset}_convergence.csv').open(
            'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['algorithm', 'iteration', 'cost'])
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    for name, hist in histories.items():
        its = np.array([h[0] for h in hist], dtype=float)
        cost = np.array([h[1] for h in hist], dtype=float)
        c_min, c_0 = cost.min(), cost[0]
        denom = max(c_0 - c_min, 1e-12)
        ax.semilogy(its, np.maximum((cost - c_min) / denom, 1e-8),
                    label=name, linewidth=1.6)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Normalised suboptimality $(E_t-E_{\min})/(E_0-E_{\min})$')
    ax.set_title(f'{args.dataset.upper()}: convergence of the five objectives\n'
                 f'(b={args.block_size}, {args.n_blocks} block(s), '
                 f'{args.fraction:.0%} of images corrupted)', fontsize=10)
    ax.grid(True, alpha=0.25, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f'{args.dataset}_convergence.png', dpi=200)
    plt.close(fig)
    print(f'\nwrote {out_dir / f"{args.dataset}_convergence.png"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['orl', 'yaleb'], required=True)
    p.add_argument('--data-root', default='data')
    p.add_argument('--output', default='results')
    p.add_argument('--reduce', type=int, default=None)
    p.add_argument('--block-size', type=int, default=10)
    p.add_argument('--n-blocks', type=int, default=1)
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--k', type=int, default=None)
    p.add_argument('--sample-fraction', type=float, default=0.90)
    p.add_argument('--max-iter', type=int, default=300)
    p.add_argument('--seed', type=int, default=2026)
    run(p.parse_args())
