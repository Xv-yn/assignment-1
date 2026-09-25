"""Hyper-parameter sensitivity studies.

Two of the algorithms carry a constant that was chosen by hand, and a
marker is entitled to ask whether the conclusions survive a different
choice.  This script answers that question with data instead of prose:

  1. ``--study delta``  -- the IRLS smoothing constant of L1-NMF,
     D = 1 / (|r| + delta).  Too small and the weights blow up on the
     well-fitted pixels; too large and the loss is quadratic everywhere
     and L1-NMF degenerates into the Frobenius baseline.  The sweep shows
     the plateau in between, i.e. that the default is not a lucky pick.

  2. ``--study lam``    -- the L1 penalty of the sparse-outlier model,
     which is the residual magnitude above which a pixel is declared an
     outlier.  Also reports the fraction of entries actually flagged, so
     the threshold can be read against the true occlusion density.

  3. ``--study rank``   -- the rank k.  Not a robustness knob but a
     modelling choice; the brief's protocol fixes k to the number of
     classes, and this sweep documents how much that choice costs.

Usage (from ``code/``):

    python run_sensitivity.py --study delta --dataset orl --data-root data
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
from algorithm.nmf_hypersurface import nmf_hypersurface  # noqa: E402
from algorithm.nmf_l1 import nmf_l1  # noqa: E402
from algorithm.nmf_l1_reg import nmf_l1_reg  # noqa: E402
from algorithm.noise import occlusion_noise  # noqa: E402

#: iteration budget used to produce every number in the report.  These are the
#: values run_all.sh passes; making them the DEFAULT means a bare
#: `python <script>.py --dataset orl` reproduces the committed results instead
#: of silently running a shorter, different experiment.
REPORT_MAX_ITER = {'orl': 400, 'yaleb': 250}


def resolve_max_iter(args):
    if args.max_iter is None:
        args.max_iter = REPORT_MAX_ITER[args.dataset]
    return args


STUDIES = {
    'delta': (r'L1-NMF smoothing constant $\delta$', True),
    'scale': (r'Hypersurface transition scale $c$', True),
    'lam': (r'Outlier penalty $\lambda$', True),
    'rank': (r'Rank $k$', False),
}


def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    xlabel, log_x = STUDIES[args.study]

    V, Y, height, width = load(args.dataset, args.data_root, reduce=args.reduce)
    n_classes = len(np.unique(Y))
    rows = []

    for run_id in range(args.runs):
        rng = np.random.default_rng(args.seed + run_id)
        n_keep = int(round(args.sample_fraction * V.shape[1]))
        idx = rng.choice(V.shape[1], size=n_keep, replace=False)
        V_clean, Y_sub = V[:, idx], Y[idx]
        V_noisy = occlusion_noise(
            V_clean, height=height, width=width, block=args.block_size,
            n_blocks=args.n_blocks,
            rng=np.random.default_rng(args.seed + 7919 * run_id),
            fraction=args.fraction)
        k_default = len(np.unique(Y_sub))

        for value in args.values:
            if args.study == 'delta':
                W, H, info = nmf_l1(V_noisy, k=k_default, delta=value,
                                    max_iter=args.max_iter, tol=args.tol,
                                    seed=args.seed + run_id)
                extra = {'algorithm': 'L1-NMF'}
            elif args.study == 'scale':
                W, H, info = nmf_hypersurface(V_noisy, k=k_default, scale=value,
                                              max_iter=args.max_iter,
                                              tol=args.tol,
                                              seed=args.seed + run_id)
                extra = {'algorithm': 'Hypersurface'}
            elif args.study == 'lam':
                W, H, info = nmf_l1_reg(V_noisy, k=k_default, lam=value,
                                        max_iter=args.max_iter, tol=args.tol,
                                        seed=args.seed + run_id)
                extra = {'algorithm': 'L1-Reg Robust',
                         'outlier_fraction': info['outlier_fraction']}
            else:  # rank
                extra = {}
                for name, fn in ALGORITHMS.items():
                    W, H, info = fn(V_noisy, k=int(value),
                                    max_iter=args.max_iter, tol=args.tol,
                                    seed=args.seed + run_id)
                    rows.append({'run': run_id, 'value': value,
                                 'algorithm': name,
                                 'rre': relative_reconstruction_error(
                                     V_clean, W, H),
                                 'seconds': info['seconds']})
                    print(f'  run={run_id + 1} k={value} {name:14s} '
                          f'RRE={rows[-1]["rre"]:.4f}', flush=True)
                continue

            rre = relative_reconstruction_error(V_clean, W, H)
            rows.append({'run': run_id, 'value': value, 'rre': rre,
                         'seconds': info['seconds'], **extra})
            print(f'  run={run_id + 1} {args.study}={value:g} '
                  f'RRE={rre:.4f}', flush=True)

    csv_path = out_dir / f'{args.dataset}_sensitivity_{args.study}.csv'
    keys = sorted({key for r in rows for key in r})
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    fig, ax = plt.subplots(figsize=(7, 4.6))
    algorithms = sorted({r.get('algorithm', 'L1-NMF') for r in rows})
    for name in algorithms:
        means, stds, xs = [], [], []
        for value in args.values:
            vals = [r['rre'] for r in rows
                    if r['value'] == value and r.get('algorithm', name) == name]
            if not vals:
                continue
            xs.append(value)
            means.append(float(np.mean(vals)))
            stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        ax.errorbar(xs, means, yerr=stds, marker='o', capsize=3, label=name)
    if log_x:
        ax.set_xscale('log')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Relative Reconstruction Error')
    ax.set_title(f'{args.dataset.upper()}: sensitivity to {args.study}',
                 fontsize=10)
    ax.grid(True, alpha=0.25, which='both')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f'{args.dataset}_sensitivity_{args.study}.png', dpi=200)
    plt.close(fig)
    print(f'wrote {csv_path}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--study', choices=list(STUDIES), required=True)
    p.add_argument('--dataset', choices=['orl', 'yaleb'], default='orl')
    p.add_argument('--data-root', default='data')
    p.add_argument('--output', default='results')
    p.add_argument('--reduce', type=int, default=None)
    p.add_argument('--values', type=float, nargs='+', default=None)
    p.add_argument('--block-size', type=int, default=10)
    p.add_argument('--n-blocks', type=int, default=1)
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--runs', type=int, default=3)
    p.add_argument('--sample-fraction', type=float, default=0.90)
    p.add_argument('--max-iter', type=int, default=None,
                   help='iterations used for the reported results: 400 (ORL), 250 (YaleB)')
    p.add_argument('--tol', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=2026)
    args = resolve_max_iter(p.parse_args())
    if args.values is None:
        args.values = {
            'delta': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'lam': [0.01, 0.02, 0.05, 0.1, 0.2, 0.4],
            'scale': [0.01, 0.02, 0.05, 0.1, 0.5, 1.0],
            'rank': [10, 20, 40, 60, 80],
        }[args.study]
    run(args)
