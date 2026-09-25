"""Unified robustness experiment driver for both datasets.

This single script replaces the four near-identical per-dataset,
per-sweep scripts that came before it.  Besides being the thing a marker
actually has to read, merging them buys a large amount of compute: the
three sweeps share conditions (they all pass through b=10, one block,
100% contaminated), and a merged driver fits each *unique* condition once
instead of once per sweep.

Protocol (as prescribed by the assignment brief):
  * randomly sample 90% of the images,
  * repeat the whole experiment 5 times with independent subsets,
  * report mean and standard deviation of every metric across the repeats.

Three noise axes are swept:
  1. block size      b in {0, 5, 10, 15, 20}, one block, all images
  2. number of blocks n in {0, 1, 2, 3, 4},   b = 10, all images
  3. contamination    f in {0.1, 0.25, 0.5, 0.75, 1.0}, b = 15, two blocks

Axis 3 is the one that separates per-pixel robustness from per-image
robustness; see the module docstring of ``algorithm/noise.py``.
b = 0 / n = 0 denote the uncontaminated control condition.

Usage (run from the ``code/`` directory):

    python run_experiment.py --dataset orl   --data-root data
    python run_experiment.py --dataset yaleb --data-root data

``--max-iter`` defaults to the budget used for the reported results
(400 for ORL, 250 for Extended YaleB), so the bare commands above
reproduce the committed CSVs.

Outputs, under ``--output`` (default ``results/``):
    <dataset>_raw.csv                  one row per (run, condition, algorithm)
    <dataset>_<sweep>_summary.csv      mean/std of each metric
    <dataset>_<sweep>_rre.png          RRE   vs the swept axis
    <dataset>_<sweep>_acc.png          Acc   vs the swept axis
    <dataset>_<sweep>_nmi.png          NMI   vs the swept axis
"""
from __future__ import annotations

import argparse
import csv
import itertools
import time
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from algorithm import ALGORITHMS  # noqa: E402
from algorithm.data_io import load  # noqa: E402
from algorithm.evaluate import (  # noqa: E402
    clustering_metrics, relative_reconstruction_error)
from algorithm.noise import corrupted_pixel_fraction, occlusion_noise  # noqa: E402

#: iteration budget used to produce every number in the report.  These are the
#: values run_all.sh passes; making them the DEFAULT means a bare
#: `python <script>.py --dataset orl` reproduces the committed results instead
#: of silently running a shorter, different experiment.
REPORT_MAX_ITER = {'orl': 400, 'yaleb': 250}


def resolve_max_iter(args):
    if args.max_iter is None:
        args.max_iter = REPORT_MAX_ITER[args.dataset]
    return args


# ---------------------------------------------------------------- sweeps

#: sweep name -> (axis column, x-axis label, human-readable description)
SWEEPS = {
    'blocksize': ('block_size', 'Occlusion block edge length $b$ (pixels)',
                  'one block per image, all images corrupted'),
    'blockcount': ('n_blocks', 'Number of occlusion blocks per image',
                   'block size $b=10$, all images corrupted'),
    'contamination': ('fraction', 'Fraction of images corrupted',
                      'block size $b=15$, two blocks per corrupted image'),
}

METRICS = {
    'rre': 'Relative Reconstruction Error (lower is better)',
    'accuracy': 'Clustering accuracy (higher is better)',
    'nmi': 'Normalised Mutual Information (higher is better)',
}


def build_conditions(block_sizes, block_counts, fractions,
                     count_block=10, contam_block=15, contam_blocks=2):
    """Return (conditions, sweep_membership).

    ``conditions`` is a de-duplicated list of ``(block, n_blocks, fraction)``
    triples; ``sweep_membership`` maps a sweep name to the ordered list of
    (axis_value, condition) pairs belonging to it, so that each unique
    condition is fitted exactly once but can appear in several plots.
    """
    membership = {name: [] for name in SWEEPS}
    ordered: list[tuple[int, int, float]] = []

    def add(sweep, axis_value, cond):
        if cond not in ordered:
            ordered.append(cond)
        membership[sweep].append((axis_value, cond))

    for b in block_sizes:
        add('blocksize', b, (b, 1, 1.0) if b > 0 else (0, 0, 0.0))
    for n in block_counts:
        add('blockcount', n, (count_block, n, 1.0) if n > 0 else (0, 0, 0.0))
    # The contamination axis deliberately uses a HEAVIER per-image
    # corruption than the other two axes.  Its purpose is to make the
    # corrupted and clean images differ as much as possible, since that
    # difference is the only signal a column-wise (per-image) robust loss
    # can exploit; a mild corruption spread over few images would leave
    # every column residual roughly equal and hide the effect.
    for f in fractions:
        add('contamination', f,
            (contam_block, contam_blocks, f) if f > 0 else (0, 0, 0.0))

    return ordered, membership


# ------------------------------------------------------------------ core

def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    V, Y, height, width = load(args.dataset, args.data_root,
                               reduce=args.reduce)
    n_classes = len(np.unique(Y))
    print(f'{args.dataset}: V={V.shape}  classes={n_classes}  '
          f'image={height}x{width}')

    algorithms = {name: fn for name, fn in ALGORITHMS.items()
                  if not args.algorithms or name in args.algorithms}
    conditions, membership = build_conditions(
        args.block_sizes, args.block_counts, args.fractions,
        count_block=args.count_block, contam_block=args.contam_block,
        contam_blocks=args.contam_blocks)

    total = args.runs * len(conditions) * len(algorithms)
    print(f'{len(conditions)} unique noise conditions x {len(algorithms)} '
          f'algorithms x {args.runs} runs = {total} factorisations')

    rows = []
    done = 0
    t_start = time.perf_counter()

    for run_id in range(args.runs):
        # Independent 90% subsample per repeat, shared by all conditions in
        # that repeat so that conditions are compared on identical data.
        subset_rng = np.random.default_rng(args.seed + run_id)
        n_keep = int(round(args.sample_fraction * V.shape[1]))
        idx = subset_rng.choice(V.shape[1], size=n_keep, replace=False)
        V_clean, Y_sub = V[:, idx], Y[idx]
        k = args.k if args.k else len(np.unique(Y_sub))

        for block, n_blocks, fraction in conditions:
            # Noise seed depends on the condition so that, e.g., the b=10
            # condition of the block-size sweep and of the block-count
            # sweep are literally the same corrupted matrix.
            noise_rng = np.random.default_rng(
                (args.seed + run_id) * 1_000_003
                + block * 10_007 + n_blocks * 101 + int(round(fraction * 100)))
            V_noisy, corrupted = occlusion_noise(
                V_clean, height=height, width=width, block=block,
                n_blocks=n_blocks, rng=noise_rng, fraction=fraction,
                return_mask=True)
            pixel_frac = corrupted_pixel_fraction(V_clean, V_noisy)

            for alg_name, alg_fn in algorithms.items():
                W, H, info = alg_fn(V_noisy, k=k, max_iter=args.max_iter,
                                    tol=args.tol, seed=args.seed + run_id,
                                    verbose=False)
                # RRE is always measured against the CLEAN matrix: the task
                # is to recover the signal, not to fit the corruption.
                rre = relative_reconstruction_error(V_clean, W, H)
                metrics = ({'accuracy': float('nan'),
                            'accuracy_hungarian': float('nan'),
                            'nmi': float('nan')}
                           if args.skip_clustering
                           else clustering_metrics(H, Y_sub, seed=args.seed))

                rows.append({
                    'run': run_id, 'block_size': block, 'n_blocks': n_blocks,
                    'fraction': fraction, 'corrupted_images': int(corrupted.sum()),
                    'corrupted_pixel_fraction': round(pixel_frac, 6),
                    'algorithm': alg_name, 'k': k, 'rre': rre,
                    'accuracy': metrics['accuracy'],
                    'accuracy_hungarian': metrics['accuracy_hungarian'],
                    'nmi': metrics['nmi'],
                    'iterations': info['iterations'],
                    'converged': int(info['converged']),
                    'final_cost': info['cost'], 'seconds': info['seconds'],
                })

                done += 1
                elapsed = time.perf_counter() - t_start
                eta = elapsed / done * (total - done)
                print(f'[{done:4d}/{total}] run={run_id + 1} '
                      f'b={block} n={n_blocks} f={fraction:.2f} '
                      f'{alg_name:14s} RRE={rre:.4f} '
                      f'({info["seconds"]:.1f}s, ETA {eta / 60:.1f} min)',
                      flush=True)

            # Flush after every condition so a long YaleB job is never lost.
            _write_csv(out_dir / f'{args.dataset}_raw.csv', rows)

    _write_csv(out_dir / f'{args.dataset}_raw.csv', rows)
    print(f'\nraw results -> {out_dir / f"{args.dataset}_raw.csv"}')

    for sweep, pairs in membership.items():
        if len(pairs) < 2:
            continue
        summarise_and_plot(args.dataset, sweep, pairs, rows, algorithms,
                           out_dir)

    print(f'\ntotal wall clock: {(time.perf_counter() - t_start) / 60:.1f} min')


# ------------------------------------------------------- summary + plots

def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarise_and_plot(dataset, sweep, pairs, rows, algorithms, out_dir):
    axis_col, xlabel, subtitle = SWEEPS[sweep]
    summary = []

    for (axis_value, cond), alg_name in itertools.product(pairs, algorithms):
        block, n_blocks, fraction = cond
        sel = [r for r in rows
               if r['block_size'] == block and r['n_blocks'] == n_blocks
               and r['fraction'] == fraction and r['algorithm'] == alg_name]
        if not sel:
            continue
        entry = {axis_col: axis_value, 'algorithm': alg_name,
                 'n_runs': len(sel)}
        for metric in METRICS:
            vals = np.array([r[metric] for r in sel], dtype=float)
            entry[f'mean_{metric}'] = float(np.nanmean(vals))
            entry[f'std_{metric}'] = (float(np.nanstd(vals, ddof=1))
                                      if len(vals) > 1 else 0.0)
        entry['mean_seconds'] = float(np.mean([r['seconds'] for r in sel]))
        summary.append(entry)

    _write_csv(out_dir / f'{dataset}_{sweep}_summary.csv', summary)

    xs = [axis_value for axis_value, _ in pairs]
    for metric, ylabel in METRICS.items():
        if all(np.isnan([e[f'mean_{metric}'] for e in summary])):
            continue
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for alg_name in algorithms:
            means, stds = [], []
            for axis_value in xs:
                item = next((e for e in summary
                             if e[axis_col] == axis_value
                             and e['algorithm'] == alg_name), None)
                means.append(np.nan if item is None else item[f'mean_{metric}'])
                stds.append(0.0 if item is None else item[f'std_{metric}'])
            ax.errorbar(xs, means, yerr=stds, marker='o', capsize=3,
                        linewidth=1.6, markersize=5, label=alg_name)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f'{dataset.upper()}: {metric.upper()} vs '
                     f'{axis_col.replace("_", " ")}\n({subtitle})',
                     fontsize=10)
        ax.set_xticks(xs)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f'{dataset}_{sweep}_{metric}.png', dpi=200)
        plt.close(fig)

    print(f'  {sweep}: summary + {len(METRICS)} plots written')


# ------------------------------------------------------------------- cli

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', choices=['orl', 'yaleb'], required=True)
    p.add_argument('--data-root', default='data',
                   help='folder containing ORL/ and CroppedYaleB/ '
                        '(a parent of it also works)')
    p.add_argument('--output', default='results')
    p.add_argument('--reduce', type=int, default=None,
                   help='image downscale factor (default: 3 ORL, 4 YaleB)')
    p.add_argument('--block-sizes', type=int, nargs='+',
                   default=[0, 5, 10, 15, 20])
    p.add_argument('--block-counts', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    p.add_argument('--fractions', type=float, nargs='+',
                   default=[0.1, 0.25, 0.5, 0.75, 1.0])
    p.add_argument('--count-block', type=int, default=10,
                   help='block size held fixed while sweeping block count')
    p.add_argument('--contam-block', type=int, default=15,
                   help='block size used on the contamination-ratio axis')
    p.add_argument('--contam-blocks', type=int, default=2,
                   help='blocks per corrupted image on the contamination axis')
    p.add_argument('--algorithms', nargs='*', default=None,
                   help='subset of algorithm names (default: all five)')
    p.add_argument('--runs', type=int, default=5)
    p.add_argument('--sample-fraction', type=float, default=0.90)
    p.add_argument('--k', type=int, default=None,
                   help='rank; default = number of classes in the subset')
    p.add_argument('--max-iter', type=int, default=None,
                   help='iterations used for the reported results: 400 (ORL), 250 (YaleB)')
    p.add_argument('--tol', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--include-clustering', action='store_true',
                   help='accepted for backwards compatibility with the '
                        'older per-dataset scripts; clustering metrics are '
                        'computed by default here, so this is a no-op.')
    p.add_argument('--skip-clustering', action='store_true',
                   help='skip K-means Acc/NMI (RRE only); much faster')
    run(resolve_max_iter(p.parse_args()))


if __name__ == '__main__':
    main()
