"""Qualitative figures for the report.

Three figures per dataset, each answering a question a marker will ask:

  ``<ds>_noise_demo.png``
      "Show the original image as well as the image contaminated by
      noise" -- required explicitly by the brief.  We show one clean face
      and the same face under every noise level used in the experiments,
      so the reader can see what b=5 versus b=20 actually means.

  ``<ds>_reconstruction.png``
      What the numbers mean.  Same occluded face reconstructed by all five
      algorithms.  A low RRE should be visible as "the white square is
      gone"; this figure is what turns the RRE table into an argument.

  ``<ds>_basis.png``
      The learned basis W (first few columns) per algorithm.  Occlusion
      damages a non-robust basis in a very characteristic way -- bright
      square-shaped blobs get absorbed into the parts -- and showing that
      directly explains *why* the Frobenius baseline loses.

Usage (from ``code/``):

    python make_figures.py --dataset orl --data-root data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from algorithm import ALGORITHMS  # noqa: E402
from algorithm.data_io import load  # noqa: E402
from algorithm.evaluate import relative_reconstruction_error  # noqa: E402
from algorithm.noise import occlusion_noise  # noqa: E402


def _show(ax, vec, height, width, title=None, fontsize=8):
    ax.imshow(vec.reshape(height, width), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=fontsize)


def noise_demo(V, height, width, dataset, out_path, seed, conditions):
    """One clean face plus the same face under every noise condition."""
    clean = V[:, [0]]
    n = len(conditions) + 1
    fig, axes = plt.subplots(1, n, figsize=(1.7 * n, 2.4))

    _show(axes[0], clean[:, 0], height, width, 'Original')
    for ax, (block, n_blocks) in zip(axes[1:], conditions):
        noisy = occlusion_noise(clean, height=height, width=width,
                                block=block, n_blocks=n_blocks,
                                rng=np.random.default_rng(seed + block * 31
                                                          + n_blocks))
        _show(ax, noisy[:, 0], height, width,
              f'$b$={block}, {n_blocks} block' + ('s' if n_blocks > 1 else ''))

    fig.suptitle(f'{dataset}: occlusion noise used in the experiments',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def reconstruction_and_basis(V, Y, height, width, dataset, out_dir, args):
    """Fit every algorithm once and render reconstructions + bases."""
    rng = np.random.default_rng(args.seed)
    n_keep = int(round(args.sample_fraction * V.shape[1]))
    idx = rng.choice(V.shape[1], size=n_keep, replace=False)
    V_clean, Y_sub = V[:, idx], Y[idx]
    k = len(np.unique(Y_sub))

    V_noisy = occlusion_noise(V_clean, height=height, width=width,
                              block=args.block_size, n_blocks=args.n_blocks,
                              rng=np.random.default_rng(args.seed + 1),
                              fraction=args.fraction)

    show_cols = list(range(0, min(3 * 40, V_clean.shape[1]), 40))[:3]
    results = {}
    for name, fn in ALGORITHMS.items():
        print(f'fitting {name} ...', flush=True)
        W, H, info = fn(V_noisy, k=k, max_iter=args.max_iter, tol=args.tol,
                        seed=args.seed)
        results[name] = (W, H, relative_reconstruction_error(V_clean, W, H))
        print(f'  RRE={results[name][2]:.4f}  {info["seconds"]:.1f}s')

    # --- reconstructions -------------------------------------------------
    n_rows = len(show_cols)
    n_cols = 2 + len(results)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.55 * n_cols, 1.9 * n_rows))
    axes = np.atleast_2d(axes)
    for r, col in enumerate(show_cols):
        _show(axes[r, 0], V_clean[:, col], height, width,
              'Original' if r == 0 else None)
        _show(axes[r, 1], V_noisy[:, col], height, width,
              'Occluded' if r == 0 else None)
        for c, (name, (W, H, rre)) in enumerate(results.items(), start=2):
            _show(axes[r, c], (W @ H)[:, col], height, width,
                  f'{name}\nRRE={rre:.3f}' if r == 0 else None, fontsize=7)
    fig.suptitle(f'{dataset}: reconstruction of occluded faces '
                 f'($b$={args.block_size}, {args.n_blocks} block(s), '
                 f'{args.fraction:.0%} corrupted)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f'{args.dataset}_reconstruction.png', dpi=220,
                bbox_inches='tight')
    plt.close(fig)

    # --- learned bases ---------------------------------------------------
    n_basis = args.n_basis
    fig, axes = plt.subplots(len(results), n_basis,
                             figsize=(1.35 * n_basis, 1.6 * len(results)))
    axes = np.atleast_2d(axes)
    for r, (name, (W, _H, rre)) in enumerate(results.items()):
        # show the components carrying the most energy, for comparability
        order = np.argsort(-np.linalg.norm(W, axis=0))[:n_basis]
        for c, comp in enumerate(order):
            _show(axes[r, c], W[:, comp], height, width)
        axes[r, 0].set_ylabel(name, fontsize=7, rotation=0, ha='right',
                              va='center', labelpad=34)
    fig.suptitle(f'{dataset}: leading basis images of $W$ learned from '
                 f'occluded data', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f'{args.dataset}_basis.png', dpi=220,
                bbox_inches='tight')
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['orl', 'yaleb'], required=True)
    p.add_argument('--data-root', default='data')
    p.add_argument('--output', default='results')
    p.add_argument('--reduce', type=int, default=None)
    p.add_argument('--block-size', type=int, default=10)
    p.add_argument('--n-blocks', type=int, default=1)
    p.add_argument('--fraction', type=float, default=1.0)
    p.add_argument('--n-basis', type=int, default=6)
    p.add_argument('--sample-fraction', type=float, default=0.90)
    p.add_argument('--max-iter', type=int, default=300)
    p.add_argument('--tol', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--skip-fits', action='store_true',
                   help='only regenerate the cheap noise-demo figure')
    args = p.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    V, Y, height, width = load(args.dataset, args.data_root, reduce=args.reduce)
    label = {'orl': 'ORL', 'yaleb': 'Extended YaleB'}[args.dataset]
    print(f'{args.dataset}: V={V.shape}  classes={len(np.unique(Y))}')

    noise_demo(V, height, width, label,
               out_dir / f'{args.dataset}_noise_demo.png', args.seed,
               conditions=[(5, 1), (10, 1), (15, 1), (20, 1), (10, 2), (10, 3)])
    print(f'wrote {out_dir / f"{args.dataset}_noise_demo.png"}')

    if not args.skip_fits:
        reconstruction_and_basis(V, Y, height, width, label, out_dir, args)
        print(f'wrote {out_dir / f"{args.dataset}_reconstruction.png"} and '
              f'{out_dir / f"{args.dataset}_basis.png"}')


if __name__ == '__main__':
    main()
