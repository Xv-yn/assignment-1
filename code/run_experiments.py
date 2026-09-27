"""Main experiment runner: NMF robustness under occlusion noise.

Usage (from the `code/` folder, datasets must sit in ./data):
    python run_experiments.py                     # full sweep, both datasets
    python run_experiments.py --quick           # fast smoke test (ORL only)
    python run_experiments.py --datasets orl --seeds 3

Outputs land in ./results: raw CSV, mean/std tables (.tex) and figures
noise demos, RRE vs (#blocks) per block size, clustering Acc/NMI.
"""
import argparse
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algorithm import ALGORITHMS  # noqa: E402
from algorithm.data_io import load_orl, load_yaleb  # noqa: E402
from algorithm.evaluate import clustering_metrics, relative_reconstruction_error  # noqa: E402
from algorithm.noise import occlusion_noise  # noqa: E402

matplotlib.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 200,
})

DATASETS = {
    'orl':   {'loader': load_orl,   'reduce': 3},
    'yaleb': {'loader': load_yaleb, 'reduce': 4},
}
ALGO_COLORS = {'F-norm MU': '#1f77b4', 'L1-NMF': '#d62728', 'L2,1-NMF': '#2ca02c'}
ALGO_MARKERS = {'F-norm MU': 'o', 'L1-NMF': 's', 'L2,1-NMF': '^'}


def load_with_cache(name, data_root, cache_dir):
    """Load dataset once, cache as .npy for fast reruns."""
    os.makedirs(cache_dir, exist_ok=True)
    fV = os.path.join(cache_dir, f'{name}_V.npy')
    fY = os.path.join(cache_dir, f'{name}_Y.npy')
    if os.path.exists(fV) and os.path.exists(fY):
        return np.load(fV), np.load(fY)
    cfg = DATASETS[name]
    V, Y = cfg['loader'](data_root, reduce=cfg['reduce'])
    np.save(fV, V)
    np.save(fY, Y)
    return V, Y


def make_noise_demo(ds, V, h, w, blocks_grid, out_dir):
    """Figure showing the original image and occlusion-contaminated copies."""
    rng = np.random.default_rng([0, len(ds)])
    rng_ind = int(rng.integers(0, V.shape[1]))
    ncols = 1 + len(blocks_grid)
    fig, axes = plt.subplots(1, ncols, figsize=(2.6 * ncols, 2.6 / h * w))
    if ncols == 1:
        axes = [axes]
    axes[0].imshow(V[:, rng_ind].reshape(h, w), cmap='gray')
    axes[0].set_title('Original')
    for ax, b in zip(axes[1:], blocks_grid):
        noisy = occlusion_noise(V[:, [rng_ind]], h, w, b, len(blocks_grid), rng)
        ax.imshow(noisy[:, 0].reshape(h, w), cmap='gray')
        ax.set_title(f'b={b}, {len(blocks_grid)} blocks')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(f'{ds.upper()}: occlusion noise demo')
    fig.tight_layout()
    path = os.path.join(out_dir, f'fig_noise_{ds}.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def run_dataset(ds, args, out_dir, block_sizes, n_blocks_grid):
    print(f'==> Dataset {ds}: loading ...')
    V, Y = load_with_cache(ds, args.data_root, os.path.join(args.out, 'cache'))
    n_classes = len(np.unique(Y))
    # original (width, height) -> downscaled dims used by the loader
    cfg = DATASETS[ds]
    orig_wh = {'orl': (92, 112), 'yaleb': (168, 192)}[ds]
    w, h = orig_wh[0] // cfg['reduce'], orig_wh[1] // cfg['reduce']
    m, n = V.shape
    print(f'    V={V.shape}, k={n_classes}, image {w}x{h}')

    if not args.no_fig_noise:
        make_noise_demo(ds, V, h, w, block_sizes[-2:] if len(block_sizes) > 1 else block_sizes, out_dir)

    rows = []
    csv_path = os.path.join(args.out, f'results_raw_{ds}.csv')
    if os.path.exists(csv_path) and not args.overwrite:
        print(f'    reusing cached {csv_path} (use --overwrite to rerun)')
        return csv_path

    print(f'    grid: blocks b={block_sizes} x n_blocks={n_blocks_grid} '
          f'x {args.seeds} seeds')
    for b in block_sizes:
        for nb in n_blocks_grid:
            for seed in range(args.seeds):
                ss = np.random.SeedSequence(seed)
                rng_noise, _ = ss.spawn(2)
                V_noisy = occlusion_noise(V, h, w, b, nb,
                                          np.random.default_rng(rng_noise))
                subsample = None
                if args.subsample < 1.0:
                    rs = np.random.default_rng([seed, 7])
                    idx = rs.choice(n, size=int(np.floor(args.subsample * n)),
                                   replace=False)
                    subsample = np.sort(idx)
                # train + evaluate on the same 90% subsample
                if subsample is not None:
                    Vs = V_noisy[:, subsample]
                    Vhat_s = V[:, subsample]
                    Y_s = Y[subsample]
                else:
                    Vs, Vhat_s, Y_s = V_noisy, V, Y
                for a_i, (aname, algo) in enumerate(ALGORITHMS.items()):
                    W, H, info = algo(Vs, n_classes,
                                     max_iter=args.max_iter, tol=args.tol,
                                     seed=seed * 100 + a_i)
                    rre = relative_reconstruction_error(Vhat_s, W, H)
                    acc, nmi = clustering_metrics(H, Y_s, seed=seed)
                    rows.append({'dataset': ds, 'block': b, 'n_blocks': nb,
                               'seed': seed, 'algorithm': aname,
                               'iters': info['iterations'],
                               'secs': round(info['seconds'], 2),
                               'rre': round(rre, 5),
                               'acc': round(acc, 4), 'nmi': round(nmi, 4)})
                    print(f'      b={b:>2} nb={nb} seed={seed} {aname:<9} '
                          f'iters={info["iterations"]:>4} {info["seconds"]:5.1f}s '
                          f'RRE={rre:.4f} ACC={acc:.4f} NMI={nmi:.4f}')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f'    wrote {csv_path} ({len(rows)} rows)')
    return csv_path


def load_rows(csv_path):
    out = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            for k in ('block', 'n_blocks', 'seed', 'iters'):
                r[k] = int(r[k])
            for k in ('secs', 'rre', 'acc', 'nmi'):
                r[k] = float(r[k])
            out.append(r)
    return out


def agg(rows, algo):
    """mean, std over seeds for each (block, n_blocks) config."""
    sel = [r for r in rows if r['algorithm'] == algo]
    blocks = sorted({r['block'] for r in sel})
    nbs = sorted({r['n_blocks'] for r in sel})
    out = {}
    for b in blocks:
        for nb in nbs:
            vals = np.array([r['rre'] for r in sel if r['block'] == b and r['n_blocks'] == nb])
            out[(b, nb)] = (vals.mean(), vals.std(),
                            np.mean([r['acc'] for r in sel if r['block'] == b and r['n_blocks'] == nb]),
                            np.std([r['acc'] for r in sel if r['block'] == b and r['n_blocks'] == nb]),
                            np.mean([r['nmi'] for r in sel if r['block'] == b and r['n_blocks'] == nb]),
                            np.std([r['nmi'] for r in sel if r['block'] == b and r['n_blocks'] == nb]))
    return out


def plot_rre(ds, rows, out_dir):
    blocks = sorted({r['block'] for r in rows})
    nbs = sorted({r['n_blocks'] for r in rows})
    ncols = min(len(blocks), 2)
    nrows = int(np.ceil(len(blocks) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                            squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for b, ax in zip(blocks, axes.flat):
        ax.axis('on')
        for a in ALGORITHMS:
            m = agg(rows, a)
            ys = [m[(b, nb)][0] for nb in nbs]
            es = [m[(b, nb)][1] for nb in nbs]
            ax.errorbar(nbs, ys, yerr=es, marker=ALGO_MARKERS[a],
                       color=ALGO_COLORS[a], capsize=2, label=a)
        ax.set_title(f'block size b={b}')
        ax.set_xlabel('# occlusion blocks')
        ax.set_ylabel('RRE')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(labels), framealpha=1)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    path = os.path.join(out_dir, f'fig_rre_{ds}.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def plot_clustering(ds, rows, out_dir):
    nbs = sorted({r['n_blocks'] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    for ax, key, name in zip(axes, ('acc', 'nmi'), ('Accuracy', 'NMI')):
        for a in ALGORITHMS:
            sel = [r for r in rows if r['algorithm'] == a]
            xs, ys, es = [], [], []
            for nb in nbs:
                vals = np.array([r[key] for r in sel if r['n_blocks'] == nb])
                xs.append(nb)
                ys.append(vals.mean())
                es.append(vals.std())
            ax.errorbar(xs, ys, yerr=es, marker=ALGO_MARKERS[a],
                       color=ALGO_COLORS[a], capsize=2, label=a)
        ax.set_xlabel('# occlusion blocks (avg over block sizes)')
        ax.set_ylabel(name)
    axes[0].legend(framealpha=1)
    fig.tight_layout()
    path = os.path.join(out_dir, f'fig_clustering_{ds}.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def latex_table(ds, rows, metric, out_dir):
    """Table: rows = algorithms, columns = (block b, n_blocks nb)."""
    blocks = sorted({r['block'] for r in rows})
    nbs = sorted({r['n_blocks'] for r in rows})
    header = ' & '.join(f'({b},{nb})' for b in blocks for nb in nbs)
    lines = [
        r'\begin{table}[t]',
        r'\centering',
        rf'\caption{{{ds.upper()}: {metric.upper()} (mean $\pm$ std over seeds) '
        rf'for occlusion block size $b$ and number of blocks $n_b$.}}',
        r'\begin{tabular}{l' + 'c' * (len(blocks) * len(nbs)) + '}',
        r'\toprule',
        f'Algorithm & {header} \\\\',
        r'\midrule',
    ]
    for a in ALGORITHMS:
        m = agg(rows, a)
        cells = []
        for b in blocks:
            for nb in nbs:
                r, s, acc, accs, nmi, nmis = m[(b, nb)]
                val = {'rre': (r, s), 'acc': (acc, accs), 'nmi': (nmi, nmis)}[metric]
                cells.append(f'{val[0]:.3f} $\\pm$ {val[1]:.3f}')
        lines.append(f'{a.replace(",", ",")} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path = os.path.join(out_dir, f'table_{ds}_{metric}.tex')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'  wrote {path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-root', default='data',
                   help='folder containing ORL/ and CroppedYaleB/')
    p.add_argument('--datasets', default='orl,yaleb')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--max-iter', type=int, default=500)
    p.add_argument('--tol', type=float, default=1e-5)
    p.add_argument('--subsample', type=float, default=0.9)
    p.add_argument('--blocks-orl', default='4,8,12,16')
    p.add_argument('--blocks-yaleb', default='6,12,18,24')
    p.add_argument('--n-blocks', default='1,2,3,4')
    p.add_argument('--out',
                   default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       'results'))
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--no-fig-noise', action='store_true')
    p.add_argument('--quick', action='store_true',
                   help='smoke test: ORL only, 2 block sizes, seeds=1')
    args = p.parse_args()

    if args.quick:
        args.datasets, args.blocks_orl, args.n_blocks = 'orl', '8,16', '1,2'
        args.seeds, args.max_iter = 1, 300

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(os.path.join(args.out, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(args.out, 'tables'), exist_ok=True)

    ds_list = args.datasets.split(',')
    for ds in ds_list:
        blocks = [int(x) for x in (args.blocks_orl if ds == 'orl'
                                    else args.blocks_yaleb).split(',')]
        nbs = [int(x) for x in args.n_blocks.split(',')]
        csv_path = run_dataset(ds, args, os.path.join(args.out, 'figures'),
                               blocks, nbs)
        rows = [r for r in load_rows(csv_path) if r['dataset'] == ds]
        plot_rre(ds, rows, os.path.join(args.out, 'figures'))
        plot_clustering(ds, rows, os.path.join(args.out, 'figures'))
        for metric in ('rre', 'acc', 'nmi'):
            latex_table(ds, rows, metric, os.path.join(args.out, 'tables'))
        secs = np.mean([r['secs'] for r in rows])
        print(f'==> {ds}: done. avg fit time {secs:.1f}s\n')


if __name__ == '__main__':
    main()
