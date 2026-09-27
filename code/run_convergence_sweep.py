"""Convergence study: how does RRE evolve with the MU iteration budget?

Every experiment in this project stops at a fixed `max_iter` because the
relative-cost tolerance never triggers in practice. That makes the iteration
budget a free hyper-parameter, so this script measures how sensitive each
algorithm's RRE is to it.

A single fit is run per (algorithm, repeat) and the RRE against the *clean*
matrix is traced along the way via the kernel's callback hook, which is far
cheaper than refitting once per budget and gives exactly the same trace.

Run from `code/`:

    python run_convergence_sweep.py --dataset orl
    python run_convergence_sweep.py --dataset yaleb
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from algorithm.data_io import load_orl, load_yaleb
from algorithm.noise import occlusion_noise
from algorithm.evaluate import relative_reconstruction_error
from algorithm import ALGORITHMS

DATASETS = {
    "orl": {"loader": load_orl, "reduce": 3, "shape": (112 // 3, 92 // 3)},
    "yaleb": {"loader": load_yaleb, "reduce": 4, "shape": (192 // 4, 168 // 4)},
}


def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = DATASETS[args.dataset]
    height, width = cfg["shape"]

    V, Y = cfg["loader"](args.data_root, reduce=cfg["reduce"])
    assert V.shape[0] == height * width
    print(f"{args.dataset} loaded: V={V.shape}, classes={len(np.unique(Y))}")
    print(f"tracing RRE every {args.stride} iters up to {args.max_iter}\n")

    rows = []
    for run_id in range(args.runs):
        # Same subsample + noise protocol as the main experiments.
        subset_rng = np.random.default_rng(args.seed + run_id)
        n_keep = int(round(args.sample_fraction * V.shape[1]))
        idx = subset_rng.choice(V.shape[1], size=n_keep, replace=False)
        V_clean = V[:, idx]
        k = len(np.unique(Y[idx]))

        noise_rng = np.random.default_rng(
            args.seed + 10_000 * run_id + args.block_size)
        V_noisy = occlusion_noise(V_clean, height=height, width=width,
                                  block=args.block_size,
                                  n_blocks=args.n_blocks, rng=noise_rng)

        for alg_name, alg_fn in ALGORITHMS.items():
            print(f"run={run_id + 1}/{args.runs} alg={alg_name}")
            trace = []

            def record(iteration, W, H, _t=trace):
                _t.append((iteration,
                           float(relative_reconstruction_error(V_clean, W, H))))

            alg_fn(V_noisy, k=k, max_iter=args.max_iter, tol=args.tol,
                   seed=args.seed + run_id, verbose=False,
                   callback=record, callback_every=args.stride)

            for iteration, rre in trace:
                rows.append({"run": run_id, "dataset": args.dataset,
                             "block_size": args.block_size,
                             "n_blocks": args.n_blocks,
                             "algorithm": alg_name,
                             "iteration": iteration, "rre": rre})
            best_it, best_rre = min(trace, key=lambda t: t[1])
            print(f"    best RRE {best_rre:.4f} @ iter {best_it}; "
                  f"final {trace[-1][1]:.4f} @ iter {trace[-1][0]}")

    raw_csv = out_dir / f"{args.dataset}_convergence_raw.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    iterations = sorted({r["iteration"] for r in rows})
    summary = []
    for alg_name in ALGORITHMS:
        for iteration in iterations:
            vals = np.array([r["rre"] for r in rows
                             if r["algorithm"] == alg_name
                             and r["iteration"] == iteration])
            summary.append({
                "dataset": args.dataset, "algorithm": alg_name,
                "iteration": iteration, "mean_rre": float(vals.mean()),
                "std_rre": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            })

    summary_csv = out_dir / f"{args.dataset}_convergence_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(7, 5))
    for alg_name in ALGORITHMS:
        pts = [s for s in summary if s["algorithm"] == alg_name]
        xs = [p["iteration"] for p in pts]
        means = np.array([p["mean_rre"] for p in pts])
        stds = np.array([p["std_rre"] for p in pts])
        line, = ax.plot(xs, means, label=alg_name)
        ax.fill_between(xs, means - stds, means + stds, alpha=0.15,
                        color=line.get_color())
    ax.axvline(300, linestyle="--", linewidth=1, color="0.4")
    ax.annotate("budget used by\nmain experiments", xy=(300, ax.get_ylim()[1]),
                xytext=(6, -12), textcoords="offset points",
                fontsize=8, color="0.3", va="top")
    ax.set_xlabel("MU iterations")
    ax.set_ylabel("Relative Reconstruction Error (RRE)")
    ax.set_title(f"{args.dataset.upper()}: RRE vs iteration budget "
                 f"(b={args.block_size}, {args.n_blocks} block(s))")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / f"{args.dataset}_rre_vs_iterations.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"\nRRE at selected budgets ({args.dataset}, mean over "
          f"{args.runs} run(s)):")
    checkpoints = [i for i in (100, 300, 600, 1000, 1500, 2000)
                   if i in iterations]
    header = "  ".join(f"{i:>8d}" for i in checkpoints)
    print(f"{'algorithm':12s} {header}")
    for alg_name in ALGORITHMS:
        cells = []
        for i in checkpoints:
            s = next(x for x in summary
                     if x["algorithm"] == alg_name and x["iteration"] == i)
            cells.append(f"{s['mean_rre']:8.4f}")
        print(f"{alg_name:12s} " + "  ".join(cells))

    print(f"\nSaved raw results: {raw_csv}")
    print(f"Saved summary:     {summary_csv}")
    print(f"Saved plot:        {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="orl")
    parser.add_argument("--data-root", default="data",
                        help="folder containing ORL and CroppedYaleB")
    parser.add_argument("--output", default="results")
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--sample-fraction", type=float, default=0.90)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--stride", type=int, default=25)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    run(parser.parse_args())
