"""ORL experiment: robustness vs occlusion block size.

Place this file in the repo's `code/` directory, next to the `algorithm/` package,
then run from `code/`:

    python run_orl_blocksize_experiment.py --data-root ../data

It follows the assignment's suggested protocol: randomly sample 90% of the
images, repeat each experiment 5 times, and report mean/std RRE.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from algorithm.data_io import load_orl
from algorithm.noise import occlusion_noise
from algorithm.evaluate import relative_reconstruction_error
from algorithm import ALGORITHMS


def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    V, Y = load_orl(args.data_root, reduce=3)
    height, width = 112 // 3, 92 // 3  # 37 x 30
    assert V.shape[0] == height * width

    print(f"ORL loaded: V={V.shape}, classes={len(np.unique(Y))}")
    print(f"Image size after resize: {height}x{width}")

    rows = []

    # Repeat the experiment on independently sampled 90% subsets.
    for run_id in range(args.runs):
        subset_rng = np.random.default_rng(args.seed + run_id)
        n_keep = int(round(args.sample_fraction * V.shape[1]))
        idx = subset_rng.choice(V.shape[1], size=n_keep, replace=False)
        V_clean = V[:, idx]
        Y_sub = Y[idx]
        k = len(np.unique(Y_sub))

        # Same subset for all block sizes in a run; condition-specific noise.
        for block in args.block_sizes:
            noise_rng = np.random.default_rng(args.seed + 10_000 * run_id + block)
            V_noisy = occlusion_noise(
                V_clean,
                height=height,
                width=width,
                block=block,
                n_blocks=1,
                rng=noise_rng,
            )

            for alg_name, alg_fn in ALGORITHMS.items():
                print(f"run={run_id + 1}/{args.runs} block={block:2d} alg={alg_name}")
                W, H, info = alg_fn(
                    V_noisy,
                    k=k,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    seed=args.seed + run_id,
                    verbose=False,
                )
                rre = float(relative_reconstruction_error(V_clean, W, H))
                rows.append({
                    "run": run_id,
                    "block_size": block,
                    "algorithm": alg_name,
                    "rre": rre,
                    "iterations": info["iterations"],
                    "seconds": info["seconds"],
                })

    raw_csv = out_dir / "orl_blocksize_raw.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for block in args.block_sizes:
        for alg_name in ALGORITHMS:
            values = np.array([
                r["rre"] for r in rows
                if r["block_size"] == block and r["algorithm"] == alg_name
            ])
            summary.append({
                "block_size": block,
                "algorithm": alg_name,
                "mean_rre": float(values.mean()),
                "std_rre": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            })

    summary_csv = out_dir / "orl_blocksize_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    # Publication-ready-enough plot; do not hard-code colors.
    fig, ax = plt.subplots(figsize=(7, 5))
    for alg_name in ALGORITHMS:
        means, stds = [], []
        for block in args.block_sizes:
            item = next(
                x for x in summary
                if x["block_size"] == block and x["algorithm"] == alg_name
            )
            means.append(item["mean_rre"])
            stds.append(item["std_rre"])
        ax.errorbar(args.block_sizes, means, yerr=stds, marker="o", capsize=3,
                    label=alg_name)

    ax.set_xlabel("Occlusion block size (pixels)")
    ax.set_ylabel("Relative Reconstruction Error (RRE)")
    ax.set_title("ORL: Robustness to Increasing Occlusion Size")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    plot_path = out_dir / "orl_rre_vs_block_size.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print("\nSummary (mean ± std RRE):")
    for block in args.block_sizes:
        print(f"\nblock={block}")
        for alg_name in ALGORITHMS:
            item = next(
                x for x in summary
                if x["block_size"] == block and x["algorithm"] == alg_name
            )
            print(f"  {alg_name:12s}: {item['mean_rre']:.4f} ± {item['std_rre']:.4f}")

    print(f"\nSaved raw results: {raw_csv}")
    print(f"Saved summary:     {summary_csv}")
    print(f"Saved plot:        {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../data",
                        help="folder containing ORL and CroppedYaleB")
    parser.add_argument("--output", default="results")
    parser.add_argument("--block-sizes", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sample-fraction", type=float, default=0.90)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    run(parser.parse_args())
