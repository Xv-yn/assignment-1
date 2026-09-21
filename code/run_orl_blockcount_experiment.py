"""ORL experiment: robustness vs number of occlusion blocks.

Place this file in the repo's `code/` directory, next to the `algorithm/` package,
then run from `code/`:

    py run_orl_blockcount_experiment.py --data-root ../data/data

Protocol:
- ORL dataset resized by factor 3
- randomly sample 90% of images
- repeat 5 times
- fixed occlusion block size = 10 pixels
- vary number of blocks = 1, 2, 3
- compare all algorithms registered in algorithm.ALGORITHMS
- report mean/std RRE and save CSV + plot
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithm.data_io import load_orl
from algorithm.noise import occlusion_noise
from algorithm.evaluate import clustering_metrics, relative_reconstruction_error
from algorithm import ALGORITHMS


def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    V, Y = load_orl(args.data_root, reduce=3)
    height, width = 112 // 3, 92 // 3  # 37 x 30
    assert V.shape[0] == height * width

    print(f"ORL loaded: V={V.shape}, classes={len(np.unique(Y))}")
    print(f"Image size after resize: {height}x{width}")
    print(f"Fixed block size: {args.block_size}")

    rows = []

    for run_id in range(args.runs):
        subset_rng = np.random.default_rng(args.seed + run_id)
        n_keep = int(round(args.sample_fraction * V.shape[1]))
        idx = subset_rng.choice(V.shape[1], size=n_keep, replace=False)

        V_clean = V[:, idx]
        Y_sub = Y[idx]
        k = len(np.unique(Y_sub))

        for n_blocks in args.block_counts:
            noise_rng = np.random.default_rng(
                args.seed + 10_000 * run_id + 100 * n_blocks + args.block_size
            )
            V_noisy = occlusion_noise(
                V_clean,
                height=height,
                width=width,
                block=args.block_size,
                n_blocks=n_blocks,
                rng=noise_rng,
            )

            for alg_name, alg_fn in ALGORITHMS.items():
                print(
                    f"run={run_id + 1}/{args.runs} "
                    f"blocks={n_blocks} alg={alg_name}"
                )

                W, H, info = alg_fn(
                    V_noisy,
                    k=k,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    seed=args.seed + run_id,
                    verbose=False,
                )

                row = {
                    "run": run_id,
                    "block_size": args.block_size,
                    "n_blocks": n_blocks,
                    "algorithm": alg_name,
                    "rre": float(relative_reconstruction_error(V_clean, W, H)),
                    "iterations": info["iterations"],
                    "seconds": info["seconds"],
                }
                if args.include_clustering:
                    acc, nmi = clustering_metrics(H, Y_sub, seed=args.seed + run_id)
                    row["accuracy"] = float(acc)
                    row["nmi"] = float(nmi)
                rows.append(row)

    raw_csv = out_dir / "orl_blockcount_raw.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for n_blocks in args.block_counts:
        for alg_name in ALGORITHMS:
            values = np.array([
                r["rre"]
                for r in rows
                if r["n_blocks"] == n_blocks and r["algorithm"] == alg_name
            ])

            item = {
                "block_size": args.block_size,
                "n_blocks": n_blocks,
                "algorithm": alg_name,
                "mean_rre": float(values.mean()),
                "std_rre": (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                ),
            }
            if args.include_clustering:
                acc_values = np.array([
                    r["accuracy"]
                    for r in rows
                    if r["n_blocks"] == n_blocks and r["algorithm"] == alg_name
                ])
                nmi_values = np.array([
                    r["nmi"]
                    for r in rows
                    if r["n_blocks"] == n_blocks and r["algorithm"] == alg_name
                ])
                item.update({
                    "mean_accuracy": float(acc_values.mean()),
                    "std_accuracy": (
                        float(acc_values.std(ddof=1)) if len(acc_values) > 1 else 0.0
                    ),
                    "mean_nmi": float(nmi_values.mean()),
                    "std_nmi": (
                        float(nmi_values.std(ddof=1)) if len(nmi_values) > 1 else 0.0
                    ),
                })
            summary.append(item)

    summary_csv = out_dir / "orl_blockcount_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)

    fig, ax = plt.subplots(figsize=(7, 5))

    for alg_name in ALGORITHMS:
        means = []
        stds = []

        for n_blocks in args.block_counts:
            item = next(
                x for x in summary
                if x["n_blocks"] == n_blocks and x["algorithm"] == alg_name
            )
            means.append(item["mean_rre"])
            stds.append(item["std_rre"])

        ax.errorbar(
            args.block_counts,
            means,
            yerr=stds,
            marker="o",
            capsize=3,
            label=alg_name,
        )

    ax.set_xlabel("Number of occlusion blocks")
    ax.set_ylabel("Relative Reconstruction Error (RRE)")
    ax.set_title(
        f"ORL: Robustness to Increasing Number of Occlusions "
        f"(block size={args.block_size})"
    )
    ax.set_xticks(args.block_counts)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    plot_path = out_dir / "orl_rre_vs_block_count.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print("\nSummary (mean ± std RRE):")
    for n_blocks in args.block_counts:
        print(f"\nblocks={n_blocks}")
        for alg_name in ALGORITHMS:
            item = next(
                x for x in summary
                if x["n_blocks"] == n_blocks and x["algorithm"] == alg_name
            )
            print(
                f"  {alg_name:12s}: "
                f"{item['mean_rre']:.4f} ± {item['std_rre']:.4f}"
            )

    print(f"\nSaved raw results: {raw_csv}")
    print(f"Saved summary:     {summary_csv}")
    print(f"Saved plot:        {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="../data/data",
        help="folder containing ORL and CroppedYaleB",
    )
    parser.add_argument("--output", default="results")
    parser.add_argument(
        "--block-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument("--include-clustering", action="store_true",
                        help="also compute accuracy and NMI from H via K-means")
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sample-fraction", type=float, default=0.90)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)

    run(parser.parse_args())
