"""Faster Extended YaleB block-count experiment.

This script REUSES the already-computed YaleB result for:
    block_size = 10, n_blocks = 1
from:
    results/yaleb_blocksize_raw.csv

It therefore only runs the missing n_blocks = 2 and 3 conditions, cutting
the number of NMF fits from 45 to 30 while keeping the same 5-run protocol.

Run from the repo's `code/` directory:

    py run_yaleb_blockcount_experiment_fast.py --data-root ../data/data

Expected existing file:
    results/yaleb_blocksize_raw.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithm.data_io import load_yaleb
from algorithm.noise import occlusion_noise
from algorithm.evaluate import clustering_metrics, relative_reconstruction_error
from algorithm import ALGORITHMS


def _read_reused_rows(raw_path: Path, block_size: int):
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Cannot find {raw_path}. Run the YaleB block-size experiment first."
        )

    reused = []
    with raw_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["block_size"]) == block_size and int(row["n_blocks"]) == 1:
                reused.append({
                    "run": int(row["run"]),
                    "block_size": int(row["block_size"]),
                    "n_blocks": int(row["n_blocks"]),
                    "algorithm": row["algorithm"],
                    "rre": float(row["rre"]),
                    "iterations": int(float(row["iterations"])),
                    "seconds": float(row["seconds"]),
                })

    expected = 5 * len(ALGORITHMS)
    if len(reused) != expected:
        raise ValueError(
            f"Expected {expected} reusable rows for block_size={block_size}, "
            f"n_blocks=1, but found {len(reused)}."
        )

    return reused


def run(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    V, Y = load_yaleb(args.data_root, reduce=4)
    height, width = 192 // 4, 168 // 4  # 48 x 42
    assert V.shape[0] == height * width

    print(f"YaleB loaded: V={V.shape}, classes={len(np.unique(Y))}")
    print(f"Image size after resize: {height}x{width}")
    print(f"Fixed block size: {args.block_size}")

    if args.include_clustering:
        rows = []
        missing_counts = list(args.block_counts)
        print("Clustering metrics requested; recomputing all block-count settings.")
    else:
        reused_path = out_dir / "yaleb_blocksize_raw.csv"
        rows = _read_reused_rows(reused_path, args.block_size)
        print(
            f"Reused {len(rows)} existing rows for n_blocks=1 "
            f"from {reused_path}"
        )
        # Only compute the missing block-count settings.
        missing_counts = [b for b in args.block_counts if b != 1]

    for run_id in range(args.runs):
        subset_rng = np.random.default_rng(args.seed + run_id)
        n_keep = int(round(args.sample_fraction * V.shape[1]))
        idx = subset_rng.choice(V.shape[1], size=n_keep, replace=False)

        V_clean = V[:, idx]
        Y_sub = Y[idx]
        k = len(np.unique(Y_sub))

        for n_blocks in missing_counts:
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

    rows.sort(key=lambda r: (r["run"], r["n_blocks"], r["algorithm"]))

    raw_csv = out_dir / "yaleb_blockcount_raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
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

    summary_csv = out_dir / "yaleb_blockcount_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
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
        f"Extended YaleB: Robustness to Increasing Number of Occlusions "
        f"(block size={args.block_size})"
    )
    ax.set_xticks(args.block_counts)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    plot_path = out_dir / "yaleb_rre_vs_block_count.png"
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
    parser.add_argument("--data-root", default="../data/data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument(
        "--block-counts",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument("--include-clustering", action="store_true",
                        help="also compute accuracy and NMI from H via K-means")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--sample-fraction", type=float, default=0.90)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)

    run(parser.parse_args())
