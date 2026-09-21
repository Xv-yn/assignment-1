"""Create original-vs-occluded examples for the report.

Place this file in the repo's `code/` directory and run from there:

    py make_noise_demo.py --data-root ../data/data

Outputs:
    results/orl_noise_demo.png
    results/yaleb_noise_demo.png

This script is fast: it only loads the datasets and adds occlusion noise.
It does NOT run NMF.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from algorithm.data_io import load_orl, load_yaleb
from algorithm.noise import occlusion_noise


def save_demo(V, height, width, dataset_name, output_path, seed, block, n_blocks):
    # Use a deterministic, representative image.
    image_index = 0
    clean = V[:, [image_index]]

    rng = np.random.default_rng(seed)
    noisy = occlusion_noise(
        clean,
        height=height,
        width=width,
        block=block,
        n_blocks=n_blocks,
        rng=rng,
    )

    clean_img = clean[:, 0].reshape(height, width)
    noisy_img = noisy[:, 0].reshape(height, width)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))

    axes[0].imshow(clean_img, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(noisy_img, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title(f"Occluded (b={block}, blocks={n_blocks})")
    axes[1].axis("off")

    fig.suptitle(dataset_name)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main(args):
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    V_orl, _ = load_orl(args.data_root, reduce=3)
    save_demo(
        V_orl,
        height=112 // 3,
        width=92 // 3,
        dataset_name="ORL",
        output_path=out_dir / "orl_noise_demo.png",
        seed=args.seed,
        block=args.block_size,
        n_blocks=args.n_blocks,
    )

    V_yale, Y_yale = load_yaleb(args.data_root, reduce=4)
    if len(np.unique(Y_yale)) != 38 or V_yale.shape[1] != 2414:
        raise ValueError(
            f"Unexpected YaleB data: V={V_yale.shape}, classes={len(np.unique(Y_yale))}. "
            "Expected 2414 images and 38 subjects."
        )

    save_demo(
        V_yale,
        height=192 // 4,
        width=168 // 4,
        dataset_name="Extended YaleB",
        output_path=out_dir / "yaleb_noise_demo.png",
        seed=args.seed + 1,
        block=args.block_size,
        n_blocks=args.n_blocks,
    )

    print(f"Saved: {out_dir / 'orl_noise_demo.png'}")
    print(f"Saved: {out_dir / 'yaleb_noise_demo.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="../data/data")
    parser.add_argument("--output", default="results")
    parser.add_argument("--block-size", type=int, default=10)
    parser.add_argument("--n-blocks", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    main(parser.parse_args())
