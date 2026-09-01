"""Occlusion noise generation.

Contiguous white blocks (value 255 in 8-bit scale, i.e. 1.0 after
normalization) are pasted at random positions of each image to simulate
extreme outlier corruption. Blocks may overlap within an image.
"""
import numpy as np


def occlusion_noise(V, height, width, block, n_blocks, rng, fill=1.0):
    """Corrupt every column (image) of V with random white square blocks.

    Args:
        V:        (n_pixels, n_images) clean data, pixels in [0, 1].
        height:   image height in pixels (n_pixels = height * width).
        width:    image width in pixels.
        block:    edge length b of each square block (pixels).
        n_blocks: number of blocks per image.
        rng:      numpy Generator for reproducibility.
        fill:     value used to fill occluded pixels (1.0 == 255 in 8-bit).

    Returns:
        V_noisy: corrupted copy of V.
    """
    V_noisy = V.copy()
    r_max = max(height - block + 1, 1)
    c_max = max(width - block + 1, 1)
    for j in range(V.shape[1]):
        for _ in range(n_blocks):
            r0 = int(rng.integers(0, r_max))
            c0 = int(rng.integers(0, c_max))
            rows = np.arange(r0, min(r0 + block, height))
            cols = np.arange(c0, min(c0 + block, width))
            pix_idx = (rows[:, None] * width + cols[None, :]).ravel()
            V_noisy[pix_idx, j] = fill
    return V_noisy
