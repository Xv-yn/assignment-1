"""Occlusion noise generation.

Contiguous white blocks (value 255 in the 8-bit scale, i.e. 1.0 after
normalisation to [0, 1]) are pasted at random positions of an image to
simulate extreme, structured outlier corruption. Blocks may overlap
within an image.

Two knobs are required by the assignment brief -- the edge length ``b`` of
each block and the number of blocks per image.  We add a third,
``fraction``: the proportion of *images* that are corrupted at all.  This
third knob matters because the three robust losses we compare react to
corruption at different granularities:

* an element-wise loss (L1, hypersurface) reacts to *how many pixels* are
  corrupted, regardless of how they are spread over the images;
* a column-wise loss (L2,1) reacts to *how many images* are corrupted,
  because its reweighting ``d_j = 1 / ||v_j - (WH)_j||_2`` can only
  discriminate between columns when the columns actually differ in how
  badly they are damaged.

If every image receives an identical amount of occlusion, the L2,1 weights
are near-uniform and the method provably degenerates to plain
Frobenius NMF.  Varying ``fraction`` is therefore what makes the
robustness difference between the element-wise and column-wise families
observable at all.
"""
import numpy as np


def occlusion_noise(V, height, width, block, n_blocks, rng, fill=1.0,
                    fraction=1.0, return_mask=False):
    """Corrupt columns (images) of ``V`` with random white square blocks.

    Args:
        V:        (n_pixels, n_images) clean data, pixels in [0, 1].
        height:   image height in pixels (n_pixels == height * width).
        width:    image width in pixels.
        block:    edge length ``b`` of each square block, in pixels.
                  ``block <= 0`` returns a clean copy (useful as a baseline).
        n_blocks: number of blocks pasted into each corrupted image.
        rng:      numpy Generator, for reproducibility.
        fill:     value written into occluded pixels (1.0 == 255 in 8-bit).
        fraction: proportion of images that get corrupted, in [0, 1].
                  1.0 (default) reproduces the original behaviour where
                  every image is occluded.  The corrupted subset is drawn
                  uniformly at random without replacement.
        return_mask: if True also return a boolean array of shape
                  (n_images,) that is True for corrupted columns.

    Returns:
        V_noisy, or (V_noisy, corrupted_mask) when ``return_mask``.
    """
    V_noisy = V.copy()
    n_images = V.shape[1]
    corrupted = np.zeros(n_images, dtype=bool)

    if block <= 0 or n_blocks <= 0 or fraction <= 0.0:
        return (V_noisy, corrupted) if return_mask else V_noisy

    n_corrupt = int(round(fraction * n_images))
    targets = rng.choice(n_images, size=n_corrupt, replace=False)
    corrupted[targets] = True

    r_max = max(height - block + 1, 1)
    c_max = max(width - block + 1, 1)
    for j in targets:
        for _ in range(n_blocks):
            r0 = int(rng.integers(0, r_max))
            c0 = int(rng.integers(0, c_max))
            rows = np.arange(r0, min(r0 + block, height))
            cols = np.arange(c0, min(c0 + block, width))
            pix_idx = (rows[:, None] * width + cols[None, :]).ravel()
            V_noisy[pix_idx, j] = fill

    return (V_noisy, corrupted) if return_mask else V_noisy


def corrupted_pixel_fraction(V_clean, V_noisy):
    """Proportion of entries actually changed by the occlusion.

    Reported alongside every experiment so that conditions with different
    (b, n_blocks, fraction) can be compared on a common axis: overlapping
    blocks mean the nominal budget ``n_blocks * b^2`` overstates the real
    damage.
    """
    return float(np.mean(V_clean != V_noisy))
