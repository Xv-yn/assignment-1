"""NMF algorithm implementations.

The factorisation code itself (``common``, ``nmf_*``) uses only numpy and
the Python standard library, as required.  scipy appears once, in
``evaluate`` (Hungarian matching), and scikit-learn only in ``evaluate``
(K-means, accuracy, NMI) -- i.e. exclusively for measuring results, never
for computing them.  ``data_io`` uses Pillow to decode the .pgm files,
matching the provided ``assignment1.ipynb`` tutorial.

The five algorithms span three distinct robustness mechanisms, which is
what makes the comparison in the report informative rather than a list of
near-identical curves:

  * no down-weighting at all           -- F-norm MU          (baseline)
  * per-PIXEL down-weighting           -- L1-NMF, Hypersurface
  * per-IMAGE down-weighting           -- L2,1-NMF
  * explicit sparse OUTLIER variable   -- L1-Reg Robust NMF
"""
from .common import weighted_mu_nmf
from .data_io import DATASETS, load, load_dataset, load_orl, load_yaleb
from .evaluate import clustering_metrics, relative_reconstruction_error
from .noise import corrupted_pixel_fraction, occlusion_noise
from .nmf_mu import nmf_frobenius
from .nmf_l1 import nmf_l1
from .nmf_l21 import nmf_l21
from .nmf_hypersurface import nmf_hypersurface
from .nmf_l1_reg import nmf_l1_reg

#: display name -> algorithm callable.  Every callable has the signature
#: fn(V, k, max_iter, tol, seed, verbose, record_history) -> (W, H, info)
ALGORITHMS = {
    'F-norm MU': nmf_frobenius,
    'L1-NMF': nmf_l1,
    'Hypersurface': nmf_hypersurface,
    'L2,1-NMF': nmf_l21,
    'L1-Reg Robust': nmf_l1_reg,
}

#: which granularity each algorithm down-weights at; used to order and
#: colour the figures consistently across the report
ALGORITHM_FAMILY = {
    'F-norm MU': 'none',
    'L1-NMF': 'pixel',
    'Hypersurface': 'pixel',
    'L2,1-NMF': 'image',
    'L1-Reg Robust': 'outlier-model',
}
