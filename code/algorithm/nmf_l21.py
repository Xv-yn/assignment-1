"""Algorithm 3: L2,1-norm based robust NMF (Ding, Li & Jordan, TPAMI 2010)
-- NOT taught in this course.

    min_{W,H >= 0}  ||V - WH||_{2,1}
                     = sum_j sqrt( sum_i (V_ij - (WH)_ij)^2 )

Each column j of V is one image, so the per-column L2,1 norm measures the
per-image reconstruction residual. Heavily occluded images would dominate an
Frobenius objective (squared residual), but under L2,1 their influence is
only linear; the MM reweighting d_j = 1/||V_j - (WH)_j||_2 actively
down-weights the most corrupted images.

MM: since ||r_j||_2 = min_{d_j>0} ( d_j/2 ||r_j||_2^2 + 1/(2 d_j) ),
with optimum d_j = 1/||r_j||_2, the surrogate is column-weighted
Frobenius:
    E(W,H) <= sum_j d_j/2 ||V_j - (WH)_j||_2^2 + const.
MU updates:
    W <- W * (V H^T D) / ((WH) H^T D)
    H <- H * (W^T V D) / (W^T W H D),   D = diag(d_j).
"""
import numpy as np

from .common import EPS, weighted_mu_nmf


def _cost(V, R):
    return np.sqrt((R * R).sum(axis=0)).sum()


def _weights(V, WH):
    # column-wise weights d_j = 1 / ||V_j - (WH)_j||_2, shape (n,)
    col_norm = np.sqrt(((V - WH) ** 2).sum(axis=0))
    return 1.0 / (col_norm + 1e-6)


def nmf_l21(V, k, max_iter=500, tol=1e-5, seed=0, verbose=False):
    """L2,1-norm robust NMF. Returns (W, H, info)."""
    return weighted_mu_nmf(V, k, _cost, weights_fn=_weights,
                           max_iter=max_iter, tol=tol, seed=seed,
                           verbose=verbose)
