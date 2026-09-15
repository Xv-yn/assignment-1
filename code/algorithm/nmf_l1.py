"""Algorithm 2: L1-norm based NMF (Ding, Li & Jordan, NeurIPS 2008) --
NOT taught in this course.

    min_{W,H >= 0}  ||V - WH||_1 = sum_ij |V_ij - (WH)_ij|

The L1 elementwise norm ignores the magnitude of an outlier once it is
"far enough", making the factorization robust to salt-and-pepper /
occlusion corruption.

MM derivation: majorize each |r| by its tangent parabola
    |r| <= r^2/(2s) + s/2   (s = |r^t| from the previous iterate),
giving a weighted-Frobenius surrogate with elementwise weights
    D_ij = 1 / |V_ij - (WH^t)_ij|.
MU updates (per iteration, reweighting -> IRLS-style):

    W <- W * ((D.V) H^T) / ((D.WH) H^T)
    H <- H * (W^T (D.V)) / (W^T (D.WH))
"""
import numpy as np

from .common import EPS, weighted_mu_nmf


def _cost(V, R):
    return np.abs(R).sum()


def _weights(V, WH):
    # elementwise weights; clipped for numerical stability
    return 1.0 / (np.abs(V - WH) + 1e-3)


def nmf_l1(V, k, max_iter=500, tol=1e-5, seed=0, verbose=False):
    """L1-norm NMF. Returns (W, H, info)."""
    return weighted_mu_nmf(V, k, _cost, weights_fn=_weights,
                           max_iter=max_iter, tol=tol, seed=seed,
                           verbose=verbose)
