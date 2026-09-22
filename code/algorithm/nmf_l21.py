"""Algorithm 3: L2,1-norm based robust NMF (Ding, Li & Jordan, TPAMI 2010)
-- NOT taught in this course.

    min_{W,H >= 0}  ||V - WH||_{2,1}
                     = sum_j sqrt( sum_i (V_ij - (WH)_ij)^2 )

Each column j of V is one image, so the per-column L2,1 norm measures the
*per-image* reconstruction residual.  Heavily occluded images would
dominate a Frobenius objective (squared residual), but under L2,1 their
influence is only linear; the MM reweighting d_j = 1/||v_j - (WH)_j||_2
actively down-weights the most corrupted images -- it is, in effect, a
soft sample-selection mechanism.

IMPORTANT experimental consequence, and the reason `noise.occlusion_noise`
grew a `fraction` argument: the discrimination happens *between columns*.
If every image in the dataset receives the same amount of occlusion, all
residual norms ||r_j||_2 are nearly equal, the weights d_j are nearly
uniform, and a uniform column weighting is absorbed by the scaling
freedom of the factorisation -- L2,1-NMF then reduces numerically to
plain Frobenius NMF.  L2,1 can only demonstrate its robustness when the
corruption is *unevenly distributed over samples*, i.e. when a subset of
the images is corrupted and the rest is clean.

MM: since ||r_j||_2 = min_{d_j>0} ( d_j/2 ||r_j||_2^2 + 1/(2 d_j) ),
with optimum d_j = 1/||r_j||_2, the surrogate is column-weighted
Frobenius:
    E(W,H) <= sum_j d_j/2 ||V_j - (WH)_j||_2^2 + const.
MU updates:
    W <- W * ((V D) H^T) / (((WH) D) H^T)
    H <- H * (W^T (V D)) / (W^T ((WH) D)),   D = diag(d_j).
"""
import numpy as np

from .common import weighted_mu_nmf

DEFAULT_DELTA = 1e-6


def _cost(V, R):
    return float(np.sqrt((R * R).sum(axis=0)).sum())


def make_weights(delta):
    def _weights(V, WH):
        # column-wise weights d_j = 1 / ||v_j - (WH)_j||_2, shape (1, n)
        col_norm = np.sqrt(((V - WH) ** 2).sum(axis=0, keepdims=True))
        return 1.0 / (col_norm + delta)
    return _weights


def nmf_l21(V, k, max_iter=500, tol=1e-5, seed=0, verbose=False,
            delta=DEFAULT_DELTA, record_history=False):
    """L2,1-norm robust NMF. Returns (W, H, info)."""
    return weighted_mu_nmf(V, k, _cost, weights_fn=make_weights(delta),
                           max_iter=max_iter, tol=tol, seed=seed,
                           verbose=verbose, record_history=record_history)
