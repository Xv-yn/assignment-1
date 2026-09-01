"""Shared machinery for multiplicative-update (MU) NMF algorithms.

All three algorithms minimize a cost of the form

    E(W, H) = sum_ij D_ij * (V_ij - (WH)_ij)^2 + const,   W, H >= 0

where the weight matrix D >= 0 is *fixed* for the iteration. At every
iteration D is recomputed from the current residual so that the weighted
quadratic majorizes the true (possibly non-smooth) cost, touching it at
the current iterate. This majorize-minimize (MM) scheme yields the
multiplicative updates below, whose multiplicative form keeps W, H >= 0.

References:
  - Lee & Seung, NeurIPS 1999 (Frobenius / KL NMF).
  - Lin, Neural Comput. 2007 (convergence of MU).
  - Ding, Li & Jordan, NeurIPS 2008 / IEEE TPAMI 2010 (L1- and
    L2,1-norm based NMF).
"""
import time

import numpy as np

EPS = 1e-10


def initialize(V, k, rng):
    """Random uniform init, scaled so that sum(WH) ~= sum(V)."""
    m, n = V.shape
    scale = np.sqrt(2.0 * V.mean())
    W = rng.uniform(0.0, scale, size=(m, k))
    H = rng.uniform(0.0, scale, size=(k, n))
    total = float(np.dot(W.sum(axis=0), H.sum(axis=1)))
    H *= V.sum() / max(total, EPS)
    return W, H


def weighted_mu_nmf(V, k, cost_fn, weights_fn=None, max_iter=500,
                    tol=1e-5, seed=0, verbose=False):
    """Weighted-Frobenius MM/MU kernel shared by all NMF algorithms.

    Args:
        V:         (m, n) nonnegative noisy data matrix.
        k:         number of components (n_components).
        cost_fn:   (V, R) -> scalar objective, R = V - WH (for monitoring).
        weights_fn: (V, WH) -> elementwise weights D >= 0 (or broadcastable,
                   e.g. shape (n,) column weights). None => D == 1.
        max_iter:  max MU iterations.
        tol:       stop when (relative cost decrease over 5 iters) < tol.
        seed:      RNG seed for initialization (reproducibility).
        verbose:   print cost every 50 iterations.

    Returns:
        W (m, k), H (k, n), info dict (iterations, final cost, seconds).
    """
    rng = np.random.default_rng(seed)
    W, H = initialize(V, k, rng)
    t0 = time.perf_counter()
    prev, it = np.inf, 0
    for it in range(max_iter):
        WH = W @ H
        D = None if weights_fn is None else weights_fn(V, WH)
        WV = V if D is None else V * D
        WNV = WH if D is None else WH * D
        # W-update: W <- W * ((D.V) H^T) / ((D.WH) H^T)
        W *= (WV @ H.T) / ((WNV @ H.T) + EPS)
        WH = W @ H
        WV = V if D is None else V * D
        WNV = WH if D is None else WH * D
        # H-update: H <- H * (W^T (D.V)) / (W^T (D.WH))
        H *= (W.T @ WV) / (W.T @ WNV + EPS)

        if (it + 1) % 5 == 0 or it == max_iter - 1:
            cost = cost_fn(V, V - W @ H)
            if verbose and (it + 1) % 50 == 0:
                print(f'  iter {it + 1:4d}  cost {cost:.6f}')
            if np.isfinite(prev) and prev - cost <= tol * prev:
                break
            prev = cost
    info = {'iterations': it + 1, 'cost': float(prev),
            'seconds': time.perf_counter() - t0}
    return W, H, info
