"""Shared machinery for multiplicative-update (MU) NMF algorithms.

Every algorithm in this package (except the explicit outlier model in
``nmf_l1_reg``) minimises a cost that can be *majorised* by a weighted
quadratic

    E(W, H) <= sum_ij D_ij * (V_ij - (WH)_ij)^2 + const,   W, H >= 0,

where the weight matrix ``D >= 0`` is held *fixed* within an iteration.
At every iteration ``D`` is recomputed from the current residual so that
the weighted quadratic majorises the true (possibly non-smooth) cost and
touches it at the current iterate.  This majorise-minimise (MM) scheme
yields the multiplicative updates below, whose multiplicative form keeps
``W, H >= 0`` automatically, with no projection step.

Writing all algorithms against this single kernel means the *only*
difference between them is the weight function ``D``, which is exactly
the thing the report needs to compare:

    Frobenius     D_ij = 1                          (no down-weighting)
    L1            D_ij = 1 / |r_ij|                 (per pixel)
    Hypersurface  D_ij = 1 / sqrt(1 + r_ij^2)       (per pixel, bounded)
    L2,1          D_ij = 1 / ||r_:j||_2             (per image)

References:
  - Lee & Seung, NeurIPS 1999 (Frobenius / KL NMF).
  - Lin, Neural Comput. 2007 (convergence of MU).
  - Hamza & Brady, IEEE TSP 2006 (hypersurface cost NMF).
  - Ding, Li & Jordan, NeurIPS 2008 / IEEE TPAMI 2010 (L1- and
    L2,1-norm based NMF).
  - Zhang, Liang & Zhang, ICDM 2011 (L1-regularised robust NMF).
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
                    tol=1e-5, seed=0, verbose=False, check_every=5,
                    record_history=False):
    """Weighted-Frobenius MM/MU kernel shared by the MU-family algorithms.

    Args:
        V:         (m, n) nonnegative data matrix (possibly corrupted).
        k:         number of components (rank of the factorisation).
        cost_fn:   (V, R) -> scalar objective, R = V - WH.
        weights_fn: (V, WH) -> weights D >= 0, either elementwise (m, n)
                   or broadcastable (e.g. shape (1, n) for column weights).
                   None means D == 1 (plain Frobenius).
        max_iter:  maximum number of MU iterations.
        tol:       stop when the relative cost decrease over ``check_every``
                   iterations falls below this value.
        seed:      RNG seed for the initialisation (reproducibility).
        verbose:   print the cost every 50 iterations.
        check_every: how often the (relatively expensive) cost is evaluated.
        record_history: if True, ``info['history']`` holds the list of
                   ``(iteration, cost)`` pairs actually evaluated.  Used by
                   the convergence study; off by default so that the
                   benchmark timings are not polluted.

    Returns:
        W (m, k), H (k, n), info dict with keys ``iterations``, ``cost``,
        ``seconds``, ``converged`` and optionally ``history``.
    """
    rng = np.random.default_rng(seed)
    W, H = initialize(V, k, rng)
    t0 = time.perf_counter()
    prev = np.inf
    cost = np.inf
    converged = False
    history = []
    it = 0

    for it in range(max_iter):
        WH = W @ H
        D = None if weights_fn is None else weights_fn(V, WH)
        DV = V if D is None else V * D
        # W-update: W <- W * ((D.V) H^T) / ((D.WH) H^T)
        W *= (DV @ H.T) / (((WH if D is None else WH * D) @ H.T) + EPS)
        # H-update uses the refreshed WH but the same D (standard IRLS:
        # the weights are frozen for the whole MM step).
        WH = W @ H
        H *= (W.T @ DV) / ((W.T @ (WH if D is None else WH * D)) + EPS)

        last = (it == max_iter - 1)
        if (it + 1) % check_every == 0 or last:
            cost = float(cost_fn(V, V - W @ H))
            if record_history:
                history.append((it + 1, cost))
            if verbose and (it + 1) % 50 == 0:
                print(f'  iter {it + 1:4d}  cost {cost:.6f}')
            if np.isfinite(prev) and prev - cost <= tol * prev:
                converged = True
                break
            prev = cost

    info = {
        'iterations': it + 1,
        # the *final* cost, not the previous checkpoint
        'cost': float(cost) if np.isfinite(cost) else float(cost_fn(V, V - W @ H)),
        'seconds': time.perf_counter() - t0,
        'converged': converged,
    }
    if record_history:
        info['history'] = history
    return W, H, info
