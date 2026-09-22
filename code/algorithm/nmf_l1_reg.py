"""Algorithm 5: L1-norm Regularised Robust NMF (Zhang, Liang & Zhang,
ICDM 2011; the "RobustNMF" / sparse-outlier model) -- NOT taught in this
course.

    min_{W,H >= 0, S}  1/2 ||V - WH - S||_F^2 + lambda ||S||_1

Unlike the other four algorithms, this one does not merely *down-weight*
corrupted entries: it gives the corruption its own variable.  ``S`` is an
explicit sparse outlier matrix that absorbs the occlusion, so that ``WH``
is fitted to the *cleaned* residual ``V - S``.  This is the natural model
for the noise used in this assignment: a white square pasted onto a face
is exactly a sparse, large-magnitude, spatially contiguous perturbation,
i.e. it is sparse in the pixel domain but definitely not small.

Optimisation: block-coordinate descent, alternating

  1. S-step (closed form).  With W, H fixed the problem separates over
     entries and is the classic lasso/proximal problem
         min_S 1/2 (R_ij - S_ij)^2 + lambda |S_ij|,   R = V - WH,
     solved by soft-thresholding
         S = sign(R) * max(|R| - lambda, 0).

  2. (W, H)-step.  With S fixed the problem is a plain Frobenius NMF on
     the target T = V - S, so the Lee-Seung multiplicative updates apply:
         W <- W * (T_+ H^T) / (W H H^T)
         H <- H * (W^T T_+) / (W^T W H)
     ``T`` can have negative entries once S has removed an outlier, so we
     clip it at zero (``T_+``) before the multiplicative update; this
     keeps the numerator nonnegative and hence W, H >= 0, and it is the
     standard practical fix for this model.

Each block step does not increase the objective, so the overall
objective is monotonically non-increasing and bounded below by 0.

Choosing lambda: the threshold is the residual magnitude above which a
pixel is declared an outlier rather than model error.  We default to
``lambda = 1 / sqrt(max(m, n))``, the standard Robust-PCA scaling
(Candes et al., 2011), which needs no tuning per dataset; it can be
overridden from the command line for a sensitivity study.
"""
import time

import numpy as np

from .common import EPS, initialize


def soft_threshold(R, lam):
    """Elementwise soft-thresholding: prox operator of lam * ||.||_1."""
    return np.sign(R) * np.maximum(np.abs(R) - lam, 0.0)


def _cost(V, W, H, S, lam):
    R = V - W @ H - S
    return 0.5 * float(np.sum(R * R)) + lam * float(np.abs(S).sum())


def nmf_l1_reg(V, k, max_iter=500, tol=1e-5, seed=0, verbose=False,
               lam=None, record_history=False, check_every=5):
    """L1-regularised robust NMF with an explicit sparse outlier matrix.

    Args:
        lam: L1 weight on the outlier matrix S.  ``None`` uses the
             Robust-PCA default 1 / sqrt(max(m, n)).

    Returns:
        W, H, info.  ``info['outlier_fraction']`` is the share of entries
        that ended up flagged as outliers (S != 0), which the report uses
        to check that the model really did localise the occlusion.
    """
    m, n = V.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))

    rng = np.random.default_rng(seed)
    W, H = initialize(V, k, rng)
    S = np.zeros_like(V)

    t0 = time.perf_counter()
    prev = np.inf
    cost = np.inf
    converged = False
    history = []
    it = 0

    for it in range(max_iter):
        # 1. S-step: soft-threshold the current residual.
        S = soft_threshold(V - W @ H, lam)

        # 2. (W, H)-step: Frobenius MU on the cleaned target.
        T = np.maximum(V - S, 0.0)
        W *= (T @ H.T) / ((W @ (H @ H.T)) + EPS)
        H *= (W.T @ T) / (((W.T @ W) @ H) + EPS)

        last = (it == max_iter - 1)
        if (it + 1) % check_every == 0 or last:
            cost = _cost(V, W, H, S, lam)
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
        'cost': float(cost) if np.isfinite(cost) else _cost(V, W, H, S, lam),
        'seconds': time.perf_counter() - t0,
        'converged': converged,
        'lam': float(lam),
        'outlier_fraction': float(np.mean(S != 0.0)),
    }
    if record_history:
        info['history'] = history
    return W, H, info
