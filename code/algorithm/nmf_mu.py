"""Algorithm 1: classical Frobenius-norm NMF via Lee-Seung MU (taught).

    min_{W,H >= 0}  ||V - WH||_F^2

This is the non-robust baseline.  Because the residual enters squared, a
single occluded pixel with residual r contributes r^2 to the cost, so the
handful of saturated pixels inside a white block can dominate the entire
objective and drag the learned basis towards the occlusion.  Every other
algorithm in this package is a way of making that contribution grow more
slowly than quadratically.

Lee-Seung auxiliary-function preconditioning of the gradient gives:

    W <- W * (V H^T) / (W H H^T)
    H <- H * (W^T V) / (W^T W H)

Implemented as the D = 1 special case of `weighted_mu_nmf`.
"""
import numpy as np

from .common import weighted_mu_nmf


def _cost(V, R):
    return float(np.sum(R * R))


def nmf_frobenius(V, k, max_iter=500, tol=1e-5, seed=0, verbose=False,
                  record_history=False):
    """Frobenius-norm NMF. Returns (W, H, info)."""
    return weighted_mu_nmf(V, k, _cost, weights_fn=None, max_iter=max_iter,
                           tol=tol, seed=seed, verbose=verbose,
                           record_history=record_history)
