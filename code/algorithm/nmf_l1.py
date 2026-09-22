"""Algorithm 2: L1-norm based NMF (Ding, Li & Jordan, NeurIPS 2008) --
NOT taught in this course.

    min_{W,H >= 0}  ||V - WH||_1 = sum_ij |V_ij - (WH)_ij|

The elementwise L1 norm makes the influence of a residual grow only
linearly rather than quadratically, so once a pixel is "far enough" from
the model its magnitude stops mattering.  This is what makes the
factorisation robust to salt-and-pepper / occlusion corruption, which is
sparse in the pixel domain.

MM derivation: majorise each |r| by its tangent parabola
    |r| <= r^2 / (2 s) + s / 2      (s = |r^t| from the previous iterate),
giving a weighted-Frobenius surrogate with elementwise weights
    D_ij = 1 / |V_ij - (WH^t)_ij|.
MU updates (per iteration, reweighting -> IRLS-style):

    W <- W * ((D.V) H^T) / ((D.WH) H^T)
    H <- H * (W^T (D.V)) / (W^T (D.WH))

Numerical note (`delta`): the weight 1/|r| diverges as a residual goes to
zero, which is exactly what happens on the clean, well-fitted pixels that
make up most of the image.  The usual remedy is Huber-style smoothing,
D = 1 / (|r| + delta), which is the IRLS weight of the smoothed loss
sqrt(r^2 + delta^2) rather than of |r| itself.  `delta` therefore is not
a free knob: it sets the residual scale below which the loss is treated
as quadratic.  Since pixels live in [0, 1] here, delta = 1e-3 means
"residuals under one tenth of one percent of the dynamic range count as
noise-free".  `experiments/run_sensitivity.py` sweeps it so that the
report can show the conclusions do not hinge on this choice.
"""
import numpy as np

from .common import weighted_mu_nmf

DEFAULT_DELTA = 1e-3


def _cost(V, R):
    return float(np.abs(R).sum())


def make_weights(delta):
    def _weights(V, WH):
        return 1.0 / (np.abs(V - WH) + delta)
    return _weights


def nmf_l1(V, k, max_iter=500, tol=1e-5, seed=0, verbose=False,
           delta=DEFAULT_DELTA, record_history=False):
    """L1-norm NMF. Returns (W, H, info)."""
    return weighted_mu_nmf(V, k, _cost, weights_fn=make_weights(delta),
                           max_iter=max_iter, tol=tol, seed=seed,
                           verbose=verbose, record_history=record_history)
