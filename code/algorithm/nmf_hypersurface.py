"""Algorithm 3: Hypersurface Cost NMF (Hamza & Brady, IEEE TSP 2006)
-- NOT taught in this course.

    min_{W,H >= 0}  sum_ij ( sqrt(c^2 + (V_ij - (WH)_ij)^2) - c )

The hypersurface cost (also known as the pseudo-Huber loss) is a smooth
interpolation between the two losses we already have:

    phi(r) ~= r^2 / (2c)   for |r| << c   (quadratic, like Frobenius)
    phi(r) ~= |r|          for |r| >> c   (linear,    like L1)

so it behaves like least squares on the clean pixels -- keeping the fast,
well-conditioned convergence of the Frobenius objective -- and like an
absolute-deviation loss on the occluded pixels, whose influence therefore
grows only linearly instead of quadratically.  Unlike the L1 cost it is
differentiable everywhere, so its transition point is a modelling choice
rather than a numerical patch applied to a non-differentiable loss.

THE SCALE CONSTANT c IS NOT OPTIONAL HERE.  The form quoted in the
original paper fixes c = 1, which is the right choice for data whose
residuals are large compared with 1.  Our pixels are normalised to
[0, 1], so every residual satisfies |r| <= 1 and the loss never leaves
its quadratic regime: with c = 1 the weights
D = 1/sqrt(1 + r^2) lie in [1/sqrt(2), 1], i.e. they are nearly uniform,
and the algorithm silently degenerates into the Frobenius baseline.
(We observed exactly that: identical RREs to four decimal places.)
The constant must therefore be set relative to the dynamic range of the
data.  We default to c = 0.05, i.e. "residuals above 5% of the dynamic
range are treated as outliers", which puts the saturated occlusion
pixels (residual ~ 1) firmly in the linear regime while leaving ordinary
model error in the quadratic one.

MM derivation: phi is concave in u = r^2, hence
    phi(r) <= phi(r^t) + (r^2 - (r^t)^2) / (2 sqrt(c^2 + (r^t)^2)),
a weighted quadratic majoriser with elementwise weights

    D_ij = 1 / sqrt(c^2 + (V_ij - (WH^t)_ij)^2).

Note the weights are bounded above by 1/c: an arbitrarily large residual
is damped but never given exactly zero weight, which makes this cost less
aggressive -- and numerically better behaved -- than L1 reweighting.
"""
import numpy as np

from .common import weighted_mu_nmf

#: transition scale, as a fraction of the [0, 1] pixel dynamic range
DEFAULT_SCALE = 0.05


def make_cost(scale):
    def _cost(V, R):
        return float(np.sum(np.sqrt(scale * scale + R * R) - scale))
    return _cost


def make_weights(scale):
    def _weights(V, WH):
        R = V - WH
        return 1.0 / np.sqrt(scale * scale + R * R)
    return _weights


def nmf_hypersurface(V, k, max_iter=500, tol=1e-5, seed=0, verbose=False,
                     scale=DEFAULT_SCALE, record_history=False):
    """Hypersurface-cost (pseudo-Huber) NMF. Returns (W, H, info)."""
    return weighted_mu_nmf(V, k, make_cost(scale),
                           weights_fn=make_weights(scale),
                           max_iter=max_iter, tol=tol, seed=seed,
                           verbose=verbose, record_history=record_history)
