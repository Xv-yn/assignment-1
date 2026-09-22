"""Self-checks for the NMF implementations.

Run directly -- no pytest required:

    cd code && python test_algorithms.py

The checks are the ones that would actually catch a wrong implementation,
rather than the ones that are easy to write:

  1. Non-negativity is preserved.  The whole point of the multiplicative
     form is that W, H >= 0 needs no projection; if a sign ever appears,
     an update has a subtraction in it.
  2. The objective is monotonically non-increasing.  Every algorithm here
     is derived as a majorise-minimise scheme, and MM guarantees descent.
     A violation means the majoriser does not actually dominate the cost
     (a wrong weight formula is the usual cause), so this is the single
     most informative test in the file.
  3. Near-exact recovery of a synthetic rank-k non-negative matrix.  If an
     algorithm cannot fit noiseless data that is exactly rank k, nothing
     it reports on real data means anything.
  4. Robustness actually happens: on synthetic data with a few grossly
     corrupted entries, the robust methods must beat the Frobenius
     baseline.  This is the property the whole report is about, so it is
     worth asserting rather than assuming.
  5. The noise generator does what it claims: the right number of images
     touched, the right fill value, and nothing changed outside the
     blocks.

Exit status is 0 if everything passes, 1 otherwise, so this can be wired
into CI or a pre-submission check.
"""
import sys

import numpy as np

from algorithm import ALGORITHMS
from algorithm.evaluate import relative_reconstruction_error
from algorithm.noise import occlusion_noise

PASS, FAIL = 0, 0


def check(name, condition, detail=''):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f'  PASS  {name}')
    else:
        FAIL += 1
        print(f'  FAIL  {name}  {detail}')


def synthetic(m=60, n=80, k=5, seed=0):
    """An exactly rank-k non-negative matrix."""
    rng = np.random.default_rng(seed)
    W = rng.uniform(0.1, 1.0, size=(m, k))
    H = rng.uniform(0.1, 1.0, size=(k, n))
    return W @ H, k


# ---------------------------------------------------------------- tests

def test_nonnegativity_and_shapes():
    print('\n[1] non-negativity and output shapes')
    V, k = synthetic()
    for name, fn in ALGORITHMS.items():
        W, H, info = fn(V, k=k, max_iter=40, tol=0.0, seed=0)
        check(f'{name}: shapes', W.shape == (V.shape[0], k)
              and H.shape == (k, V.shape[1]))
        check(f'{name}: W, H >= 0', W.min() >= 0 and H.min() >= 0,
              f'min(W)={W.min():.3g} min(H)={H.min():.3g}')
        check(f'{name}: finite', np.isfinite(W).all() and np.isfinite(H).all())


def test_monotone_descent():
    """Every objective must be non-increasing along its own iterates."""
    print('\n[2] monotone descent of each objective')
    V, k = synthetic()
    for name, fn in ALGORITHMS.items():
        _W, _H, info = fn(V, k=k, max_iter=60, tol=0.0, seed=1,
                          record_history=True)
        hist = [c for _it, c in info['history']]
        # allow a tiny tolerance for floating-point noise near convergence
        worst = max((hist[i + 1] - hist[i]) / max(abs(hist[i]), 1e-12)
                    for i in range(len(hist) - 1))
        check(f'{name}: cost non-increasing', worst <= 1e-6,
              f'largest relative increase = {worst:.2e}')


def test_exact_recovery():
    print('\n[3] recovery of an exactly rank-k matrix')
    V, k = synthetic()
    for name, fn in ALGORITHMS.items():
        W, H, _ = fn(V, k=k, max_iter=800, tol=1e-12, seed=2)
        err = relative_reconstruction_error(V, W, H)
        # Two algorithms are held to a looser bar, for reasons that are
        # properties of the methods rather than of the implementation:
        #   L1-Reg    deliberately shrinks part of the residual into S;
        #   L1-NMF    has an IRLS weight 1/(|r| + delta) whose smoothing
        #             constant puts a floor on the residual it will chase,
        #             and its linear loss gives a much weaker gradient than
        #             a quadratic one once the residual is already small,
        #             so it approaches an exact fit slowly.
        bound = {'L1-Reg Robust': 0.10, 'L1-NMF': 0.08}.get(name, 0.05)
        check(f'{name}: RRE < {bound}', err < bound, f'RRE={err:.4f}')


def test_robustness_beats_baseline():
    print('\n[4] robust losses beat the baseline on corrupted data')
    V, k = synthetic(m=60, n=80, k=4, seed=3)
    V = V / V.max()                       # put it on the [0, 1] pixel scale
    rng = np.random.default_rng(4)
    V_noisy = V.copy()
    # gross, sparse corruption: 3% of entries saturated
    mask = rng.random(V.shape) < 0.03
    V_noisy[mask] = 1.0

    errs = {}
    for name, fn in ALGORITHMS.items():
        W, H, _ = fn(V_noisy, k=k, max_iter=400, tol=1e-10, seed=5)
        errs[name] = relative_reconstruction_error(V, W, H)
        print(f'        {name:15s} RRE={errs[name]:.4f}')
    baseline = errs['F-norm MU']
    for name in ('L1-NMF', 'Hypersurface', 'L1-Reg Robust'):
        check(f'{name} < F-norm MU', errs[name] < baseline,
              f'{errs[name]:.4f} vs {baseline:.4f}')


def test_noise_generator():
    print('\n[5] occlusion noise generator')
    rng = np.random.default_rng(6)
    height, width, n = 20, 16, 50
    V = np.full((height * width, n), 0.3)

    V_noisy, corrupted = occlusion_noise(V, height, width, block=5,
                                         n_blocks=1, rng=rng, fraction=0.4,
                                         return_mask=True)
    check('fraction honoured', corrupted.sum() == 20,
          f'{corrupted.sum()} images corrupted, expected 20')
    check('clean columns untouched',
          np.array_equal(V_noisy[:, ~corrupted], V[:, ~corrupted]))
    check('fill value is the saturated one',
          set(np.unique(V_noisy)) <= {0.3, 1.0})

    changed = (V_noisy[:, corrupted] != V[:, corrupted]).sum(axis=0)
    check('each corrupted image loses exactly b^2 pixels',
          np.all(changed == 25), f'got {np.unique(changed)}')

    same = occlusion_noise(V, height, width, block=0, n_blocks=1, rng=rng)
    check('block=0 is the clean control', np.array_equal(same, V))

    # more blocks must never corrupt fewer pixels
    counts = []
    for n_blocks in (1, 2, 3):
        out = occlusion_noise(V, height, width, block=4, n_blocks=n_blocks,
                              rng=np.random.default_rng(7))
        counts.append((out != V).sum())
    check('more blocks -> at least as much damage',
          counts[0] <= counts[1] <= counts[2], str(counts))


def test_determinism():
    print('\n[6] determinism under a fixed seed')
    V, k = synthetic()
    for name, fn in ALGORITHMS.items():
        W1, H1, _ = fn(V, k=k, max_iter=30, tol=0.0, seed=11)
        W2, H2, _ = fn(V, k=k, max_iter=30, tol=0.0, seed=11)
        check(f'{name}: identical across runs',
              np.allclose(W1, W2) and np.allclose(H1, H2))


def main():
    print('Self-checks for the NMF implementations')
    test_nonnegativity_and_shapes()
    test_monotone_descent()
    test_exact_recovery()
    test_robustness_beats_baseline()
    test_noise_generator()
    test_determinism()
    print(f'\n{PASS} passed, {FAIL} failed')
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
