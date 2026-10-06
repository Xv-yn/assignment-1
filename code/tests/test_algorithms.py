"""Validation tests for Phase 1: correctness of MU kernel, robustness ordering,
noise integrity, and YaleB loading."""
import sys
import time

import numpy as np

import os
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(HERE)
sys.path.insert(0, CODE_DIR)
DATA_ROOT = os.path.join(os.path.dirname(CODE_DIR), 'data', 'data')
from algorithm import ALGORITHMS
from algorithm.common import initialize
from algorithm.nmf_mu import nmf_frobenius
from algorithm.nmf_l1 import nmf_l1
from algorithm.nmf_l21 import nmf_l21
from algorithm.noise import occlusion_noise

rng = np.random.default_rng(0)
ok = True

def check(name, cond):
    global ok
    print(f'  [{"PASS" if cond else "FAIL"}] {name}')
    ok = ok and cond

# 1. clean low-rank recovery: F-norm should reach tiny RRE
m, n, k = 60, 120, 5
W0 = np.abs(rng.normal(size=(m, k)))
H0 = np.abs(rng.normal(size=(k, n)))
V = W0 @ H0
W, H, info = nmf_frobenius(V, k, max_iter=2000, seed=1)
rre = np.linalg.norm(V - W @ H) / np.linalg.norm(V)
check(f'clean recovery RRE={rre:.4f} < 0.05', rre < 0.05)
check('W,H nonneg', (W >= 0).all() and (H >= 0).all())

# 2. cost monotonic decrease (L1 kernel, reweighted surrogate tracks true cost)
from algorithm.nmf_l1 import _cost
W, H = initialize(V, k, np.random.default_rng(2))
costs = []
for _ in range(50):
    costs.append(_cost(V, V - W @ H))
    WH = W @ H
    D = 1.0 / (np.abs(V - WH) + 1e-3)
    W *= ((V * D) @ H.T) / ((WH * D) @ H.T + 1e-10)
    WH = W @ H
    D = 1.0 / (np.abs(V - WH) + 1e-3)
    H *= (W.T @ (V * D)) / (W.T @ (WH * D) + 1e-10)
diffs = np.diff(costs)
check('L1 cost monotonically decreases', (diffs <= 1e-6).all())

# 3. robustness ordering on spike outliers (high-magnitude sparse noise)
Vn = V.copy()
spike = rng.random(V.shape) < 0.05
Vn[spike] = 3 * Vn[spike] + 2.0
res = {}
for name, algo in ALGORITHMS.items():
    W, H, info = algo(Vn, k, max_iter=1500, seed=3)
    res[name] = np.linalg.norm(V - W @ H) / np.linalg.norm(V)
print(f'    spike RRE: {res}')
check('L1 beats F-norm on spike outliers', res['L1-NMF'] < res['F-norm MU'])
check('L2,1 beats F-norm on spike outliers', res['L2,1-NMF'] < res['F-norm MU'])

# 4. occlusion noise integrity
Vimg = np.full((30 * 37, 4), 0.5)
noisy = occlusion_noise(Vimg, 30, 37, 10, 2, np.random.default_rng(0))
check('noise fill == 1.0', noisy.max() == 1.0 and (noisy >= Vimg).all())
nz = [(noisy[:, j] != 0.5).sum() for j in range(4)]
check(f'~2*10x10 pixels corrupted per image (overlap ok): {nz}',
      all(90 <= c <= 200 for c in nz))
check('original untouched', Vimg.min() == 0.5)

# 5. clustering metrics shape
from algorithm.evaluate import clustering_metrics
W, H, _ = nmf_frobenius(V, k, max_iter=500, seed=0)
acc, nmi = clustering_metrics(H, rng.integers(0, 5, size=n), seed=0)
check(f'metrics finite: acc={acc:.3f} nmi={nmi:.3f}', 0 <= acc <= 1 and 0 <= nmi <= 1)

# 6. YaleB loads cleanly: dense labels, yaleB39 ambient subject excluded
t0 = time.perf_counter()
from algorithm.data_io import load_yaleb
Vy, Yy = load_yaleb(DATA_ROOT, reduce=4)
dt = time.perf_counter() - t0
labels = np.unique(Yy)
dense = (labels == np.arange(labels.size)).all()
check(f'YaleB {Vy.shape} subjects={labels.size} dense-labels={dense} in {dt:.1f}s',
      Vy.shape[0] == 42 * 48 and labels.size == 37 and dense
      and Vy.shape[1] == 2350 and Vy.min() >= 0)
np.save('/home/elwood/Documents/machine-learning/assignment-1/code/results/cache/yaleb_V.npy', Vy)
np.save('/home/elwood/Documents/machine-learning/assignment-1/code/results/cache/yaleb_Y.npy', Yy)
print('    re-cached yaleb_V.npy for Phase 2')

# 7. ORL loads cleanly: 40 subjects x 10
from algorithm.data_io import load_orl
Vo, Yo = load_orl(DATA_ROOT, reduce=3)
check(f'ORL {Vo.shape} subjects={np.unique(Yo).size}',
      Vo.shape == (30 * 37, 400) and np.unique(Yo).size == 40
      and (np.unique(Yo) == np.arange(40)).all())
np.save('/home/elwood/Documents/machine-learning/assignment-1/code/results/cache/orl_V.npy', Vo)
np.save('/home/elwood/Documents/machine-learning/assignment-1/code/results/cache/orl_Y.npy', Yo)
print('    re-cached orl_V.npy for Phase 2')

print('\nALL PASS' if ok else '\nSOME FAILED')
sys.exit(0 if ok else 1)
