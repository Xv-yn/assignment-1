"""Evaluation metrics.

The assignment allows scikit-learn *for evaluation only*; the NMF
algorithms themselves (``nmf_*.py``, ``common.py``) import nothing beyond
numpy and the Python standard library.  Everything in this module is
strictly post-hoc measurement.

Three metrics are provided, as listed in the brief:

  * Relative Reconstruction Error (RRE) -- required.
  * Average clustering accuracy       -- optional.
  * Normalised Mutual Information     -- optional.

Note on what RRE is measured against: the brief defines
``RRE = ||V_hat - WH||_F / ||V_hat||_F`` where ``V_hat`` is the CLEAN
data and ``W, H`` are fitted on the CONTAMINATED data ``V``.  So a low
RRE means the algorithm recovered the underlying clean signal despite
being shown only the corrupted version -- it is a denoising score, not a
fitting score.  Fitting the noise well would make it worse, not better.
"""
from collections import Counter

import numpy as np

# SciPy and scikit-learn are imported lazily, inside the functions that
# need them, rather than at module scope.  RRE -- the only metric the
# brief actually requires -- depends on numpy alone, so an environment
# without scikit-learn can still import ``algorithm`` and reproduce every
# RRE number in the report.  A top-level import would make the whole
# package unimportable instead.


def _sklearn():
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import (accuracy_score,
                                     normalized_mutual_info_score)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            'The optional clustering metrics (accuracy, NMI) need '
            'scikit-learn. Install it with '
            '`pip install -r code/requirements.txt`, or run the '
            'RRE-only experiments with --skip-clustering.') from exc
    return KMeans, accuracy_score, normalized_mutual_info_score


def relative_reconstruction_error(V_hat, W, H):
    """RRE = ||V_hat - WH||_F / ||V_hat||_F (lower is better).

    ``V_hat`` must be the CLEAN matrix; ``W, H`` come from the noisy one.
    """
    denom = float(np.linalg.norm(V_hat))
    if denom == 0.0:
        raise ValueError(
            'RRE is undefined: the clean data matrix has zero Frobenius '
            'norm. Check that the dataset loaded correctly.')
    return float(np.linalg.norm(V_hat - W @ H) / denom)


def kmeans_clusters(H, Y, seed=0, n_init=10):
    """K-means on the NMF codes (columns of H). Returns cluster ids."""
    KMeans, _, _ = _sklearn()
    n_clusters = len(np.unique(Y))
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    return kmeans.fit_predict(H.T)  # columns of H = sample codes


def map_majority(y_cluster, Y):
    """Each cluster takes the majority true label inside it.

    This is the mapping used by the assignment tutorial notebook, so it is
    the one we report as "Average Accuracy".
    """
    Y_pred = np.zeros(len(Y), dtype=int)
    for c in np.unique(y_cluster):
        ind = y_cluster == c
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
    return Y_pred


def map_hungarian(y_cluster, Y):
    """Optimal one-to-one cluster/label matching (Kuhn-Munkres).

    Stricter than majority voting, which may map several clusters to the
    same class and therefore flatters a degenerate clustering.  Reported
    alongside the tutorial metric as a sanity check.
    """
    from scipy.optimize import linear_sum_assignment

    classes = np.unique(Y)
    class_index = {c: i for i, c in enumerate(classes)}
    cost = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for cluster, true in zip(y_cluster, Y):
        cost[cluster, class_index[true]] -= 1
    rows, cols = linear_sum_assignment(cost)
    mapping = {r: classes[c] for r, c in zip(rows, cols)}
    return np.array([mapping[c] for c in y_cluster])


def assign_cluster_label(H, Y, seed=0, method='majority'):
    """Backwards-compatible wrapper around the two mappings above."""
    y_cluster = kmeans_clusters(H, Y, seed=seed)
    if method == 'majority':
        return map_majority(y_cluster, Y)
    if method == 'hungarian':
        return map_hungarian(y_cluster, Y)
    raise ValueError(f'unknown mapping method: {method}')


def clustering_metrics(H, Y, seed=0):
    """Return accuracy (both mappings) and NMI from a SINGLE K-means fit.

    NMI is mapping-invariant, so it is computed from the raw clustering;
    only accuracy depends on how clusters are named.  Fitting K-means once
    and deriving all three metrics from it halves the evaluation cost of
    the full experiment grid and, more importantly, guarantees the two
    accuracies describe the same clustering.
    """
    _, accuracy_score, normalized_mutual_info_score = _sklearn()
    y_cluster = kmeans_clusters(H, Y, seed=seed)
    Y_major = map_majority(y_cluster, Y)
    Y_hung = map_hungarian(y_cluster, Y)
    return {
        'accuracy': float(accuracy_score(Y, Y_major)),
        'accuracy_hungarian': float(accuracy_score(Y, Y_hung)),
        'nmi': float(normalized_mutual_info_score(Y, y_cluster)),
    }
