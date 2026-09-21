"""Evaluation metrics (allowed to use scikit-learn for evaluation only)."""
from collections import Counter

import numpy as np


def relative_reconstruction_error(V_hat, W, H):
    """RRE = ||V_hat - WH||_F / ||V_hat||_F (lower is better)."""
    denom = np.linalg.norm(V_hat)
    if denom == 0:
        raise ValueError("Cannot compute RRE because the clean data norm is zero.")
    return np.linalg.norm(V_hat - W @ H) / denom


def _load_sklearn_metrics():
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import accuracy_score, normalized_mutual_info_score
    except ImportError as exc:
        raise ImportError(
            "Clustering metrics require scikit-learn. Install dependencies with "
            "`pip install -r requirements.txt`, or run RRE-only experiments."
        ) from exc
    return KMeans, accuracy_score, normalized_mutual_info_score


def assign_cluster_label(H, Y, seed=0):
    """K-means on the rows of H (sample codes); map clusters to the
    majority true label inside each cluster (label-induction trick)."""
    KMeans, _, _ = _load_sklearn_metrics()
    kmeans = KMeans(n_clusters=len(set(Y)), n_init=10, random_state=seed)
    y_cluster = kmeans.fit_predict(H.T)  # rows of H = sample representations
    Y_pred = np.zeros(len(Y), dtype=int)
    for c in set(y_cluster):
        ind = y_cluster == c
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
    return Y_pred


def clustering_metrics(H, Y, seed=0):
    """Return (accuracy, NMI) of NMF-coefficient clustering."""
    _, accuracy_score, normalized_mutual_info_score = _load_sklearn_metrics()
    Y_pred = assign_cluster_label(H, Y, seed=seed)
    return accuracy_score(Y, Y_pred), normalized_mutual_info_score(Y, Y_pred)
