"""Evaluation metrics (allowed to use scikit-learn for evaluation only)."""
from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score


def relative_reconstruction_error(V_hat, W, H):
    """RRE = ||V_hat - WH||_F / ||V_hat||_F (lower is better)."""
    return np.linalg.norm(V_hat - W @ H) / np.linalg.norm(V_hat)


def assign_cluster_label(H, Y, seed=0):
    """K-means on the rows of H (sample codes); map clusters to the
    majority true label inside each cluster (label-induction trick)."""
    kmeans = KMeans(n_clusters=len(set(Y)), n_init=10, random_state=seed)
    y_cluster = kmeans.fit_predict(H.T)  # rows of H = sample representations
    Y_pred = np.zeros(len(Y), dtype=int)
    for c in set(y_cluster):
        ind = y_cluster == c
        Y_pred[ind] = Counter(Y[ind]).most_common(1)[0][0]
    return Y_pred


def clustering_metrics(H, Y, seed=0):
    """Return (accuracy, NMI) of NMF-coefficient clustering."""
    Y_pred = assign_cluster_label(H, Y, seed=seed)
    return accuracy_score(Y, Y_pred), normalized_mutual_info_score(Y, Y_pred)
