"""Shared utilities for neural encoding analyses."""

import numpy as np
from sklearn.linear_model import RidgeCV


RIDGE_ALPHAS = np.logspace(-2, 6, 20)


def mean_neural_r2(features, neural_activity):
    """
    Ridge regression R² per neuron, averaged over neurons.

    Parameters
    ----------
    features : ndarray [n_scenes × D]
    neural_activity : ndarray [n_scenes × N_NEURONS]

    Returns
    -------
    mean_r2 : float
    per_neuron_r2 : ndarray [N_NEURONS]
    """
    n_neurons = neural_activity.shape[1]
    r2s = np.zeros(n_neurons)
    for j in range(n_neurons):
        ridge = RidgeCV(alphas=RIDGE_ALPHAS)
        ridge.fit(features, neural_activity[:, j])
        r2s[j] = ridge.score(features, neural_activity[:, j])
    return float(r2s.mean()), r2s
