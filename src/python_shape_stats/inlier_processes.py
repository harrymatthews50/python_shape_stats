import numpy as np
from python_shape_stats.helpers import validate_landmark_configuration_and_weights


def update_inlier_weights_gaussian(p: np.ndarray, q: np.ndarray, w: np.ndarray = None, kappa: float = 3) -> np.ndarray:
    """
    Updates inlier weights (e.g. as part of an EM algorithm) between two landmark configurations to flag regions that are unusually diferent between them

    :param p: an n (landmarks) by 3 array of landmark coordinates
    :param q: an n (landmarks) by 3 array of landmark coordinates
    :param w: previously calculated inlier weights (e.g. on the previous iteration of the EM algorithm)
    :param kappa: a threshold for determining inliers/outliers in units of standrad deviation
    :return: inlier weights for each vertex in p and q
    """
    w = validate_landmark_configuration_and_weights(p, q=q, w=w)
    d = np.linalg.norm(p - q, axis=1)
    sigma = _weighted_standard_deviation(d, w)
    return _compute_inlier_weights_gaussian(d, sigma, kappa=kappa)


def _compute_inlier_weights_gaussian(d, sigma, kappa=3):
    normpdf = lambda z: np.exp((-z ** 2) * 0.5) / np.sqrt(2 * np.pi)
    z = d / sigma
    l = normpdf(kappa)
    inlier_prob = normpdf(z)
    return inlier_prob / (inlier_prob + l)


def _weighted_standard_deviation(d, w):
    return np.sqrt(np.sum((d ** 2) * w) / np.sum(w))
