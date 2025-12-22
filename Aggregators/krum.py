import numpy as np

from .base import _BaseAggregator


def _compute_scores(distances, i, n, f):
    """Compute scores for node i.
    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        i {int} -- index of worker, starting from 0.
        n {int} -- total number of workers
        f {int} -- Total number of Byzantine workers.
    Returns:
        float -- krum distance score of i.
    """
    s = [distances[j][i] ** 2 for j in range(i)] + [
        distances[i][j] ** 2 for j in range(i + 1, n)
    ]
    _s = sorted(s)[: n - f - 2]
    return sum(_s)


def krum(distances, n, f):
    """Krum algorithm
    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
    Returns:
        int -- Index of the selected worker.
    """
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            if distances[i][j] < 0:
                raise ValueError(
                    "The distance between node {} and {} should be non-negative: Got {}.".format(
                        i, j, distances[i][j]
                    )
                )

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return sorted_scores[0][0]

def multi_krum(distances, n, f, m):
    """Multi_Krum algorithm
    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist. i, j starts with 0.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.
    Returns:
        list -- A list indices of worker indices for aggregation. length <= m
    """
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if m < 1 or m > n:
        raise ValueError(
            "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                m, n
            )
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            if distances[i][j] < 0:
                raise ValueError(
                    "The distance between node {} and {} should be non-negative: Got {}.".format(
                        i, j, distances[i][j]
                    )
                )

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return list(map(lambda x: x[0], sorted_scores))[:m]


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def pairwise_euclidean_distances(vectors):
    """Compute the pairwise euclidean distance.
    Arguments:
        vectors {list} -- A list of vectors.
    Returns:
        dict -- A dict of dict of distances {i:{j:distance}}
    """
    n = len(vectors)

    distances = {}
    for i in range(n - 1):
        distances[i] = {}
        for j in range(i + 1, n):
            distances[i][j] = _compute_euclidean_distance(vectors[i], vectors[j]) ** 2
    return distances


class Krum(_BaseAggregator):
    r"""
    This script implements KRUM and Multi-KRUM algorithms.
    Blanchard, Peva, Rachid Guerraoui, and Julien Stainer.
    "Machine learning with adversaries: Byzantine tolerant gradient descent."
    Advances in Neural Information Processing Systems. 2017.
    """

    def __init__(self, n, f, m=None,mk=True):
        self.n = n
        self.f = f
        self.m = m
        self.mk = mk ## Multi-Krum
        self.success = 1
        self.impact_ratio = 1
        super(Krum, self).__init__()

    def __call__(self, inputs):
        distances = pairwise_euclidean_distances(inputs)
        if not self.mk:
            selected_index = krum(distances, self.n, self.f)
            top_m_indices = [selected_index]
            #print(selected_index)
        else:
            top_m_indices = multi_krum(distances, self.n, self.f, self.m)

        byzantine_clients = np.arange(0, self.n)[-self.f:]
        bypassed = 0
        for cl in byzantine_clients:
            if cl in top_m_indices:
                bypassed += 1
        if self.f > 0:
            self.success = bypassed / self.f
            self.impact_ratio = bypassed / len(top_m_indices)
        else:
            self.success = 0
            self.impact_ratio = 0
        values = sum(inputs[i] for i in top_m_indices) / len(top_m_indices)
        return values

    def get_attack_stats(self) ->dict:
        return {'Krum-Bypassed':self.success,
                'Krum-Impact': self.impact_ratio
                }
