import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    """ train_test_split """
    assert X.shape[0] == y.shape[0], \
        "the size of X must be equal to the size of y"
    assert 0.0 <= test_ratio <= 1.0, \
        "test_ratio must be valid"

    if seed:
        np.random.seed(seed)

    shuffle_indexes = np.random.permutation(len(X))

    test_size = int(len(X) * test_ratio)
    test_indexes = shuffle_indexes[:test_size]
    train_indexes = shuffle_indexes[test_size:]

    return X[train_indexes], X[test_indexes], y[train_indexes], y[test_indexes]
