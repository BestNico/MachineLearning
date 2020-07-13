""" kNN by python & numpy """
from dataclasses import dataclass
from abc import ABC
from math import sqrt
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier

import numpy as np


# 纯Python实现
def knn_classify(k: int, x_trains, y_train, target):

    assert 1 <= k <= x_trains.shape[0], "k must be valid"
    assert x_trains.shape[0] == y_train.shape[0], \
        "the size of X_train must equal to the size of y_train."
    assert x_trains.shape[1] == target.shape[0], \
        "the feature number of x must be equal to X_train."

    distances = [sqrt(np.sum((x_train - target)**2)) for x_train in x_trains]
    nearest = np.argsort(distances)

    top_k_y = [y_train[i] for i in nearest[:k]]
    votes = Counter(top_k_y)

    return votes.most_common()[0][0]


# sklearn调用
def knn_classifier(x_train: list, y_train: list, x):
    knn_clf = KNeighborsClassifier(n_neighbors=6)

    knn_clf.fit(x_train, y_train)

    x_predict_target = x.reshape(1, -1)
    return knn_clf.predict(x_predict_target)


# 类sklearn实现
@dataclass()
class KnnData(ABC):
    k: int
    _X_train = None
    _Y_train = None

    def fit(self, x_train, y_train):
        self._X_train = x_train
        self._Y_train = y_train

        return self


class KnnClassifier(KnnData):
    def predict(self, target):
        distances = [sqrt(np.sum((x_train - target) ** 2))
                     for x_train in self._X_train]
        nearest = np.argsort(distances)

        top_k_y = [self._Y_train[i] for i in nearest[:self.k]]
        votes = Counter(top_k_y)

        return votes.most_common()[0]


# run
raw_data_X = [
    [3.393533211, 2.331273381],
    [3.110073483, 1.781539638],
    [1.343808831, 3.368360954],
    [3.582294042, 4.679179110],
    [2.280362439, 2.866990263],
    [7.423436942, 4.696522875],
    [5.745051997, 3.533989803],
    [9.172168622, 2.511101045],
    [7.792783481, 3.424088941],
    [7.739820817, 0.791637231]
]

raw_data_Y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
X_train = np.array(raw_data_X)
Y_train = np.array(raw_data_Y)

x = np.array([8.093607318, 3.365731514])

knn_clf = KnnClassifier(k=6)
knn_clf.fit(X_train, Y_train)
x_predict = knn_clf.predict(x)
print(x_predict[0])
