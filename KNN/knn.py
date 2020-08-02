""" kNN by python & numpy """
import numpy as np
from dataclasses import dataclass
from abc import ABC
from math import sqrt
from collections import Counter

from .metrics import accuracy_score


@dataclass()
class KNNAbstract(ABC):

    k: int
    _X_train = None
    _y_train = None

    def fit(self, X_train, y_train):
        self._X_train = X_train
        self._y_train = y_train

        return self

    def __repr__(self):
        return "KNN(k=%d)" % self.k


class KNNClassifier(KNNAbstract):
    def predict(self, X_predict):
        y_predict = [self._predict(x) for x in X_predict]
        return np.array(y_predict)

    def _predict(self, x):
        distances = [sqrt(np.sum((x_train - x) ** 2)) for x_train in self._X_train]

        nearest = np.argsort(distances)

        topK_y = [self._y_train[i] for i in nearest[:self.k]]
        votes = Counter(topK_y)

        return votes.most_common(1)[0][0]

    def score(self, X_test, y_test):
        """ score """
        y_predict = self.predict(X_test)
        return accuracy_score(y_test, y_predict)
