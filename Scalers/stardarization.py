import numpy as np
from abc import ABC
from dataclasses import dataclass
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


@dataclass()
class StandardAbstract(ABC):
    mean_ = None
    scale_ = None


# by self
class StandardScaler(StandardAbstract):

    def fit(self, X):
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        if self.mean_ is None or self.scale_ is None:
            print("Please run fit function first.")
            return

        if X.shape[1] != len(self.mean_):
            print("The feature number of X must equal to mean_ and std_")
            return

        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / self.scale_[col]

        return resX


# run test
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
standardScaler = StandardScaler()
standardScaler.fit(X_train)

X_train = standardScaler.transform(X_train)
X_test_standard = standardScaler.transform(X_test)

knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test_standard, y_test))
