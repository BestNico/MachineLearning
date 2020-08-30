import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.preprocessing import MinMaxScaler


# by self
class MinMaxScaler:

    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, X):
        self.min = np.array([np.min(X[:, i]) for i in range(X.shape[1])])
        self.max = np.array([np.max(X[:, i]) for i in range(X.shape[1])])

        return self

    def transform(self, X):
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.min[col]) / (self.max[col] - self.min[col])

        return resX


# iris
iris = datasets.load_iris()
X, y = iris.data, iris.target
print(X.shape[1])
print(X[:, 1])
knn_clf = KNeighborsClassifier(n_neighbors=3)
minmaxScaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)
minmaxScaler.fit(X_train)
X_train = minmaxScaler.transform(X_train)
X_test_scaler = minmaxScaler.transform(X_test)

knn_clf.fit(X_train, y_train)
print(knn_clf.score(X_test_scaler, y_test))


