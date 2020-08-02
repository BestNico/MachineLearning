import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


X = None    # numpy.array
y = None    # numpy.array

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


def knn_classifier(x_train: list, y_train: list, x):
    knn_clf = KNeighborsClassifier(n_neighbors=6)

    knn_clf.fit(x_train, y_train)

    x_predict_target = x.reshape(1, -1)
    return knn_clf.predict(x_predict_target)