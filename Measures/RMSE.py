""" Mean squared error regression loss """

from sklearn.metrics import mean_squared_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

rmse = mean_squared_error(y_true, y_pred)

print(rmse)

y_true_second = [[0.5, 1], [-1, 1], [7, -6]]
y_pred_second = [[0, 2], [-1, 2], [8, -5]]

rmse_second = mean_squared_error(y_true_second, y_pred_second)

print(rmse_second)