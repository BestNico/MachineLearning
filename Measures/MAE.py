""" Mean absolute error regression loss """

from sklearn.metrics import mean_absolute_error

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

mae = mean_absolute_error(y_true, y_pred)
print(mae)

y_true_second = [[0.5, 1], [-1, 1], [7, -6]]
y_pred_second = [[0, 2], [-1, 2], [8, -5]]

mae_second = mean_absolute_error(y_true_second, y_pred_second)
print(mae_second)
