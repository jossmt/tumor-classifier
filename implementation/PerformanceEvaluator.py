# • Mean squared error (very easy, but may be naïve)
# • R-squared coefficient (easy and relatively robust)
# • F-test (complicated but accurate) - better for regression tasks
from sklearn.metrics import mean_squared_error
import numpy as np


def mean_squared_error_calc(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)


def r_squared_coefficient(x, y):
    correlation_matrix = np.corrcoef(x, y)
    correlation_xy = correlation_matrix[0, 1]
    return correlation_xy ** 2
