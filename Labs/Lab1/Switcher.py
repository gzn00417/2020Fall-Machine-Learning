import numpy as np


def get_x_series(x, order) -> [float]:
    """return 1, x^1, x^2,..., x^n, n = order
    """
    series = [1.0]
    for _ in range(order):
        series.append(series[-1] * x)
    return series


def get_x_matrix(x_vec, order: int = 1) -> [[float]]:
    x_matrix = []
    for i in range(len(x_vec)):
        x_matrix.append(get_x_series(x_vec[i], order))
    return np.asarray(x_matrix)


def switch_deri_func_for_conjugate_gradient(x_matrix, t_vec, lambda_penalty, w_vec):
    A = x_matrix.T @ x_matrix - lambda_penalty * np.identity(len(x_matrix.T))
    x = w_vec
    b = x_matrix.T @ t_vec
    return A, x, b

