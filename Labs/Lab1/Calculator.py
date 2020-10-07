import numpy as np


def calc_e_rms(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


def calc_loss(x_matrix, t_vec, lambda_penalty, w_vec):
    Xw_T = x_matrix @ w_vec - t_vec
    return 0.5 * np.mean(Xw_T.T @ Xw_T + lambda_penalty * w_vec @ np.transpose([w_vec]))


def calc_derivative(x_matrix, t_vec, lambda_penalty, w_vec):
    return x_matrix.T @ x_matrix @ w_vec - x_matrix.T @ t_vec + lambda_penalty * w_vec
