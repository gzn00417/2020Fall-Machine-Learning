import numpy as np


def get_params(x_matrix, t_vec) -> [float]:
    return np.linalg.pinv(x_matrix.T @ x_matrix) @ x_matrix.T @ t_vec


def get_params_with_penalty(x_matrix, t_vec, lambda_penalty):
    return (
        np.linalg.pinv(
            x_matrix.T @ x_matrix + lambda_penalty * np.identity(x_matrix.shape[1])
        )
        @ x_matrix.T
        @ t_vec
    )
