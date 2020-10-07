import numpy as np
from Calculator import *


def gradient_descent_fit(
    x_matrix, t_vec, lambda_penalty, w_vec_0, learning_rate=0.1, delta=1e-6
):
    loss_0 = calc_loss(x_matrix, t_vec, lambda_penalty, w_vec_0)
    k = 0
    w = w_vec_0
    while True:
        w_ = w - learning_rate * calc_derivative(x_matrix, t_vec, lambda_penalty, w)
        loss = calc_loss(x_matrix, t_vec, lambda_penalty, w_)
        if np.abs(loss - loss_0) < delta:
            break
        else:
            k += 1
            if loss > loss_0:
                learning_rate *= 0.5
            loss_0 = loss
            w = w_
    return k, w
