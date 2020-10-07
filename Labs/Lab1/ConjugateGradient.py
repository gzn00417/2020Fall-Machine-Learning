import numpy as np


def conjugate_gradient_fit(A, x_0, b, delta=1e-6):
    """解Ax=b
    """
    x = x_0
    r_0 = b - A @ x
    p = r_0
    k = 0
    while True:
        alpha = (r_0.T @ r_0) / (p.T @ A @ p)
        x = x + alpha * p
        r = r_0 - alpha * A @ p
        if r_0.T @ r_0 < delta:
            break
        beta = (r.T @ r) / (r_0.T @ r_0)
        p = r + beta * p
        r_0 = r
        k += 1
    return k, x

