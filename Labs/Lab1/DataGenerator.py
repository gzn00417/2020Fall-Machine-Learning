import numpy as np
import pandas as pd


def get_data(
    x_range: (float, float) = (0, 1),
    sample_num: int = 10,
    base_func=lambda x: np.sin(2 * np.pi * x),
    noise_scale=0.25,
) -> "pd.DataFrame":
    X = np.linspace(x_range[0], x_range[1], num=sample_num)
    Y = base_func(X) + np.random.normal(scale=noise_scale, size=X.shape)
    data = pd.DataFrame(data=np.dstack((X, Y))[0], columns=["X", "Y"])
    return data
