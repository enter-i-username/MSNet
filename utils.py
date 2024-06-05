import numpy as np


def rx(x: np.ndarray):
    rows, cols, bands = x.shape
    x_2d = x.reshape((rows * cols, bands))

    inv_Sig = np.linalg.inv(np.cov(x_2d.T))
    mu = np.mean(x_2d, axis=0, keepdims=True)
    get_M_dis = lambda _x: (_x - mu) @ inv_Sig @ (_x - mu).T
    dm = np.array([get_M_dis(_x) for _x in x_2d])

    dm = dm.reshape((rows, cols))

    return dm


class MinMaxNorm:

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.max = None

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
        return self

    def transform(self, x):
        x_std = (x - self.min) / (self.max - self.min)
        x_norm = x_std * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]
        return x_norm

    def inverse_transform(self, x_norm):
        x_std = (x_norm - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0])
        x = x_std * (self.max - self.min) + self.min
        return x


class ZScoreNorm:

    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, x):
        self.means = np.mean(x, axis=(0, 1))
        self.stds = np.std(x, axis=(0, 1))
        return self

    def transform(self, x):
        x_norm = np.zeros_like(x)
        for i in range(x.shape[2]):
            x_norm[:, :, i] = (x[:, :, i] - self.means[i]) / self.stds[i]
        return x_norm

    def inverse_transform(self, x_norm):
        x = np.zeros_like(x_norm)
        for i in range(x_norm.shape[2]):
            x[:, :, i] = x_norm[:, :, i] * self.stds[i] + self.means[i]
        return x
