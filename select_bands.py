import numpy as np


def OPBS(
        x: np.ndarray,
        num_bs: int
):
    """
    Ref:
    W. Zhang, X. Li, Y. Dou, and L. Zhao, “A geometry-based band
    selection approach for hyperspectral image analysis,” IEEE Transactions
    on Geoscience and Remote Sensing, vol. 56, no. 8, pp. 4318–4333, 2018.
    """
    rows, cols, bands = x.shape
    eps = 1e-9

    x_2d = np.reshape(x, (rows * cols, bands))
    y_2d = x_2d.copy()
    h = np.zeros(bands)
    band_idx = []

    idx = np.argmax(np.var(x_2d, axis=0))
    band_idx.append(idx)
    h[idx] = np.sum(x_2d[:, band_idx[-1]] ** 2)

    i = 1
    while i < num_bs:
        id_i_1 = band_idx[i - 1]

        _elem, _idx = -np.inf, 0
        for t in range(bands):
            if t not in band_idx:
                y_2d[:, t] = y_2d[:, t] - y_2d[:, id_i_1] * (np.dot(y_2d[:, id_i_1], y_2d[:, t]) / (h[id_i_1] + eps))
                h[t] = np.dot(y_2d[:, t], y_2d[:, t])

                if h[t] > _elem:
                    _elem = h[t]
                    _idx = t

        band_idx.append(_idx)
        i += 1

    band_idx = sorted(band_idx)
    return band_idx
