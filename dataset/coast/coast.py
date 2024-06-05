import scipy.io as sio
import os


name = 'coast'
path = os.path.dirname(__file__)
file_name = os.path.join(path, name + '.mat')


def get_data():
    mat = sio.loadmat(file_name)
    data = mat['data'].astype(float)
    gt = mat['map'].astype(bool)

    return data, gt
