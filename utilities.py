import numpy as np


def rgb2gray(array, method="standard"):
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    if method == "standard":
        return (r * 0.299 + g * 0.587 + b * 0.114)
    if method == "average":
        return (r / 3 + g / 3 + b / 3)


def bilinear_interpolate(array, i, j, axes=(-2, -1)):
    size = array.shape[axes[0]], array.shape[axes[1]]

    i0 = np.clip(np.floor(i).astype(int), 0, size[0] - 1)
    j0 = np.clip(np.floor(j).astype(int), 0, size[1] - 1)
    i1 = np.clip(i0 + 1, 0, size[0] - 1)
    j1 = np.clip(j0 + 1, 0, size[1] - 1)

    def slice_array(x, y):
        slice_array = [slice(None)] * len(array.shape)
        slice_array[axes[0]] = x
        slice_array[axes[1]] = y
        return slice_array

    broadcast = []
    for a in range(len(array.shape)):
        if a not in axes:
            broadcast.append(None)
        if a in axes and slice(None) not in broadcast:
            broadcast.append(slice(None))
    broadcast = tuple(broadcast)
    a, wa = array[slice_array(i0, j0)], ((i1 - i) * (j1 - j))[broadcast]
    b, wb = array[slice_array(i1, j0)], ((i1 - i) * (j - j0))[broadcast]
    c, wc = array[slice_array(i0, j1)], ((i - i0) * (j1 - j))[broadcast]
    d, wd = array[slice_array(i1, j1)], ((i - i0) * (j - j0))[broadcast]

    return wa * a + wb * b + wc * c + wd * d
