import numpy as np
from numba import vectorize


@vectorize(["f4(f4,f4,f4)", "f8(f8,f8,f8)"], nopython=True)
def _rgb2gray(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b

_rgb2gray_uint8_LUT = _rgb2gray(*np.mgrid[0:256, 0:256, 0:256]).astype('u1')


@vectorize(["u1(u1,u1,u1)"], nopython=True)
def _grayscale_uint8(r, g, b):
    return _rgb2gray_uint8_LUT[r, g, b]


def grayscale(array):
    r, g, b = array[..., 0], array[..., 1], array[..., 2]
    if array.dtype == np.uint8:
        return np.expand_dims(_grayscale_uint8(r, g, b), axis=-1)
    else:
        return np.expand_dims(_rgb2gray(r, g, b), axis=-1)
