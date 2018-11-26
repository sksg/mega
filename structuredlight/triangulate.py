import numpy as np
import os
from ..parallize import parallize


def linearization(left_cam, right_cam, dtype=np.float32):
    P0 = left_cam.P
    P1 = right_cam.P
    e = np.eye(4, dtype=dtype)
    C = np.empty((4, 3, 3, 3), dtype)
    for i in np.ndindex((4, 3, 3, 3)):
        tmp = np.stack((P0[i[1]], P0[i[2]], P1[i[3]], e[i[0]]), axis=0)
        C[i] = np.linalg.det(tmp.T)
    C = C[..., None, None]
    yx = np.mgrid[0:left_cam.imshape[0], 0:left_cam.imshape[1]].astype(dtype)
    y, x = yx[None, 0, :, :], yx[None, 1, :, :]
    offset = C[:, 0, 1, 0] - C[:, 2, 1, 0] * x - C[:, 0, 2, 0] * y
    factor = -C[:, 0, 1, 2] + C[:, 2, 1, 2] * x + C[:, 0, 2, 2] * y
    return offset, factor


@parallize('(2),(2),_,_->(3)', isvec=True, default='vectorized')
def linear(left, right, offset, factor):
    idx = (slice(None), *(left + 0.5).astype(int).T)
    xyzw = offset[idx] + factor[idx] * right[None, ..., 1]
    return xyzw.T[..., :3] / xyzw.T[..., 3, None]
