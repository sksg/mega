import numpy as np
import os
import cv2
from numba import guvectorize
from .homography import homography


def _bilinear(image, i, j, out):
    step = 32766  # opencv limit!
    for n in range(0, i.shape[0], step):
        for m in range(0, i.shape[1], step):
            _i = i[n:n + step, m:m + step]
            _j = j[n:n + step, m:m + step]
            im = cv2.remap(image, _j, _i, cv2.INTER_LINEAR)
            out[n:n + step, m:m + step] = im

_bilinear_v = guvectorize(['(u1[:,:,::1],f4[:,::1],f4[:,::1],u1[:,:,::1])'],
                          '(n,m,c),(l,k),(l,k)->(l,k,c)')(_bilinear)


def bilinear(image, i, j, kind=None):
    _i, _j = np.atleast_2d(i), np.atleast_2d(j)
    out = _bilinear_v(image, _i.astype(np.float32), _j.astype(np.float32))
    return out.reshape((*out.shape[:-3], *i.shape[-2:], out.shape[-1]))


def _cubic_rescale(image, out):
    H, W = out.shape[:2]
    out[:] = np.atleast_3d(cv2.resize(image, (W, H), cv2.INTER_CUBIC))


def _bilinear_rescale(image, out):
    H, W = out.shape[:2]
    out[:] = cv2.resize(image, (W, H), cv2.INTER_LINEAR)

_cubic_rescale_v = guvectorize(['(u1[:,:,::1],u1[:,:,::1])'],
                               '(n,m,c),(l,k,c)')(_cubic_rescale)
_bilinear_rescale_v = guvectorize(['(u1[:,:,::1],u1[:,:,::1])'],
                                  '(n,m,c),(l,k,c)')(_bilinear_rescale)


def cubic_rescale(image, scale, scale_W=None):
    if scale_W is None:
        scale_W = scale
    H, W, C = image.shape[-3:]
    H, W = int(scale * H), int(scale_W * W)
    out = np.empty((*image.shape[:-3], H, W, C), image.dtype)
    _cubic_rescale_v(image, out)
    return out


def bilinear_rescale(image, scale, scale_W=None):
    if scale_W is None:
        scale_W = scale
    H, W, C = image.shape[-3:]
    H, W = int(scale * H), int(scale_W * W)
    out = np.empty((*image.shape[:-3], H, W, C), image.dtype)
    _bilinear_rescale_v(image, out)
    return out


def _affine(image, points, mask=None, window=10, tol=50):
    window = np.mgrid[-window:window, -window:window].T.reshape(-1, 2)
    p = points
    end = np.array(image.shape[-3:-1]) - 1
    p_window = np.clip(p.astype(int) + window, 0, end).T
    i_window = image[(*p_window,)]
    m_window = np.squeeze(mask[(*p_window,)])
    if m_window.sum() < tol:
        return np.nan
    else:
        i_window = i_window[m_window][..., ::-1]
        p_window = p_window.T[m_window].astype(np.float32)[..., ::-1]
        H = homography(p_window, i_window)
        p = H.dot(np.array([*p[[1, 0]], 1])).astype(np.float32)
        return p[[1, 0]] / p[2]

_exc = (3, 4, 5, 'mask', 'window', 'tol')
affine = np.vectorize(_affine, excluded=_exc, signature='(n,m,2),(2)->(2)')
