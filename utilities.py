import numpy as np
import os
import cv2
from concurrent.futures import ThreadPoolExecutor as thread_pool
from concurrent.futures import as_completed
from numba import jit


@jit(nopython=True)
def rint(array, dtype=int):
    return np.rint(array).astype(dtype)


@jit(nopython=True)
def fint(array, dtype=int):
    return np.floor(array).astype(dtype)


@jit(nopython=True)
def cint(array, dtype=int):
    return np.ceil(array).astype(dtype)


def ndtake(slicing, axis):  # simple slicing, much faster than np.take
    if axis >= 0:
        return (slice(None),) * axis + (slicing,)
    if axis < 0:
        return (Ellipsis, slicing,) + (slice(None),) * (-1 - axis)


def ndsplit(array, widths, axis=0):  # split axis into len(widths) parts
    splits = (0,) + tuple(np.cumsum(widths))
    return [array[ndtake(slice(s0, s1), axis)]
            if s1 - s0 > 1 else array[ndtake(s0, axis)]
            for s0, s1 in zip(splits[:-1], splits[1:])]


def parallelize(setting):
    if setting:
        os.environ['MEGA_PARALLELIZE'] = 1
    else:
        os.environ['MEGA_PARALLELIZE'] = 0


def vectordot(u, v, *args, **kwargs):
    """Specilization of the dot-operator. u and v are ndarrays of vectors"""
    u, v = np.broadcast_arrays(u, v)
    return np.einsum('...i,...i ->...', u, v).reshape(*u.shape[:-1], 1)



# Precomputed LUTs for combine_HDR
triangle_weights_half = np.arange(1, 129)
triangle_weights = np.empty((256,))
triangle_weights[:128] = triangle_weights_half
triangle_weights[128:] = triangle_weights_half[::-1]
del triangle_weights_half
linear_response = np.arange(0, 256, dtype=np.float32)
linear_response[0] = 1
log_linear_response = np.log(linear_response)
del linear_response


def combine_HDR_vectorized(frames, exposure_times=None, out=None, weight=None):
    # debevec method (very memory intensive and slow. Dont know why.)
    if exposure_times is None:
        exposures = np.arange(0, frames.shape[0], dtype=np.float32) / 9
    else:
        exposures = -np.log(exposure_times)
    if out is None:
        out = np.zeros((frames.shape[1:]), dtype=np.float32)
    if weight is None:
        weight = np.zeros((*frames.shape[1:-1], 1), dtype=np.float32)
    for i, exposure in enumerate(exposures):
        frame = frames[i]
        response = log_linear_response[frame] + exposure
        weights = np.sum(triangle_weights[frame], axis=-1, keepdims=True)
        weight += weights
        out += weights * response
    out /= weight
    return np.exp(out)


def combine_HDR(frames, exposure_times=None, step=1):
    list_shape = frames.shape[1:-3]
    if len(list_shape) == 0:
        return combine_HDR_vectorized(frames, exposure_times)
    if exposure_times is None:
        exposure_times = np.arange(0, frames.shape[0], dtype=np.float32)
        exposure_times = np.power(2, exposure_times) / 2**9
    result = np.zeros((frames.shape[1:]), dtype=np.float32)
    tmp_weights = np.zeros((step, *frames.shape[-3:-1], 1), dtype=np.float32)
    iter = np.ndindex(list_shape)
    for idx in iter:
        [iter.next() for i in range(step - 1)]
        idcs = idx[:-1] + (slice(idx[-1], idx[-1] + step),)
        ith_frames = frames[(slice(None),) + idcs]
        ith_tmp_weights = tmp_weights[:min(frames.shape[-4] - idx[-1], step)]
        result[idcs] = combine_HDR_vectorized(ith_frames, exposure_times,
                                              out=result[idcs],
                                              weight=ith_tmp_weights)
    return result


def combine_HDR_using_opencv(frames, exposure_times=None):
    list_shape = frames.shape[1:-3]
    if len(list_shape) == 0:
        return combine_HDR_using_opencv(frames[:, None], exposure_times)[0]
    if exposure_times is None:
        exposure_times = np.arange(0, frames.shape[0], dtype=np.float32)
        exposure_times = 255 * np.power(2, exposure_times)
    result = np.empty((frames.shape[1:]), dtype=np.float32)
    merge_debvec = cv2.createMergeDebevec()
    for idx in np.ndindex(list_shape):
        frame = frames[(slice(None),) + idx]
        result[idx] = merge_debvec.process(frame, times=exposure_times.copy())
    return result
