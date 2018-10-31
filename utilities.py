import numpy as np
import os
import cv2


def parallelize(setting):
    if setting:
        os.environ['MEGA_PARALLELIZE'] = 1
    else:
        os.environ['MEGA_PARALLELIZE'] = 0


def rgb2gray(array, method="standard"):
    array = np.asanyarray(array)  # will copy if images are not numpy arrays
    if method == "standard":
        v = np.array([0.299, 0.587, 0.114], np.float32)  # R, G, B
        g = np.einsum('...i,...i->...', array.astype(v.dtype), v)
        return g.astype(array.dtype)[..., None]
    if method == "average":
        return array.mean(axis=-1)[..., None]


def grayscale(array, method="standard"):
    array = np.asanyarray(array)
    if array.shape[-1] > 1:
        return rgb2gray(array, method)
    else:
        return array


def rescale(images, scale, scale_W=None):
    H, W, C = images.shape[-3:]
    if scale_W is None:
        scale_W = scale
    H, W = int(scale * H), int(scale_W * W)
    return_array = np.empty((*images.shape[:-3], H, W, C), images.dtype)
    for idx in np.ndindex(images.shape[:-3]):
        im = cv2.resize(images[idx], (W, H), cv2.INTER_CUBIC)
        return_array[idx] = im.reshape(H, W, C)
    return return_array


def remap_images(images, maps, map_depth=2):
    images, maps = np.asanyarray(images), np.asanyarray(maps)
    # Broadcast arrays (nontrivial as the shapes are not equal)
    if len(images.shape[:-3]) > len(maps.shape[1:-map_depth]):
        shape = (2,) + images.shape[:-3] + maps.shape[-map_depth:]
        maps = np.broadcast_to(maps, shape)
    if len(images.shape[:-3]) > len(maps.shape[1:-map_depth]):
        shape = maps.shape[1:-map_depth] + images.shape[-3:]
        images = np.broadcast_to(images, shape)
    result = np.empty_like(images)
    for idx in np.ndindex(images.shape[:-3]):
        result[idx] = cv2.remap(images[idx], maps[0, idx],
                                maps[1, idx], cv2.INTER_LINEAR)
    return result


def bilinear_interpolate(array, i, j, axes=(-3, -2)):
    size = array.shape[axes[0]], array.shape[axes[1]]

    i0 = np.clip(np.floor(i).astype(int), 0, size[0] - 1)
    j0 = np.clip(np.floor(j).astype(int), 0, size[1] - 1)
    i1 = np.clip(i0 + 1, 0, size[0] - 1)
    j1 = np.clip(j0 + 1, 0, size[1] - 1)

    def slice_array(x, y):
        slice_array = [slice(None)] * len(array.shape)
        slice_array[axes[0]] = x
        slice_array[axes[1]] = y
        return tuple(slice_array)

    shape = []
    count = 0
    for a in range(len(array.shape)):
        if a not in axes:
            shape.append(1)
        if a in axes and count < len(i.shape):
            shape.append(i.shape[count])
            count += 1
    a, wa = array[slice_array(i0, j0)], ((i1 - i) * (j1 - j)).reshape(shape)
    b, wb = array[slice_array(i1, j0)], ((i1 - i) * (j - j0)).reshape(shape)
    c, wc = array[slice_array(i0, j1)], ((i - i0) * (j1 - j)).reshape(shape)
    d, wd = array[slice_array(i1, j1)], ((i - i0) * (j - j0)).reshape(shape)
    result = wa * a
    result += wb * b
    result += wc * c
    result += wd * d

    return result


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
