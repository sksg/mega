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
        v = np.array([0.299, 0.587, 0.114])  # R, G, B
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
