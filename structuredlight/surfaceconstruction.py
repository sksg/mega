import numpy as np
from concurrent.futures import ThreadPoolExecutor as thread_pool
# from concurrent.futures import ProcessPoolExecutor as process_pool
# import time  # to evaluate sppedup from parallelization


def _length(v, *args, **kwargs):
    """Short hand for numpy.linalg.norm"""
    return np.linalg.norm(v, *args, **kwargs)


def _normalize(v, *args, **kwargs):
    """Short hand for v / numpy.linalg.norm(v, keepdims=True)"""
    return v / _length(v, keepdims=True, *args, **kwargs)


def _vectordot(u, v, *args, **kwargs):
    """Specilization of the dot-operator. u and v are ndarrays of vectors"""
    u, v = np.broadcast_arrays(u, v)
    return np.einsum('...i,...i ->...', u, v).reshape(*u.shape[:-1], 1)


def triangulate(calibration, indices):
    """Finds mid point between two pixel rays"""
    d = calibration.cam0.position - calibration.cam1.position
    p = calibration.to_focal_plane(indices)
    v0 = _normalize(p[0] - calibration.cam0.position, axis=1)
    v1 = _normalize(p[1] - calibration.cam1.position, axis=1)
    d_v0 = _vectordot(d, v0)
    d_v1 = _vectordot(d, v1)
    w = _vectordot(v0, v1)
    c0 = (d_v1 * w - d_v0) / (1 - w**2)
    c1 = (d_v1 - d_v0 * w) / (1 - w**2)
    p0 = calibration.cam0.position + c0 * v0
    p1 = calibration.cam1.position + c1 * v1
    return (p0 + p1) / 2


def triangulate_epipolar(calibration, indices):
    R0 = calibration.rectified_cam0.rotation.T
    Q = calibration.disparity_to_depth_map
    disparity = indices[0, None, :, 1] - indices[1, None, :, 1]
    I1 = np.ones(disparity.shape)
    depth_map = Q.dot(np.vstack((indices[0, :, ::-1].T, disparity, I1)))
    return R0.dot(depth_map[:3, :] / depth_map[3, None, :]).T


def match_epipolar_maps_vectorized(maps, masks):

    mask0, mask1 = masks[0], masks[1]
    if mask0.sum() == 0 or mask1.sum() == 0:
        return np.empty((2, 0))
    # To conserve computation size, we only take a valid continuous subset!
    b0, e0 = mask0.T.nonzero()[0][0], mask0.T.nonzero()[0][-1] + 1
    b1, e1 = mask1.T.nonzero()[0][0], mask1.T.nonzero()[0][-1] + 1
    mask0, mask1 = mask0[:, b0:e0], mask1[:, b1:e1]
    map0, map1 = maps[0, :, b0:e0], maps[1, :, b1:e1]

    # both the left and right side of a match must exist
    match = np.logical_and(mask0[:, :, None], mask1[:, None, :-1])
    match = np.logical_and(match, mask1[:, None, 1:])

    # value in [0] should lie between to neighbouring values in [1]
    match[map0[:, :, None] < map1[:, None, :-1]] = 0
    match[map0[:, :, None] >= map1[:, None, 1:]] = 0

    r, c0, c1 = match.nonzero()
    c1f = map0[r, c0] - map1[r, c1]
    c1f /= map1[r, c1 + 1] - map1[r, c1]
    c1f += c1
    return np.swapaxes(np.array([(r, c0 + b0), (r, c1f + b1)]), 1, 2)


def match_epipolar_maps_sequential(maps, masks, step=1):
    pixels = []
    for i in range(0, maps.shape[1], step):
        ith_pixels = match_epipolar_maps_vectorized(maps[:, i:i + step],
                                                    masks[:, i:i + step])
        if ith_pixels.shape[1] > 0:
            ith_pixels[:, :, 0] += i
            pixels.append(ith_pixels)
    return np.hstack(pixels)


def match_epipolar_maps(maps, masks, step=1):
    pixels = []
    pool = thread_pool()
    futures = []
    for i in range(0, maps.shape[1], step):
        futures.append(pool.submit(match_epipolar_maps_vectorized,
                                   maps[:, i:i + step], masks[:, i:i + step]))
    for i in range(0, maps.shape[1], step):
        ith_pixels = futures[i // step].result()
        if ith_pixels.shape[1] > 0:
            ith_pixels[:, :, 0] += i
            pixels.append(ith_pixels)
    return np.hstack(pixels)
