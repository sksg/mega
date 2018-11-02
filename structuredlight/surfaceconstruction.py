import numpy as np
import os
from scipy.optimize import minimize
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


def triangulate(camera0, camera1, indices, offset=None, factor=None,
                image_shape=None):
    if offset is None or factor is None:
        P0 = camera0.P
        P1 = camera1.P
        e = np.eye(4, dtype=np.float32)
        C = np.empty((4, 3, 3, 3), np.float32)
        for i in np.ndindex((4, 3, 3, 3)):
            tmp = np.stack((P0[i[1]], P0[i[2]], P1[i[3]], e[i[0]]), axis=0)
            C[i] = np.linalg.det(tmp.T)
        C = C[..., None, None]
        yx = np.mgrid[0:image_shape[-3], 0:image_shape[-2]].astype(np.float32)
        y, x = yx[None, 0, :, :], yx[None, 1, :, :]
        offset = C[:, 0, 1, 0] - C[:, 2, 1, 0] * x - C[:, 0, 2, 0] * y
        factor = -C[:, 0, 1, 2] + C[:, 2, 1, 2] * x + C[:, 0, 2, 2] * y
    idx = (slice(None), *indices[0].astype(int).T)
    xyzw = offset[idx] + factor[idx] * indices[1][None, :, 1]
    return xyzw.T[:, :3] / xyzw.T[:, 3, None]


def normals_from_gradients(projector, camera, points3D, p_pixels, c_pixels,
                           p_gradients, c_gradients):
        def normalize(v):
            return v / np.linalg.norm(v, axis=-1, keepdims=True)
        p = points3D - projector.position
        c = points3D - camera.position
        p_grad = np.zeros_like(points3D)
        c_grad = np.zeros_like(points3D)

        # Pick relevant gradients:
        p_grad[:, :2] = p_gradients[(*p_pixels.astype(int).T,)]
        c_grad[:, :2] = c_gradients[(*c_pixels.astype(int).T,)]
        # Invert gradients to get the periodic vectors (lambda)
        p_lambda = p_grad / (p_grad * p_grad).sum(-1, keepdims=True)
        c_lambda = c_grad / (c_grad * c_grad).sum(-1, keepdims=True)
        # Project to the point depth
        p_lambda[:, :2] *= p[:, -1, None] / projector.focal_vector
        c_lambda[:, :2] *= c[:, -1, None] / camera.focal_vector
        # Rotate into common world space
        p_lambda = p_lambda.dot(projector.R)
        c_lambda = c_lambda.dot(camera.R)
        # Normalize rays
        p = normalize(p)
        c = -normalize(c)  # minus from formula, though it should not matter
        # Make perpendicular to rays (needed for formulas below)
        p_lambda -= _vectordot(p_lambda, p) * p
        c_lambda -= _vectordot(c_lambda, c) * c
        # Finally, we make orthonormal sets x, x_lambda, x_omega
        p_omega = normalize(np.cross(p, p_lambda))
        c_omega = normalize(np.cross(c, c_lambda))

        # normals from projection (see paper)
        X = _vectordot(p_omega, c_lambda) * p
        X -= _vectordot(p, c_lambda) * p_omega
        Y = _vectordot((p_lambda - c_lambda), c_lambda) * p
        Y -= _vectordot(p, c_lambda) * p_lambda

        n = normalize(np.cross(X, Y))

        dev = np.array([((normals * p).sum(axis=1)),
                        ((normals * c).sum(axis=1))])

        dev = np.array([((-n * p).sum(axis=1)), ((n * c).sum(axis=1))])
        i = np.argmax(np.abs(dev), axis=0)
        n *= np.sign(dev[i, np.arange(len(i))])[:, None]
        return n, np.abs(dev)


def triangulate_epipolar(stereo, indices):
    R0 = stereo.rectified[0].R.T
    Q = stereo.Q
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


def match_epipolar_maps_parallel(maps, masks, step=1):
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


def match_epipolar_maps(maps, masks, step=1):
    if os.environ.get('MEGA_PARALLELIZE', default=False):
        return match_epipolar_maps_parallel(maps, masks, step)
    else:
        return match_epipolar_maps_sequential(maps, masks, step)


def normals_from_projection(lambda_p, lambda_c, omega_p, omega_c, p, c):
    X = _vectordot(omega_p, lambda_c) * p - _vectordot(p, lambda_c) * omega_p
    Y = _vectordot((lambda_p - lambda_c), lambda_c) * p
    Y -= _vectordot(p, lambda_c) * lambda_p
    normals = np.cross(X, Y)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    print("|A.cross(B)|", lengths.min(), lengths.mean(), lengths.max())
    print("|A.cross(B)| == 0", (lengths < 1e-2).sum())
    return normals / lengths
