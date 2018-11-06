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


def fit_planes(points, mask=None):
    barycenters = points.mean(axis=-2)[..., None, :]
    baryvectors = (points - barycenters)
    if mask is not None:
        print(points.shape)
        print(mask.shape)
        baryvectors[~mask] *= 0
    M = (baryvectors[..., None, :] * baryvectors[..., None]).sum(axis=-3)
    eig_values, eig_vectors = np.linalg.eigh(M)
    i = tuple(np.arange(0, eig_values.shape[i], dtype=int)
              for i in range(0, len(eig_values.shape) - 1))
    indices = (*i, slice(None), np.abs(eig_values).argmin(axis=-1))
    return eig_vectors[indices]


def normals_from_SVD(camera, points3D, pixels, image_shape):
        def normalize(v):
            return v / np.linalg.norm(v, axis=-1, keepdims=True)
        mask = np.zeros(image_shape[-3:-1], bool)
        mask[(*pixels.T.astype(int),)] = 1
        xyz = np.zeros(image_shape[-3:-1] + (3,), points3D.dtype)
        xyz[mask] = points3D

        p3D = np.stack((xyz[1:-1, 1:-1][mask[:-2, :-2]],
                        xyz[1:-1, 1:-1][mask[1:-1, :-2]],
                        xyz[1:-1, 1:-1][mask[2:, :-2]],
                        xyz[1:-1, 1:-1][mask[:-2, 1:-1]],
                        xyz[1:-1, 1:-1][mask[1:-1, 1:-1]],
                        xyz[1:-1, 1:-1][mask[2:, 1:-1]],
                        xyz[1:-1, 1:-1][mask[:-2, 2:]],
                        xyz[1:-1, 1:-1][mask[1:-1, 2:]],
                        xyz[1:-1, 1:-1][mask[2:, 2:]]), axis=-2)
        _mask = np.stack((mask[1:-1, 1:-1][mask[:-2, :-2]],
                          mask[1:-1, 1:-1][mask[1:-1, :-2]],
                          mask[1:-1, 1:-1][mask[2:, :-2]],
                          mask[1:-1, 1:-1][mask[:-2, 1:-1]],
                          mask[1:-1, 1:-1][mask[1:-1, 1:-1]],
                          mask[1:-1, 1:-1][mask[2:, 1:-1]],
                          mask[1:-1, 1:-1][mask[:-2, 2:]],
                          mask[1:-1, 1:-1][mask[1:-1, 2:]],
                          mask[1:-1, 1:-1][mask[2:, 2:]]), axis=0)
        n = normalize(fit_planes(p3D, _mask.reshape((-1, 9))))

        c = normalize(camera.position - points3D)
        dev = (n * c).sum(axis=1)
        n *= np.sign(dev)[:, None]
        return n, np.abs(dev)


def normals_from_depth(camera, points3D, pixels, image_shape):
        def normalize(v):
            return v / np.linalg.norm(v, axis=-1, keepdims=True)
        mask = np.zeros(image_shape[-3:-1], bool)
        mask[(*pixels.T.astype(int),)] = 1
        xyz = np.zeros(image_shape[-3:-1] + (3,), points3D.dtype)
        xyz[mask] = points3D

        dxyzdx, dxyzdy = np.gradient(xyz, axis=(0, 1))
        dxyzdx, dxyzdy = dxyzdx[mask], dxyzdy[mask]

        n = normalize(np.cross(dxyzdx, dxyzdy))

        c = normalize(camera.position - points3D)
        dev = (n * c).sum(axis=1)
        n *= np.sign(dev)[:, None]
        return n, np.abs(dev)


def normals_from_gradients(projector, camera, points3D, p_pixels, c_pixels,
                           p_gradients, c_gradients):
        def normalize(v):
            return v / np.linalg.norm(v, axis=-1, keepdims=True)
        p = points3D - projector.position
        c = points3D - camera.position
        p_grad = np.zeros_like(points3D)
        c_grad = np.zeros_like(points3D)

        # Pick relevant gradients:
        p_grad[:, [1, 0]] = p_gradients[(*p_pixels.astype(int).T,)]
        c_grad[:, [1, 0]] = c_gradients[(*c_pixels.astype(int).T,)]
        # Invert gradients to get the periodic vectors (lambda)
        p_lambda = p_grad / (p_grad * p_grad).sum(axis=-1, keepdims=True)
        c_lambda = c_grad / (c_grad * c_grad).sum(axis=-1, keepdims=True)
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

        # return c_lambda[:, [1, 0, 2]], np.ones_like(p[:, 0])

        # normals from projection (see paper)
        X = _vectordot(p_omega, c_lambda) * p
        X -= _vectordot(p, c_lambda) * p_omega
        Y = _vectordot((p_lambda - c_lambda), c_lambda) * p
        Y -= _vectordot(p, c_lambda) * p_lambda

        n = normalize(np.cross(X, Y))

        dev = np.array([((-n * p).sum(axis=1)), ((n * c).sum(axis=1))])
        i = np.argmax(np.abs(dev), axis=0)
        n *= np.sign(dev[i, np.arange(len(i))])[:, None]
        return n, np.abs(dev)


def match_epipolar_maps_vectorized(maps, masks):

    mask0, mask1 = masks[0, ..., 0], masks[1, ..., 0]
    if mask0.sum() == 0 or mask1.sum() == 0:
        return np.empty((2, 0))
    # To conserve computation size, we only take a valid continuous subset!
    nonzero0, nonzero1 = np.stack(mask0.nonzero()), np.stack(mask1.nonzero())
    b0, e0 = nonzero0.min(axis=1), nonzero0.max(axis=1) + 1
    b1, e1 = nonzero1.min(axis=1), nonzero1.max(axis=1) + 1
    nd_slice0 = tuple(slice(b, e) for b, e in zip(b0, e0))
    nd_slice1 = tuple(slice(b, e) for b, e in zip(b1, e1))
    mask0, mask1 = mask0[nd_slice0], mask1[nd_slice1]
    map0, map1 = maps[(0, *nd_slice0, 0)], maps[(1, *nd_slice1, 0)]

    # both the left and right side of a match must exist
    match = np.logical_and(mask0[:, :, None], mask1[:, None, :-1])
    match = np.logical_and(match, mask1[:, None, 1:])

    # value in [0] should lie between to neighbouring values in [1]
    match[map0[:, :, None] < map1[:, None, :-1]] = 0
    match[map0[:, :, None] >= map1[:, None, 1:]] = 0

    if match.sum() == 0:
        return np.empty((2, 0))

    r, c0, c1 = match.nonzero()
    # TODO: Remove duplicates. These shouldn't be there, need to find out why.
    _, idx, count = np.unique(np.stack((r, c0)).T,
                              return_index=True,
                              return_counts=True,
                              axis=0)
    r, c0, c1 = r[idx], c0[idx], c1[idx]
    c1f = map0[r, c0] - map1[r, c1]
    c1f /= map1[r, c1 + 1] - map1[r, c1]
    c1f += c1
    return np.swapaxes(np.array([(r, c0 + b0[1]), (r, c1f + b1[1])]), 1, 2)


def match_epipolar_maps_sequential(maps, masks, step=1):
    pixels = []
    for i in range(0, maps.shape[1], step):
        ith_pixels = match_epipolar_maps_vectorized(maps[:, i:i + step],
                                                    masks[:, i:i + step])
        if ith_pixels.shape[1] > 0:
            ith_pixels[:, :, 0] += i
            pixels.append(ith_pixels)
    if pixels == []:
        return np.array([])
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
