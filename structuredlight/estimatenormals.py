import numpy as np
import os


def vectordot(u, v, *args, **kwargs):
    """Specilization of the dot-operator. u and v are ndarrays of vectors"""
    u, v = np.broadcast_arrays(u, v)
    return np.einsum('...i,...i ->...', u, v).reshape(*u.shape[:-1], 1)


def fit_planes(points, mask=None):
    barycenters = points.mean(axis=-2)[..., None, :]
    baryvectors = (points - barycenters)
    if mask is not None:
        baryvectors[np.logical_not(mask)] *= 0
    M = (baryvectors[..., None, :] * baryvectors[..., None]).sum(axis=-3)
    eig_values = np.full((len(baryvectors), 3), np.nan)
    eig_vectors = np.full((len(baryvectors), 3, 3), np.nan)
    v = ~np.isnan(M).any(axis=-1).any(axis=-1)
    eig_values[v], eig_vectors[v] = np.linalg.eigh(M[v])
    i = tuple(np.arange(0, eig_values.shape[i], dtype=int)
              for i in range(0, len(eig_values.shape) - 1))
    _indices = np.zeros(len(eig_vectors), int)
    _indices[v] = np.nanargmin(np.abs(eig_values[v]), axis=-1)
    indices = (*i, slice(None), _indices)
    return eig_vectors[indices]


def normalize(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def from_SVD(camera, points3D, pixels):
    xyz = np.full(tuple(camera.imshape) + (3,), np.nan, points3D.dtype)
    xyz[(*(pixels + 0.5).T.astype(int),)] = points3D
    mask = np.zeros(tuple(camera.imshape) + (9,), bool)
    mask[(*(pixels + 0.5 + np.array([-1, -1])).T.astype(int), 0)] = 1
    mask[(*(pixels + 0.5 + np.array([0, -1])).T.astype(int), 1)] = 1
    mask[(*(pixels + 0.5 + np.array([1, -1])).T.astype(int), 2)] = 1
    mask[(*(pixels + 0.5 + np.array([-1, 0])).T.astype(int), 3)] = 1
    mask[(*(pixels + 0.5 + np.array([0, 0])).T.astype(int), 4)] = 1
    mask[(*(pixels + 0.5 + np.array([1, 0])).T.astype(int), 5)] = 1
    mask[(*(pixels + 0.5 + np.array([-1, 1])).T.astype(int), 6)] = 1
    mask[(*(pixels + 0.5 + np.array([0, 1])).T.astype(int), 7)] = 1
    mask[(*(pixels + 0.5 + np.array([1, 1])).T.astype(int), 8)] = 1

    p3D = np.stack([xyz[mask[:, :, i]] for i in range(9)], axis=-2)
    nmsk = np.stack([mask[mask[:, :, i]][:, 4] for i in range(9)], axis=-1)

    n = normalize(fit_planes(p3D, nmsk))

    nan_count = np.isnan(n).sum()
    if nan_count > 0:
        print('Warning: from_SVD() produced {} nans'.format(nan_count))

    c = normalize(camera.position - xyz[mask[:, :, 4]])
    dev = (n * c).sum(axis=1)
    n *= np.sign(dev)[:, None]
    return n, np.abs(dev)


def from_depth(camera, points3D, pixels):
    mask = np.zeros(camera.imshape, bool)
    mask[(*(pixels + 0.5).T.astype(int),)] = 1
    xyz = np.full(tuple(camera.imshape) + (3,), np.nan, points3D.dtype)
    xyz[mask] = points3D

    dxyzdx, dxyzdy = np.gradient(xyz, axis=(0, 1))
    dxyzdx, dxyzdy = dxyzdx[mask], dxyzdy[mask]

    n = normalize(np.cross(dxyzdx, dxyzdy))

    nan_count = np.isnan(n).sum()
    if nan_count > 0:
        print('Warning: from_depth() produced {} nans'.format(nan_count))

    c = normalize(camera.position - points3D)
    dev = (n * c).sum(axis=1)
    n *= np.sign(dev)[:, None]
    return n, np.abs(dev)


def from_gradients(projector, camera, points3D, p_pixels, c_pixels,
                   p_gradients, c_gradients):
    p = points3D - projector.position
    c = points3D - camera.position
    p_l = p.dot(projector.R.T)
    c_l = c.dot(camera.R.T)
    p_grad = np.zeros_like(points3D)
    c_grad = np.zeros_like(points3D)

    # Pick relevant gradients:
    p_grad[:, [1, 0]] = p_gradients[(*p_pixels.astype(int).T,)]
    c_grad[:, [1, 0]] = c_gradients[(*c_pixels.astype(int).T,)]
    # Invert gradients to get the periodic vectors (lambda)
    p_lambda = p_grad / (p_grad * p_grad).sum(axis=-1, keepdims=True)
    c_lambda = c_grad / (c_grad * c_grad).sum(axis=-1, keepdims=True)
    # Project to the point depth
    p_lambda[:, :2] *= p_l[:, -1, None] / projector.focal_vector
    c_lambda[:, :2] *= c_l[:, -1, None] / camera.focal_vector
    # Rotate into common world space
    p_lambda = p_lambda.dot(projector.R)
    c_lambda = c_lambda.dot(camera.R)
    # Normalize rays
    p = normalize(p)
    c = -normalize(c)  # minus from formula, though it should not matter
    # Make perpendicular to rays (needed for formulas below)
    err = (vectordot(p_lambda, p) - np.vdot(p_lambda, p))
    p_lambda -= vectordot(p_lambda, p) * p
    c_lambda -= vectordot(c_lambda, c) * c
    # Finally, we make orthonormal sets x, x_lambda, x_omega
    p_omega = normalize(np.cross(p, p_lambda))
    # c_omega = normalize(np.cross(c, c_lambda))

    # normals from projection (see paper)
    X = vectordot(p_omega, c_lambda) * p
    X -= vectordot(p, c_lambda) * p_omega
    Y = vectordot((p_lambda - c_lambda), c_lambda) * p
    Y -= vectordot(p, c_lambda) * p_lambda

    n = normalize(np.cross(X, Y))

    nan_count = np.isnan(n).sum()
    if nan_count > 0:
        print('Warning: from_gradients() produced {} nans'.format(nan_count))

    dev = np.array([((-n * p).sum(axis=1)), ((n * c).sum(axis=1))])
    i = np.argmax(np.abs(dev), axis=0)
    n *= np.sign(dev[i, np.arange(len(i))])[:, None]
    return n, np.abs(dev)
