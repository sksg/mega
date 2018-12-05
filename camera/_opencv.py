import numpy as np
import cv2


def cv_shape(shape):
    return (shape[1], shape[0]) if len(shape) < 3 else (shape[-2], shape[-3])


def cv_points(points, dtype=np.float32, mask=None):
    if points.shape[-1] == 2:
        coords_slice = slice(None, None, -1)
    else:
        coords_slice = slice(None)
    if isinstance(points, np.ndarray):
        if mask is None:
            mask = ~np.isnan(points).any(axis=-1)
        if mask.all():
            return points[..., coords_slice].astype(dtype)
        elif len(points.shape) == 2:
            return points[mask][..., coords_slice].astype(dtype)
        elif len(points.shape) > 2:
            mask = mask.any(axis=-1)
            return points[mask][..., coords_slice].astype(dtype)
    result = []
    for i, ps in enumerate(points):
        cv_ps = cv_points(ps)
        if len(cv_ps) > 0 and (mask is None or mask[i]):
            result.append(cv_ps)
    return result
