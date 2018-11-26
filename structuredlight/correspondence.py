import numpy as np
import os
from ..parallize import parallize


def _reduce_map_columns(pixel_array):
    if pixel_array.dtype == object:
        return np.hstack([x for x in pixel_array.flatten() if x.size != 0])
    return pixel_array


@parallize('(x),(y),(x),(y),_->()', isvec=True, otypes=[object], passinfo=True,
           default='sequential', reducefunc=_reduce_map_columns)
def map_columns(pinfo, left, right, left_mask, right_mask, keep_unsure=False):
    if left_mask.sum() == 0 or right_mask.sum() == 0:
        return np.empty((2, 0))
    # To conserve computation size, we only take a valid continuous subset!
    nonzero0 = np.stack(left_mask.nonzero())
    nonzero1 = np.stack(right_mask.nonzero())
    b0, e0 = nonzero0.min(axis=1), nonzero0.max(axis=1) + 1
    b1, e1 = nonzero1.min(axis=1), nonzero1.max(axis=1) + 1
    # Note that any dimension but the columns must match
    b = tuple(*np.min([b0[:-1], b1[:-1]], axis=0))
    e = tuple(*np.max([e0[:-1], e1[:-1]], axis=0))
    b0, e0 = b + (b0[-1],), e + (e0[-1],)
    b1, e1 = b + (b1[-1],), e + (e1[-1],)
    nd_slice0 = tuple(slice(b, e) for b, e in zip(b0, e0))
    nd_slice1 = tuple(slice(b, e) for b, e in zip(b1, e1))
    left_mask, right_mask = left_mask[nd_slice0], right_mask[nd_slice1]
    left, right = left[nd_slice0], right[nd_slice1]

    # both the left and right side of a match must exist
    match = np.logical_and(left_mask[..., None], right_mask[..., None, :-1])
    match = np.logical_and(match, right_mask[..., None, 1:])

    # value in left should lie between to neighbouring values in right
    match[left[..., :, None] < right[..., None, :-1]] = 0
    match[left[..., :, None] >= right[..., None, 1:]] = 0

    if not keep_unsure:
        match[match.sum(axis=1) > 1] = 0

    if match.sum() == 0:
        return np.empty((2, 0))

    if np.any(match.sum(axis=1) > 1):
        print('wrong match', (match.sum(axis=1) > 1).sum())
        errors = (match.sum(axis=1) > 1)
        for e in errors.nonzero()[0]:
            print('i', e, 'phase', left[..., e])
            _e = match[e].nonzero()[0]
            _a, _b = np.min(_e), np.max(_e) + 1
            print('a,b', _a, _b)
            print('left', right[..., :-1][..., _a:_b])
            print('right', right[..., 1:][..., _a:_b])

    *index, c0, c1 = tuple(match.nonzero())
    step = right[(*index, c1 + 1)] - right[(*index, c1)]
    c1frac = (left[(*index, c0)] - right[(*index, c1)]) / step

    index = [i + _b for i, _b in zip(index, b)]
    pindex = [*np.broadcast_to(pinfo.index, (len(pinfo.index), len(c0)))]
    index0 = pindex + index + [c0 + b0[0]]
    index1 = pindex + index + [c1 + c1frac + b1[0]]
    return np.swapaxes(np.array([index0, index1]), 1, 2)
