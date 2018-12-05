import numpy as np
import cv2
from ..interpolate import cubic_rescale as rescale
from ..parallize import parallize


@parallize('(y,x,1),_,_,_,_->(n,2)', default='sequential')
def find_checkerboard(image, NxM, coarse=None, check=None, term=None):
    rescaled = rescale(image, coarse)
    (success, corners) = cv2.findChessboardCorners(rescaled, NxM, check)
    if success:
        corners = corners[:, 0]
        if coarse is not None:
            corners /= coarse
        cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), term)
        return corners[..., ::-1]
    else:
        return corners * np.nan


class checkerboard:
    def __init__(self, NxM, size=1, coarse=None, dtype=np.float32):
        self.NxM = NxM
        self.size = size
        self.dtype = dtype
        self.coarse = coarse
        corners = np.mgrid[0:NxM[0], 0:NxM[1], 0:1].T.reshape(-1, 3)
        self.points3D = (corners * size).astype(dtype)
        self._term_crit = (cv2.TERM_CRITERIA_EPS +
                           cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self._chk_crit = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                          cv2.CALIB_CB_NORMALIZE_IMAGE +
                          cv2.CALIB_CB_FILTER_QUADS +
                          cv2.CALIB_CB_FAST_CHECK)

    def find_in_image(self, image):
        if image.shape[-1] != 1:
            raise ValueError('Image must be gray scale (image.shape[-1] = 1)!')
        points2D = find_checkerboard(image, self.NxM, self.coarse,
                                     self._chk_crit, self._term_crit)
        points3D = np.tile(self.points3D, image.shape[:-3] + (1, 1))
        points3D[np.isnan(points2D).any(axis=-1)] = np.nan
        return points3D, points2D
