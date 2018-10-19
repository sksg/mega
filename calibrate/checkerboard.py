import numpy as np
import cv2
from mega import grayscale, rescale


class checkerboard:
    """Checkerboard calibration object.

    # Parameters:
    NxM : (N, M)
        The shape of the checkerbord (corners) along the x- (N) and y-axis (M).
    size : number, optional
        The size of the checkerboard squares. Default is 1.
    corse_rescale : float between 0 and 1, or None, optional
        In the corse alignment step in find_in_image(...), the class can
        rescale the images making the alignment much faster. This as no effect
        on the fine alignment (subpixel) step afterwards. If None (default), no
        rescaling is done.
    dtype : numpy.dtype instance, optional
        The dtype of the self.points3D array. Default is numpy.float32.

    # Notes
    The checkerboard calibration uses the corners between adjacent squares to
    represent itself. As such, the shape NxM denotes the *corners* and not the
    squares, as is the standard in vision systems.

    As for any calibration object, it provides self.points3D array, which
    contains the 3D coordinates of the object in their own space. Furthermore,
    find_in_image() returns the corresponding pixels identified in an image,
    with subpixel precision.
    """

    def __init__(self, NxM, size=1, corse_rescale=None, dtype=np.float32):
        self.NxM = NxM
        self.size = size
        self.dtype = dtype
        self.corse_rescale = corse_rescale
        corners = np.mgrid[0:NxM[0], 0:NxM[1]].T.reshape(-1, 2)
        zero = np.broadcast_to(np.array(0), corners.shape[:-1])
        self.points3D = (np.c_[corners, zero] * size).astype(dtype)
        self._term_criteria = (cv2.TERM_CRITERIA_EPS +
                               cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self._check_criteria = (cv2.CALIB_CB_ADAPTIVE_THRESH +
                                cv2.CALIB_CB_NORMALIZE_IMAGE +
                                cv2.CALIB_CB_FILTER_QUADS +
                                cv2.CALIB_CB_FAST_CHECK)

    def find_in_images(self, images):
        images = grayscale(images)  # ensure gray scale
        return_shape = images.shape[:-3]
        points2D = np.empty(return_shape, object)
        points2D.fill(None)
        for idx in np.ndindex(return_shape):
            image = rescale(images[idx], self.corse_rescale)
            (success,
             corners) = cv2.findChessboardCorners(image, self.NxM,
                                                  self._check_criteria)
            if success:
                corners = corners[:, 0]
                if self.corse_rescale is not None:
                    corners *= self.corse_rescale
                cv2.cornerSubPix(images[idx], corners, (11, 11),
                                 (-1, -1), self._term_criteria)
                points2D[idx] = corners.astype(self.dtype)
        return points2D
