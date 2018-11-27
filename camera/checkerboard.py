import numpy as np
import cv2
from .remap import rescale


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

    def find_in_images(self, images, draw=None):
        images = grayscale(images)  # ensure grayscale
        shape = images.shape[:-3]
        points3D, points2D = [], []
        mask = np.zeros(shape + (len(self.points3D),), bool)  # == False
        for i in range(shape[0]):
            if len(shape) > 1:
                dr = None
                if draw is not None:
                    dr = draw[i]
                p3D, p2D, m = self.find_in_images(images[i], dr)
                if len(p3D) > 0:
                    points3D.append(p3D)
                    points2D.append(p2D)
                    mask[i] = m
            else:
                image = rescale(images[i], self.corse_rescale)
                (success,
                 corners) = cv2.findChessboardCorners(image, self.NxM,
                                                      self._check_criteria)
                if success:
                    corners = corners[:, 0]
                    if self.corse_rescale is not None:
                        corners /= self.corse_rescale
                    corners = cv2.cornerSubPix(images[i], corners, (11, 11),
                                               (-1, -1), self._term_criteria)
                    points2D.append(corners.astype(self.dtype)[..., ::-1])
                    points3D.append(self.points3D)
                    mask[i] = True
                    if draw is not None:
                        cv2.drawChessboardCorners(draw[i], self.NxM, corners,
                                                  success)
                else:
                    mask[i] = False
                    points3D.append(self.points3D)
                    points2D.append(np.nan)
        return points3D, points2D, mask


def grayscale(images):
    images = np.asanyarray(images)  # will copy if images are not numpy arrays
    if images.dtype == object:
        raise RuntimeError("For now, don't use numpy object arrays!")
    if images.shape[-1] == 1:
        return images
    if images.shape[-1] == 3:  # Assuming RGB
            v = np.array([0.299, 0.587, 0.114])
            g = np.einsum('...i,...i->...', images.astype(v.dtype), v)
            return g.astype(images.dtype)[..., None]
    else:
        err = "Image format unknown. Please perform grayscale yourself!!"
        raise RuntimeError(err)
