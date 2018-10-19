import numpy as np
import cv2
from mega import camera
from mega import remap_images
from .cameras import undistort_rectify_maps


def stereo_rectify(left, right, image_shape):
    left, right = np.asanyarray(left), np.asanyarray(right)
    left, right = np.broadcast_arrays(left, right)
    rect_left = np.empty_like(left)
    rect_right = np.empty_like(right)
    rect_Q = np.empty_like(left)
    image_shape = image_shape[1], image_shape[0]
    for idx in np.ndindex(left.shape):
        l, r = left[idx], right[idx]
        R = r.R.T.dot(l.R)
        t = r.position - l.position
        rect_tuple = cv2.stereoRectify(l.K, l.distortion, r.K, r.distortion,
                                       image_shape, R, t, flags=0)
        R0, R1 = rect_tuple[:2]
        K0, K1 = rect_tuple[2][:, :3], rect_tuple[3][:, :3]
        t0 = np.squeeze(rect_tuple[2][:, 3])
        t1 = np.squeeze(rect_tuple[3][:, 3])
        rect_left[idx] = camera(K0, R0, t0, np.array(tuple()))
        rect_left[idx] = camera(K1, R1, t1, np.array(tuple()))
        rect_Q[idx] = rect_tuple[4]
    return (rect_left, rect_right), rect_Q


class rectification:
    def __init__(self, calibration, include_distortion=False):
        self.image_shape = calibration.image_shape
        self.cameras, self.Q = stereo_rectify(*calibration.cameras,
                                              calibration.image_shape)
        if include_distortion:
            self.maps = [undistort_rectify_maps(cam, calibration.image_shape)
                         for cam in self.cameras]

    def undistort_and_rectify_image(self, image):
        if self.image_shape != image.shape[-3:]:
            raise RuntimeError("Image does not match the internal structures")
        return remap_images(image, self.maps)
