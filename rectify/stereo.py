import numpy as np
import cv2
from mega import camera
from mega import remap_images


class stereo_from_cameras:
    def __init__(self, left, right, image_shape):
        left, right = np.asanyarray(left), np.asanyarray(right)
        left, right = np.broadcast_arrays(left, right)
        self.left = np.empty_like(left)
        self.right = np.empty_like(right)
        self.Q = np.empty_like(left)
        image_shape = image_shape[1], image_shape[0]
        for idx in np.ndindex(left.shape):
            l, r = left[idx], right[idx]
            R = r.R.T.dot(l.R)
            t = r.position - l.position
            rect_tuple = cv2.stereoRectify(l.K, l.distortion, r.K,
                                           r.distortion, image_shape, R, t,
                                           flags=0)
            R0, R1 = rect_tuple[:2]
            K0, K1 = rect_tuple[2][:, :3], rect_tuple[3][:, :3]
            t0 = np.squeeze(rect_tuple[2][:, 3])
            t1 = np.squeeze(rect_tuple[3][:, 3])
            self.left[idx] = camera(K0, R0, t0, np.array(tuple()))
            self.left[idx] = camera(K1, R1, t1, np.array(tuple()))
            self.Q[idx] = rect_tuple[4]
        self.left_map = undistort_rectify_maps(self.left, image_shape)
        self.right_map = undistort_rectify_maps(self.right, image_shape)

    def undistort_and_rectify_images(self, left, right):
        if self.image_shape != image.shape[-3:]:
            raise RuntimeError("Image does not match the internal structures")
        return (remap_images(left, self.left_map),
                remap_images(right, self.right_map))


def undistort_rectify_maps(camera, image_shape):
    image_shape = image_shape[1], image_shape[0]
    return cv2.initUndistortRectifyMap(camera.K, camera.distortion,
                                       camera.R, camera.P,
                                       image_shape, cv2.CV_32F)
