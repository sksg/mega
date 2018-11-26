import numpy as np
import cv2
from .camera import camera as camera_class
from ._opencv import cv_shape
from ._opencv import cv_points
from ..parallize import parallize


p_kwargs = dict(default='sequential')


@parallize('()->(),()', otypes=[object, object], **p_kwargs)
def undistort_map(camera):
    o1, o0 = cv2.initUndistortRectifyMap(camera.K, camera.distortion,
                                         np.eye(3, dtype=np.float32),
                                         camera.K,
                                         cv_shape(camera.imshape),
                                         cv2.CV_32F)
    return o0, o1


@parallize('(),(3,3),()->(),()', otypes=[object, object], **p_kwargs)
def undistort_rectify_map(camera, R, new_camera):
    o1, o0 = cv2.initUndistortRectifyMap(camera.K, camera.distortion,
                                         R, new_camera.P,
                                         cv_shape(camera.imshape),
                                         cv2.CV_32F)
    return o0, o1


@parallize('(),(n,2)->(n,2)', **p_kwargs)
def undistort_points(camera, points2D):
    shape = points2D.shape
    p2D = cv_points(points2D.reshape((1, -1, *shape[-1:])))
    p2D = cv2.undistortPoints(p2D, camera.K, camera.distortion, P=camera.K)
    return cv_points(p2D).reshape(shape)
