import numpy as np
import cv2
from mega import camera
from mega import grayscale
from .irregular_array import irregular_array


def single_camera(points3D, points2D, image_shape):
    """calibrate a single camera from world and image calibration points.

    # Parameters:
    points3D : (N, 3) array or (V, N, 3) array
        World coordinates in 3D for V views of N 3D points. Broadcasted *up* to
        points2D.
    points2D : (V, N, 2) array or irregular_array of (2,) arrays
        Camera image corrdinates in 2D for V views of N 2D points. Not
        broadcasted. An irregular_array can contain None.
    image_shape : (height, width, channels) or (height, width)
        Image shape with our without channels.

    # Returns:
    camera : mega.camera or None
        Returns a camera, or None if calibration failed.
    ghost_cameras : array of mega.camera or None
        Returns a *ghost* camera for every view, or None if calibration failed
        or a view was None. A ghost camera represents the camera oriented at a
        particular view.

    # Notes:
    This function is the basic calibration tool for cameras.
    Whenever a None is observed in points2D, we gracefully skip it and use the
    remaining information. In case of missing views, the camera and all ghost
    cameras are estimated *except* for the missing views, where the ghost
    camera is set to None.
    """
    cam = None
    ghost_cams = np.full(len(points2D), None)
    valid = slice(None)  # to filter ghost cameras---initially all valid
    if not isinstance(points2D, np.ndarray) or points2D.dtype == object:
        # Irregular: might have None somewhere!
        valid = np.array([p is not None for p in points2D], bool)
        masks = (points2D,)
        points3D = irregular_array(points3D).filter_none(masks=masks)
        points2D = irregular_array(points2D).filter_none()
    args = points3D, points2D, (image_shape[1], image_shape[0]), None, None
    success, K, distortion, Rs, ts = cv2.calibrateCamera(*args)
    if success:
        distortion = np.squeeze(distortion)  # OpenCV quirk
        ts = np.squeeze(ts)                  #
        R, t = np.eye(len(K), dtype=K.dtype), np.zeros(len(K), K.dtype)
        cam = camera(K, R, t, distortion)
        for i, (R, t) in enumerate(zip(Rs, ts)):
            ghost_cams[valid][i] = camera(K, R, t, distortion)
    return cam, ghost_cams


def cameras(points3D, points2D, image_shape):
    """calibrate cameras from world and image calibration points.

    # Parameters:
    points3D : array
        World coordinates.
    points2D : array
        Camera image corrdinates.
    image_shape : array
        Image shape.

    # Returns:
    cameras : array of mega.camera or None
        Returns an array of cameras, or None where the calibration failed.
    ghost_cameras : array of mega.camera or None
        Returns cameras or None where calibration failed or view was None.

    # Notes:
    This function is a multidimensional for-loop over the arrays of points3D
    and points2D and calling calibrate.single_camera().
    """
    shape2D = irregular_array(points2D).shape  # Copy?? Don't think so!!
    points3D = np.broadcast_to(points3D, shape2D[:-1] + (3,))
    cams = np.full(points3D.shape[:-3], None)
    ghost_cams = np.full(points3D.shape[:-2], None)
    for idx in np.ndindex(cams.shape):
        p3D, p2D = points3D[idx], points2D[idx]
        cams[idx], ghost_cams[idx] = single_camera(p3D, p2D, image_shape)
    return cams, ghost_cams


def reprojection_error(points3D, points2D, cameras, fill_val=0):
    reproj2D = project_points3D(points3D, cameras)
    error = np.empty(cameras.shape, object)
    error.fill(None)
    has_no_None = True
    for idx in np.ndindex(cameras.shape):
        p2D, rp2D = points2D[idx], reproj2D[idx]
        if any(v is None for v in (p2D, rp2D)):
            error[idx] = fill_val
            has_no_None = False
        else:
            error[idx] = np.linalg.norm(p2D - rp2D) / len(p3D)
    if has_no_None:
        error = error.astype(np.float32)
    return error


class cameras_from_images:
    def __init__(self, calibration_object, images):
        self.points3D = calibration_object.points3D.copy()
        self.image_shape = images[0].shape
        self.points2D = calibration_object.find_in_images(grayscale(images))
        args = self.points3D, self.points2D, self.image_shape
        self.cameras, self.ghost_cameras = cameras(*args)

    def reprojection_error(self):
        return reprojection_error(self.points3D, self.points2D, self.cameras)
