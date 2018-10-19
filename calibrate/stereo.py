import numpy as np
import cv2
from mega import camera
from mega import grayscale
from mega import interpolate_affine
from mega import remap_images
from .irregular_array import irregular_array
from .cameras import cameras as calibrate_cameras
from .cameras import reprojection_error


def single_stereo(points3D, left2D, right2D, image_shape, cameras=None):
    """stereo calibrate two cameras from world and image calibration points.

    # Parameters:
    points3D : (N, 3) array or (V, N, 3) array
        World coordinates in 3D for V views of N 3D points. Broadcasted *up* to
        points2D.
    left2D and right2D : (V, N, 2) array or irregular_array of (2,) arrays
        Camera image corrdinates in 2D for V views of N 2D points. Not
        broadcasted. An irregular_array can contain None.
    image_shape : (height, width, channels) or (height, width)
        Image shape with our without channels.
    cameras : (2,) array of mega.camera, optional
        If supplied, will use these camera parameters and only calibrate the
        world transformation between the two cameras.

    # Returns:
    cameras : (2,) array of mega.cameras or Nones
        Returns two cameras, or Nones if no stereo set up could be estimated.
    essential : (3, 3) array or None
        The essential matrix, or None if the calibration failed.
    fundamental : (3, 3) array or None
        The fundamental matrix, or None if the calibration failed.

    # Notes:
    This function is the basic calibration tool for two cameras in stereo.
    Whenever a None is observed in left2D or right2D, we gracefully skip it and
    use the remaining information. In case of missing views, the camera and all
    ghost cameras are estimated *except* for the missing views, where the ghost
    camera is set to None.
    """
    if cameras is None:
        cams = np.full(2, None)
    else:
        cams = np.asanyarray(cameras)
    essential = None
    fundamental = None
    if (not isinstance(left2D, np.ndarray) or left2D.dtype == object or
       not isinstance(right2D, np.ndarray) or right2D.dtype == object):
        # Irregular: might have None somewhere!
        masks = left2D, right2D
        points3D = irregular_array(points3D).filter_none(masks=masks)
        left2D = irregular_array(left2D).filter_none(masks=masks)
        right2D = irregular_array(right2D).filter_none(masks=masks)

    args = [points3D, left2D, right2D, None, None, None, None,
            (image_shape[1], image_shape[0])]
    flags = None
    if cameras is not None:
        args[3] = cams[0].K.astype(np.float32)
        args[4] = cams[0].distortion.astype(np.float32)
        args[5] = cams[1].K.astype(np.float32)
        args[6] = cams[1].distortion.astype(np.float32)
        flags = cv2.CALIB_FIX_INTRINSIC
    success, *rest = cv2.stereoCalibrate(*args, flags=flags)
    if success:
        K0, dist0, R0, t0 = *rest[:2], np.eye(3), np.zeros(3)
        K1, dist1, R1, t1 = rest[2:6]
        dist0, dist1, t1 = np.squeeze(dist0), np.squeeze(dist1), np.squeeze(t1)
        essential, fundamental = rest[6:8]
        cams[0] = camera(K0, R0, t0, dist0)
        cams[1] = camera(K1, R1, t1, dist1)
    return cams, essential, fundamental


def stereo(points3D, left2D, right2D, image_shape, cameras=None):
    """stereo calibrate cameras from world and image calibration points.

    # Parameters:
    # Parameters:
    points3D : array
        World coordinates.
    left2D and right2D : array
        Camera image corrdinates.
    image_shape : array
        Image shape.
    cameras : array of mega.camera, optional
        Fixed camera parameters.

    # Returns:
    cameras : sequence of two arrays of mega.camera or None
        Returns cameras or Nones if the calibrations failed.
    essential : array of arrays or None
        Returns essential matriices or None if the calibrations failed.
    fundamental : array of arrays or None
        Returns fundamental matriices or None if the calibrations failed.


    # Notes:
    This function is a multidimensional for-loop over the arrays of points3D
    and points2D and calling calibrate.single_stereo().
    """
    shape2D = irregular_array(left2D).shape
    if shape2D != irregular_array(right2D).shape:
        raise RuntimeError("left2D and right2D must have same dimensions!")
    points3D = np.broadcast_to(points3D, shape2D[:-1] + (3,))
    if cameras is None:
        cams = np.full((2,) + points3D.shape[:-3], None)
    else:
        cams = np.asanyarray(cameras)
    essential = np.full(points3D.shape[:-3], None)
    fundamental = np.full(points3D.shape[:-3], None)
    for idx in np.ndindex(essential.shape):
        cidx = (slice(None),) + idx
        args = points3D[idx], left2D[idx], right2D[idx], image_shape
        if cameras is not None:
            args = args + (cams[cidx],)
        cams[cidx], essential[idx], fundamental[idx] = single_stereo(*args)
    return cams, essential, fundamental


class stereo_from_images:
    def __init__(self, calibration_object, images):
        self.points3D = calibration_object.points3D.copy()
        self.image_shape = images[0][0].shape
        self.points2D = calibration_object.find_in_images(images)
        args = self.points3D, self.points2D, self.image_shape
        self.cameras, self.ghost_cameras = calibrate_cameras(*args)
        (self.cameras,
         self.essential,
         self.fundamental) = stereo(self.points3D, *self.points2D,
                                    self.image_shape, self.cameras)

    def reprojection_error(self):
        return reprojection_error(self.points3D, self.points2D, self.cameras)


class projector_camera_from_images:
    def __init__(self, calibration_object, images, mask=None):
        self.points3D = calibration_object.points3D.copy()
        self.image_shape = images[0][0].shape
        self.points2D = calibration_object.find_in_images(images)
        window = 10
        self.points2D = [
            self.points2D,
            interpolate_affine(images[1], self.points2D, window, mask)
        ]
        self.cameras, self.ghost_cameras = calibrate_cameras(*args)
        (self.cameras,
         self.essential,
         self.fundamental) = stereo(self.points3D, *self.points2D,
                                    self.image_shape, self.cameras)

    def reprojection_error(self):
        return reprojection_error(self.points3D, self.points2D, self.cameras)
