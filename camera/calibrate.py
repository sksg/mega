import numpy as np
import cv2
from .camera import camera as _camera
from .camera import camera_project
from .remap import remap_images
from .remap import rectified_stereo_disparity_to_depth
from .remap import undistort
from .remap import undistort_and_rectify


def _cv_shape(shape):
    """Open CV always works in column-row order. We do the opposite."""
    return (shape[1], shape[0]) if len(shape) < 3 else (shape[-2], shape[-3])


def _cv_pixels(points2D):
    """Open CV always works in column-row order. We do the opposite."""
    if isinstance(points2D, np.ndarray):
        return points2D[..., ::-1].astype(np.float32)
    else:
        return [_cv_pixels(p2D) for p2D in points2D]


def camera(points3D, points2D, image_shape):
    """calibrate a single camera from world and image points."""
    reprojection_error, K, distortion, R_list, t_list = cv2.calibrateCamera(
        objectPoints=points3D,
        imagePoints=_cv_pixels(points2D),
        imageSize=_cv_shape(image_shape),
        # No inital guess!
        cameraMatrix=None, distCoeffs=None,
        flags=(cv2.CALIB_FIX_K3 +
               cv2.CALIB_FIX_ASPECT_RATIO +
               cv2.CALIB_ZERO_TANGENT_DIST)
    )
    if reprojection_error is None:  # Unsuccessful
        return None, None
    # OpenCV often represents vectors as 2D. This leads to dimensions of 1:
    distortion = np.squeeze(distortion)
    t_list = np.squeeze(t_list)
    ghosts = [_camera(K, R, t, distortion) for (R, t) in zip(R_list, t_list)]
    return _camera(K, distortion=distortion), ghosts


def stereo(points3D, points2D, image_shapes, cameras=None):
    """stereo calibrate two cameras from world and image points."""
    if cameras is None:
        cameras = [camera(points3D, points2D[0], image_shapes[0])[0],
                   camera(points3D, points2D[1], image_shapes[1])[0]]
    (reprojection_error,
     K0, distortion0,
     K1, distortion1,
     R1, t1, E, F) = cv2.stereoCalibrate(
        objectPoints=np.array(points3D),
        imagePoints1=np.array(_cv_pixels(points2D[0])),
        imagePoints2=np.array(_cv_pixels(points2D[1])),
        imageSize=_cv_shape(image_shapes[0]),
        cameraMatrix1=cameras[0].K,
        distCoeffs1=cameras[0].distortion,
        cameraMatrix2=cameras[1].K,
        distCoeffs2=cameras[1].distortion,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    if reprojection_error is None:  # Unsuccessful
        return np.full(2, None), None, None
    # OpenCV often uses 2D even for vectors. This leads to dimensions of 1:
    distortion1 = np.squeeze(distortion1)
    t1 = np.squeeze(t1)
    cameras = [_camera(cameras[0].K, distortion=cameras[0].distortion),
               _camera(cameras[1].K, R1, t1, cameras[1].distortion)]
    return cameras, E, F


def reprojection_errors(camera, points3D, points2D):
    reproj2D = camera.project(points3D)
    return [np.linalg.norm(p2D - rp2D) / len(p2D)
            for p2D, rp2D in zip(points3D, reproj2D)]


def rectify_stereo(cameras, image_shape):
    c0, c1 = cameras[:2]
    R, t = c1.relative_to(c0)
    R0, R1, P0, P1 = cv2.stereoRectify(cameraMatrix1=c0.K.astype('f8'),
                                       distCoeffs1=c0.distortion.astype('f8'),
                                       cameraMatrix2=c1.K.astype('f8'),
                                       distCoeffs2=c1.distortion.astype('f8'),
                                       imageSize=_cv_shape(image_shape),
                                       R=R.astype('f8'),
                                       T=t.astype('f8'),
                                       flags=0)[:4]
    return [_camera(R=R0, P=P0, distortion=c0.distortion),
            _camera(R=R1, P=P1, distortion=c1.distortion)]


class camera_calibration:
    def __init__(self, **kwargs):
        self.image_shape = kwargs.pop("image_shape", None)
        self.points3D = kwargs.pop("points3D", None)
        self.points2D = kwargs.pop("points2D", None)
        self.image_mask = kwargs.pop("image_mask", None)
        self.camera = kwargs.pop("camera", None)
        self.ghost_cameras = kwargs.pop("ghost_cameras", None)

    def process(self):
        self.camera, self.ghost_cameras = camera(self.points3D,
                                                 self.points2D,
                                                 self.image_shape)

    def reprojection_error(self):
        return reprojection_error(self.points3D, self.points2D, self.cameras)


def camera_from_images(calibration_object, images):
    calibration = camera_calibration()
    images = np.asarray(images)
    calibration.image_shape = images.shape[-3:]
    (calibration.points3D,
     calibration.points2D,
     calibration.image_mask) = calibration_object.find_in_images(images)
    calibration.process()
    return calibration


class stereo_calibration:
    def __init__(self, **kwargs):
        self.image_shape = kwargs.pop("image_shape", None)
        self.points3D = kwargs.pop("points3D", None)
        self.points2D = kwargs.pop("points2D", None)
        self.image_mask = kwargs.pop("image_mask", None)
        self.cameras = kwargs.pop("cameras", None)
        self.ghost_cameras = kwargs.pop("ghost_cameras", None)

    def process(self):
        self.cameras, self.ghost_cameras = [0, 0], [0, 0]  # dummy holders
        for i in (0, 1):
            c_tpl = cameras(self.points3D, self.points2D[i], self.image_shape)
            self.cameras[i], self.ghost_cameras[i] = c_tpl
        s_tuple = stereo(self.points3D, self.points2D,
                         self.image_shape, self.cameras)
        self.cameras, self.essential, self.fundamental = s_tuple
        self.rectified = None
        self.Q = None
        self.maps = None
        self.rectified_maps = None

    def reprojection_errors(self):
        errors = [reprojection_error(self.points3D, p2D, cam)
                  for p2D, cam in zip(self.points2D, self.cameras)]
        return errors

    def rectify(self, rectified_cameras=None):
        if rectified_cameras is None:
            self.rectified = rectify_stereo(self.cameras, self.image_shape)
        else:
            self.rectified = rectified_cameras
        self.Q = rectified_stereo_disparity_to_depth(self.rectified)
        self.rectified_maps = None

    def undistort_images(self, images, rectify=False):
        if rectify:
            if self.rectified is None:
                self.rectify()
            if self.rectified_maps is None:
                self.rectified_maps = undistort_and_rectify(self.cameras,
                                                            self.rectified,
                                                            self.image_shape)
            return remap_images(images, self.rectified_maps[:, None])
        elif self.maps is None:
            self.maps = undistort(self.cameras, self.image_shape)
        return remap_images(images, self.maps[:, None])


def stereo_from_images(calibration_object, images):
    calibration = stereo_calibration()
    images = np.asarray(images)
    calibration.image_shape = images.shape[-3:]
    p_tuple = calibration_object.find_in_images(images)
    points3D, calibration.points2D, calibration.image_mask = p_tuple
    calibration.points3D = points3D[0]  # Same for both cameras
    calibration.process()
    return calibration


def interpolate_affine(image, points, mask=None, window=10, tol=50):
    # Using local homography to interpolate affine image
    if isinstance(points, np.ndarray) and points.shape == (2,):
        window = np.mgrid[-window:window, -window:window].T.reshape(-1, 2)
        p = points
        max = np.array(image.shape[-3:-1]) - 1
        p_window = np.clip(p.astype(int) + window, 0, max).T
        i_window = image[(*p_window,)]
        m_window = np.squeeze(mask[(*p_window,)])
        if m_window.sum() < tol:
            return np.nan
        else:
            i_window = i_window[m_window][..., ::-1]
            p_window = p_window.T[m_window].astype(np.float32)[..., ::-1]
            H, _ = cv2.findHomography(p_window, i_window, cv2.LMEDS)
            p = H.dot(np.array([*p[[1, 0]], 1])).astype(np.float32)
            return p[[1, 0]] / p[2]
    elif isinstance(points[0], np.ndarray) and points[0].shape == (2,):
        return [interpolate_affine(image, p, mask, window, tol)
                for p in points]
    else:
        return [interpolate_affine(i, p, m, window, tol)
                for i, m, p in zip(image, mask, points)]


class projector_camera_calibration(stereo_calibration):
    def undistort_images(self, images, rectify=False):
        if rectify:
            if self.rectified is None:
                self.rectify()
            if self.rectified_maps is None:
                self.rectified_maps = undistort_and_rectify(self.cameras,
                                                            self.rectified,
                                                            self.image_shape)
            return remap_images(images, self.rectified_maps[0, None])
        elif self.maps is None:
            self.maps = undistort(self.cameras, self.image_shape)
        return remap_images(images, self.maps[0, None])


def stereo_from_image_and_LUT(calibration_object, images, LUT, mask=None):
        calib = projector_camera_calibration()
        images = np.asarray(images)
        calib.image_shape = images.shape[-3:]
        p_tuple = calibration_object.find_in_images(images)
        points3D, points2D, image_mask = p_tuple
        W = 10
        p_points2D = interpolate_affine(LUT, points2D, mask, W)
        calib.points3D = [[p3D for p3D, pp2D in zip(lp3D, lpp2D)
                           if not np.isnan(pp2D).any()]
                          for lp3D, lpp2D in zip(points3D, p_points2D)]
        _points2D = [[p2D for p2D, pp2D in zip(lp2D, lpp2D)
                      if not np.isnan(pp2D).any()]
                     for lp2D, lpp2D in zip(points2D, p_points2D)]
        p_points2D = [[pp2D for pp2D in lpp2D if not np.isnan(pp2D).any()]
                      for lpp2D in p_points2D]
        calib.points2D = [_points2D, p_points2D]
        calib.process()
        return calib
