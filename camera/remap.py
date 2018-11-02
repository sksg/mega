import numpy as np
import cv2


def _cv_shape(shape):
    """Open CV always works in column-row order. We do the opposite."""
    return (shape[1], shape[0]) if len(shape) < 3 else (shape[-2], shape[-3])


def _cv_pixels(points2D):
    """Open CV always works in column-row order. We do the opposite."""
    if isinstance(points2D, np.ndarray):
        return points2D[..., ::-1].astype(np.float32)
    else:
        return [_cv_pixels(p2D) for p2D in points2D]


def rescale(images, scale, scale_W=None):
    H, W, C = images.shape[-3:]
    if scale_W is None:
        scale_W = scale
    H, W = int(scale * H), int(scale_W * W)
    return_array = np.empty((*images.shape[:-3], H, W, C), images.dtype)
    for idx in np.ndindex(images.shape[:-3]):
        im = cv2.resize(images[idx], (W, H), interpolation=cv2.INTER_CUBIC)
        return_array[idx] = im.reshape(H, W, C)
    return return_array


def remap_single_image(image, maps):
    # Alternative: use scipy and interp1d's.
    return cv2.remap(image, *maps, cv2.INTER_LINEAR)


def remap_images(images, maps):
    images = np.asanyarray(images)
    maps = np.broadcast_to(maps, images.shape[:-3] + maps.shape[-3:])
    maps = maps.reshape((-1, *maps.shape[-3:]))
    _images = images.reshape((-1, *images.shape[-3:]))
    remapped = np.array(list(map(remap_single_image, _images, maps)))
    shape = images.shape[:-3] + maps.shape[-2:] + images.shape[-1:]
    return remapped.reshape(shape)


def rectified_stereo_disparity_to_depth(cameras):
    f = cameras[0].focal_vector.mean()
    t = cameras[1].t[0] / f
    c0x, c0y = cameras[0].K[0, 2], cameras[0].K[1, 2]
    c1x, c1y = cameras[1].K[0, 2], cameras[1].K[1, 2]
    return np.array([[1, 0, 0, -c0x],
                     [0, 1, 0, -c0y],
                     [0, 0, 0, f],
                     [0, 0, -1 / t, (c0x - c1x) / t]])


def single_undistort(camera, image_shape):
    return cv2.initUndistortRectifyMap(camera.K, camera.distortion,
                                       np.eye(3, dtype=np.float32),
                                       camera.K, _cv_shape(image_shape),
                                       cv2.CV_32F)


def undistort_points(camera, points2D):
    shape = points2D.shape
    p2D = _cv_pixels(points2D.reshape((1, -1, *shape[-1:])))
    p2D = cv2.undistortPoints(p2D, camera.K, camera.distortion, P=camera.K)
    return _cv_pixels(p2D).reshape(shape)


def single_undistort_and_rectify(camera, new_camera, image_shape):
    return cv2.initUndistortRectifyMap(camera.K, camera.distortion,
                                       new_camera.R, new_camera.P,
                                       _cv_shape(image_shape), cv2.CV_32F)


def undistort(cameras, image_shape):
    def fn(x):
        return single_undistort(x, image_shape)
    cameras = np.asanyarray(cameras)
    undistorted = np.array(list(map(fn, cameras.flatten())))
    return undistorted.reshape(cameras.shape + undistorted.shape[1:])


def undistort_and_rectify(cameras, new_cameras, image_shape):
    def fn(x, y):
        return single_undistort_and_rectify(x, y, image_shape)
    cameras, new_cameras = np.broadcast_arrays(cameras, new_cameras)
    undistorted = np.array(list(map(fn, cameras.flatten(),
                                    new_cameras.flatten())))
    return undistorted.reshape(cameras.shape + undistorted.shape[1:])
