import numpy as np
import cv2
from functools import lru_cache


class camera:
    def __init__(self, K, R=None, t=None, distortion=None, imshape=(0, 0)):
        self.K = K
        self.R = R if R is not None else np.eye(3, dtype=K.dtype)
        self.t = t if t is not None else np.zeros((3,), K.dtype)
        D = distortion
        self.distortion = D if D is not None else np.array(tuple(), K.dtype)
        IS = imshape
        self.imshape = IS if IS is not None else np.array((0, 0), int)

    @staticmethod
    def from_flat(flat):
        K = flat[:9].reshape((3, 3))
        R = flat[9:18].reshape((3, 3))
        t = flat[18:21]
        imshape = np.rint(flat[21:23]).astype(int)
        distortion = flat[23:]
        return camera(K, R, t, distortion, imshape)

    @staticmethod
    def from_P(P, R=None, distortion=None, imshape=None):
        R = R if R is not None else np.eye(3, dtype=P.dtype)
        K = P[:3, :3].dot(R.T)
        t = np.linalg.inv(K).dot(P[:3, 3])
        return camera(K, R, t, distortion, imshape)

    @property
    @lru_cache(maxsize=1)  # Now this result is fixed!!
    def position(self):
        """position of the camera in *world space*."""
        return - self.R.T.dot(self.t)

    @property
    @lru_cache(maxsize=1)  # Now this result is fixed!!
    def P(self):
        """Projection matrix, (3, 4) array."""
        return self.K.dot(np.c_[self.R, self.t])

    @property
    def focal_vector(self):
        """Focal vector, (2,) view into array self.K."""
        return np.diag(self.K)[:2]

    def copy(self):
        """Copies all data into returned new camera instance."""
        return camera.from_flat(self.flatten())

    def flatten(self):
        """1D camera vector calculated at calltime."""
        return np.r_[self.K.ravel(), self.R.ravel(), self.t,
                     np.array(self.imshape, self.K.dtype),
                     self.distortion]

    def relative_to(self, other):
        R = self.R.dot(other.R.T)
        t = R.dot(other.position - self.position)
        return R, t

    def __repr__(self):
        def arr2str(s, A):
            return s + np.array2string(A, precision=2, separator=',',
                                       suppress_small=True,
                                       prefix=s.strip('\n'))
        return (arr2str(''"camera{    K: ", self.K) + "," +
                arr2str("\n           R: ", self.R) + "," +
                arr2str("\n           t: ", self.t) + "," +
                arr2str("\n imshape: ", self.imshape) + "," +
                arr2str("\n  distortion: ", self.distortion) + "}")


def save_cameras(filename, cameras, max_flat=None, dtype=None):
    """save cameras as a nD numpy file."""
    cameras = np.asanyarray(cameras)
    if max_flat is None:  # in this case, we assume all flat are same length
        result = None
        max_flat = 50  # Just make it large enough
    else:  # here we can assume different lengths. Will cut if larger!
        result = np.empty(cameras.shape + (max_flat,), object)
    for idx in np.ndindex(cameras.shape):
        flat_cam = cameras[idx].flatten()
        if result is None and dtype is None:
            result = np.empty(cameras.shape + flat_cam.shape, flat_cam.dtype)
        elif result is None and dtype is not None:
            result = np.empty(cameras.shape + flat_cam.shape, dtype)
        elif result.dtype == object and dtype is None:
            result = result.astype(flat_cam.dtype)
        elif result.dtype == object and dtype is None:
            result = result.astype(dtype)
        result[idx] = flat_cam[:max_flat]
    np.save(filename, result)


def load_cameras(filename):
    """load cameras from nD numpy file using the .npy extension."""
    flat_cameras = np.load(filename)
    cameras = np.empty(flat_cameras.shape[:-1], object)
    for idx in np.ndindex(cameras.shape):
        cameras[idx] = camera.from_flat(flat_cameras[idx])
    return cameras
