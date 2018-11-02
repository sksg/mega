import numpy as np


class camera:
    """camera wrapper for K, R, t, and distortion arrays.

    # Parameters:
    K : (3, 3) arraylike, optional
        The intrinsic camera matrix. Stored under self.K. Do not use with flat.
    R : (3, 3) arraylike, optional
        The camera rotation matrix. Stored under self.R. Do not use with flat.
    t : (3,) arraylike, optional
        The camera translation vector. Stored under self.t. Do not use with
        flat.
    distortion : (N,) arraylike, optional
        The camera distortion vector. Can be arbitrarily long. Stored under
        self.distortion. Do not use with flat.
    P : (3, 4) arraylike, optional
        The camera projection vector. Used to estimate K and t. Do not use with
        explicit K or t.
    flat : (21 + N,) arraylike, optional
        The camera flat vector. Must be at least 21 elements long. Views of
        flat are stored under self.K, self.R, self.t, and self.distortion.
        The flat vector it self is not stored.
        Do no use flat with any of K, R, t, or distortion---this would be
        ambigious!
    warn_on_ambiguity : bool, optional
        True (default) prints a warning if using both the flat and any of the
        K, R, t, and distortion keywords at the same time. False ignores the
        ambiguity, and prints no warning.

    # Notes:
    The class does not attempt to copy any array data when initiated.
    Use the member copy() to force copy all internal data structures.
    """
    def __init__(self, K=None, R=None, t=None, distortion=None, P=None,
                 flat=None, warn_on_ambiguity=True):
        if flat is not None:
            if warn_on_ambiguity and any(v is not None for v in (K, R, t, P)):
                print("Camera warning: K, R, t take precedence over flat.")
            else:
                K = flat[:9].reshape((3, 3))
                R = flat[9:18].reshape((3, 3))
                t = flat[18:21]
                distortion = flat[21:]
        if P is not None and R is not None:
            if warn_on_ambiguity and any(v is not None for v in (K, t)):
                print("Camera warning: K, R, t take precedence over R, P.")
            else:
                K = P[:3, :3].dot(R.T)
                t = np.linalg.inv(K).dot(P[:3, 3])
        if K is None:
            raise RuntimeError("Camera Error: Missing camera matrix K.")
        if R is None:
            R = np.eye(3, dtype=K.dtype)
        if t is None:
            t = np.zeros((3,), dtype=K.dtype)
        if distortion is None:
            distortion = np.array(tuple())
        self._K = K.astype(np.float32)
        self._R = R.astype(np.float32)
        self._t = t.astype(np.float32)
        self.distortion = distortion.astype(np.float32)
        self._P_cache = None

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, K):
        self._K = K
        self._P_cache = None

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = R
        self._P_cache = None

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        self._t = t
        self._P_cache = None

    @property
    def position(self):
        """position of the camera in *world space*. Calculated at calltime."""
        return - self.R.T.dot(self.t)

    @property
    def P(self):
        """Projection matrix, (3, 4) array. Calculated at calltime."""
        if self._P_cache is None:
            if all(v is not None for v in (self.K, self.R, self.t)):
                    self._P_cache = self.K.dot(np.c_[self.R, self.t])
        return self._P_cache

    @property
    def focal_vector(self):
        """Focal vector, (2,) view into array self.K."""
        return np.diag(self.K)[:2]

    def copy(self):
        """Copies all data into returned new camera instance."""
        return camera(flat=self.flatten())

    def flatten(self):
        """1D camera vector, (21 + len(self.distortion),) array. Calculated at
        calltime."""
        return np.r_[self.K.flatten(), self.R.flatten(),
                     self.t, self.distortion]

    def relative_to(self, other):
        R = self.R.dot(other.R.T)
        t = R.dot(other.position - self.position)
        return R, t

    def project(self, points3D):
        if isinstance(points3D[0], np.ndarray) and points3D[0].shape == (3,):
            args = self.R, self.t, self.K, self.distortion
            return cv2.projectPoints(points3D, *args)[0][..., ::-1]
        return [self.project(p3D) for p3D in points3D]

    def __repr__(self):
        def arr2str(s, A):
            return s + np.array2string(A, precision=2, separator=',',
                                       suppress_small=True,
                                       prefix=s.strip('\n'))
        return (arr2str(''"camera{K: ", self.K) + "," +
                arr2str("\n       R: ", self.R) + "," +
                arr2str("\n       t: ", self.t) + "," +
                arr2str("\n       distortion: ", self.distortion) + "}")


def camera_project(points3D, cameras):
    points3D, cameras = np.asanyarray(points3D), np.asanyarray(cameras)
    shape = cameras.shape
    points2D = np.empty(shape, object)
    for idx in np.ndindex(shape):
        points2D[idx] = cameras[idx].project(points3D[idx])
    return points2D


def save_cameras(filename, cameras):
    """save cameras as a nD numpy file."""
    __MAX_CAMERA_FLAT__ = (30,)
    cameras = np.asanyarray(cameras)
    result = np.empty(cameras.shape + __MAX_CAMERA_FLAT__, object)
    for idx in np.ndindex(cameras.shape):
        flat_cam = cameras[idx].flatten()
        if result.dtype == object:
            result = result[..., :len(flat_cam)].astype(flat_cam.dtype)
        result[idx] = flat_cam
    np.save(filename, result)


def load_cameras(filename):
    """load cameras from nD numpy file using the .npy extension."""
    flat_cameras = np.load(filename)
    cameras = np.empty(flat_cameras.shape[:-1], object)
    for idx in np.ndindex(cameras.shape):
        cameras[idx] = camera(flat=flat_cameras[idx])
    return cameras
