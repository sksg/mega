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
    def __init__(self, K=None, R=None, t=None, distortion=None, flat=None,
                 warn_on_ambiguity=True):
        if flat is not None:
            if warn_on_ambiguity and any(v is not None for v in (K, R, t)):
                print("Camera warning: K, R, t take precedence over flat.")
            else:
                K = flat[:9].reshape((3, 3))
                R = flat[9:18].reshape((3, 3))
                t = flat[18:21]
                distortion = flat[21:]
        self.K = K
        self.R = R
        self.t = t
        self.distortion = distortion

    @property
    def position(self):
        """position of the camera in *world space*. Calculated at calltime."""
        return - self.R.T.dot(self.t)

    @property
    def P(self):
        """Projection matrix, (3, 4) array. Calculated at calltime."""
        if all(v is not None for v in (self.K, self.R, self.t)):
            return self.K.dot(np.c_[self.R, self.t])

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

    def project(self, points3D):
        points3D = np.asanyarray(points3D)
        points2D = np.empty(points3D.shape, points3D.dtype)
        if points3D.dtype != object:  # Standard numeric array
            args = self.R, self.t, self.K, self.distortion
            p3D = points3D.reshape(-1, 3)
            p2D, _ = cv2.projectPoints(p3D, *args)
            points2D.reshape(-1, 2)[:] = p2D
        else:
            for idx in np.ndindex(points3D.shape):
                p3D = points3D[idx]
                if p3D is None:
                    points2D[idx] = None
                else:  # Must be an array
                    points2D[idx] = self.project(p3D)
        return points2D

    def __repr__(self):
        def arr2str(s, A):
            return s + np.array2string(A, precision=2, separator=',',
                                       suppress_small=True,
                                       prefix=s.strip('\n'))
        return (arr2str(''"camera{K: ", self.K) + "," +
                arr2str("\n       R: ", self.R) + "," +
                arr2str("\n       t: ", self.t) + "," +
                arr2str("\n       distortion: ", self.distortion) + "}")


def project_in_cameras(points3D, cameras):
    points3D, cameras = np.asanyarray(points3D), np.asanyarray(cameras)
    points2D = np.empty(cameras.shape, object)
    points2D.fill(None)
    is_trivial_array = True
    if points3D.dtype != object:  # Standard numeric array
        # Broadcast arrays (nontrivial as the shapes are not equal)
        if len(points3D.shape[:-1]) > len(cameras.shape):
            cameras = np.broadcast_to(cameras, points3D.shape[:-1])
        elif len(points3D.shape[:-1]) < len(cameras.shape):
            points3D = np.broadcast_to(points3D, cameras.shape + (3,))
    else:
        # We must assume that the shapes can match.
        points3D, cameras = np.broadcast_arrays(points3D, cameras)
        is_trivial_array = False
    for idx in np.ndindex(cameras.shape):
        p3D, cam = points3D[idx], cameras[idx]
        if cam is None or p3D is None:
            points2D[idx] = None
            is_trivial_array = False
        elif isinstance(cam, camera):
            points2D[idx] = cam.project(p3D)
        else:  # must be an array of cameras
            points2D[idx] = project_cameras(p3D, cam)
            is_trivial_array = False
    if is_trivial_array:
        points2D = points2D.astype(points3D.dtype)
    return points2D


def save_cameras(filename, cameras):
    """save cameras as a nD numpy file."""
    __MAX_CAMERA_FLAT__ = (30,)
    cameras = np.asanyarray(cameras)
    result = np.empty(cameras.shape + __MAX_CAMERA_FLAT__, object)
    for idx in np.ndindex(cameras.shape):
        if cameras[idx] is not None:
            flat_cam = cameras[idx].flatten()
            if result.dtype == object:
                result = result[..., :len(flat_cam)].astype(flat_cam.dtype)
            result[idx] = flat_cam
        else:
            result[idx] = np.nan
    np.save(filename, result)


def load_cameras(filename):
    """load cameras from nD numpy file using the .npy extension."""
    flat_cameras = np.load(filename)
    cameras = np.empty(flat_cameras.shape[:-1], object)
    for idx in np.ndindex(cameras.shape):
        cameras[idx] = camera(flat=flat_cameras[idx])
    return cameras
