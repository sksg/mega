import numpy as np
import cv2


class calibration:
    """calibration contains all the system values of the scanner set up.

    The includes:
    - Camera intrinsics and extrinsics
    - Correction procedures for scans made in this set up"""

    class camera:
        """camera contains intrinsic and extrinsic values."""

        def __init__(self):
            self.focal_vector = None
            self.intrinsic = None
            self.position = None
            self.rotation = None
            self.projection = None
            self.distortion = None
            self.frame_size = None

    def __init__(self, cvFS=None, frame_size=None):
        self.cam0, self.cam1 = calibration.camera(), calibration.camera()
        self.projector = None
        self.frame_size = None
        self.rectified_cam0 = calibration.camera()
        self.rectified_cam1 = calibration.camera()
        self.rectify_maps0 = None
        self.rectify_maps1 = None
        self.disparity_to_depth_map = None
        if cvFS is not None:
            self.read_cvFS(cvFS)
        if frame_size is not None:
            self.set_frame_size(frame_size)

    def read_cvFS(self, filename):
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

        self.cam0.intrinsic = fs.getNode("K0").mat().astype(np.float64)
        self.cam0.focal_vector = np.array((self.cam0.intrinsic[0, 0],
                                           self.cam0.intrinsic[1, 1]))
        self.cam0.distortion = fs.getNode("k0").mat().astype(np.float64)
        self.cam0.position = np.array([0., 0., 0.])  # assumed reference point
        self.cam0.rotation = np.eye(3)  # assumed reference
        self.cam0.projection = np.zeros((3, 4))
        self.cam0.projection[:3, :3] = self.cam0.intrinsic

        self.cam1.intrinsic = fs.getNode("K1").mat().astype(np.float64)
        self.cam1.focal_vector = np.array((self.cam1.intrinsic[0, 0],
                                           self.cam1.intrinsic[1, 1]))
        self.cam1.distortion = fs.getNode("k1").mat().astype(np.float64)
        self.cam1.position = - fs.getNode("T1").mat()[:, 0].astype(np.float64)
        self.cam1.rotation = fs.getNode("R1").mat().astype(np.float64)
        self.cam1.projection = np.empty((3, 4))
        self.cam1.projection[:3, :3] = self.cam1.intrinsic @ self.cam1.rotation
        self.cam1.projection[:3, 3] = self.cam1.intrinsic @ self.cam1.position

    def set_frame_size(self, frame_size):
        self.cam0.frame_size = frame_size
        self.cam1.frame_size = frame_size
        self.frame_size = frame_size
        rect = cv2.stereoRectify(self.cam0.intrinsic, self.cam0.distortion,
                                 self.cam1.intrinsic, self.cam1.distortion,
                                 R=self.cam1.rotation, T=-self.cam1.position,
                                 imageSize=frame_size, flags=0)
        self.rectified_cam0.rotation = rect[0]
        self.rectified_cam1.rotation = rect[1]
        self.rectified_cam0.position = self.cam0.position.copy()
        self.rectified_cam1.position = self.cam1.position.copy()
        self.rectified_cam0.intrinsic = self.cam0.intrinsic.copy()
        self.rectified_cam1.intrinsic = self.cam1.intrinsic.copy()
        self.rectified_cam0.focal_vector = self.cam0.focal_vector.copy()
        self.rectified_cam1.focal_vector = self.cam1.focal_vector.copy()
        self.rectified_cam0.projection = rect[2]
        self.rectified_cam1.projection = rect[3]
        self.disparity_to_depth_map = rect[4]

        M0 = cv2.initUndistortRectifyMap(self.cam0.intrinsic,
                                         self.cam0.distortion,
                                         self.rectified_cam0.rotation,
                                         self.rectified_cam0.projection,
                                         frame_size, cv2.CV_32F)

        M1 = cv2.initUndistortRectifyMap(self.cam1.intrinsic,
                                         self.cam1.distortion,
                                         self.rectified_cam1.rotation,
                                         self.rectified_cam1.projection,
                                         frame_size, cv2.CV_32F)
        self.rectify_maps = (M0, M1)

    def undistort_and_rectify(self, frames, **kwargs):
        interpolation = kwargs.pop("interpolation", cv2.INTER_LINEAR)
        channels = kwargs.pop("channels", None)
        shape = frames.shape
        # shape = (*array_shape, height, width[, channels])
        axes = (-1, -2)
        if channels != 1 or frames[-1] <= 4:  # We assume that width > 4
            axes = (-2, -3)
        frame_size = (shape[axes[0]], shape[axes[1]])

        if self.frame_size != frame_size:
            self.set_frame_size(frame_size)

        result = np.empty_like(frames)
        for i in range(frames.shape[1]):
            for j, (mx, my) in zip((0, 1), self.rectify_maps):
                result[j, i] = cv2.remap(frames[j, i], mx, my, interpolation)
        return result
