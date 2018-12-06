import numpy as np
import cv2
from .camera import camera as camera_class
from ._opencv import cv_shape
from ._opencv import cv_points
from ..parallize import parallize


p_kwargs = dict(default='sequential')


@parallize('(v,n,3),(v,n,2),_->(),(v)', **p_kwargs)
def calibrate(P3D, P2D, imshape):
    
    notnans_3D = ~np.isnan(P3D).any(axis=-1)
    notnans_2D = ~np.isnan(P2D).any(axis=-1)
    notnans = notnans_3D & notnans_2D
    P3D_corrected = cv_points(P3D, mask=notnans)
    P2D_corrected = cv_points(P2D, mask=notnans)

    reprojection_error, K, distortion, R_list, t_list = cv2.calibrateCamera(
        objectPoints=P3D_corrected,
        imagePoints=P2D_corrected,
        imageSize=cv_shape(imshape),
        cameraMatrix=None, distCoeffs=None,  # No inital guess!
        flags=(cv2.CALIB_FIX_K3 +
               cv2.CALIB_FIX_ASPECT_RATIO +
               cv2.CALIB_ZERO_TANGENT_DIST)
    )
    ghosts = np.full(P3D.shape[:-2], None)
    if reprojection_error is None:  # Unsuccessful
        return None, ghosts
    # OpenCV often represents vectors as 2D arrays:
    distortion = np.squeeze(distortion)
    t_list = np.squeeze(t_list)
    ghosts[notnans.any(axis=-1)] = np.array([camera_class(K, R, t, distortion, imshape)
                      for (R, t) in zip(R_list, t_list)], object)
    cam = camera_class(K, distortion=distortion,
                       imshape=cv_shape(imshape)[::-1])
    return cam, ghosts


@parallize('(v,n,3),(v,n,2),(v,n,2),_,(),()->(),(),(3,3),(3,3)', **p_kwargs)
def stereocalibrate(P3D, left2D, right2D, imshape=None, left=None, right=None):
    if left is None:
        left = calibrate(P3D, left2D, imshape)[0]
    if right is None:
        right = calibrate(P3D, right2D, imshape)[0]
    notnans = ~np.isnan([left2D, right2D]).any(axis=0).any(axis=-1)
    P3D = cv_points(P3D, mask=notnans)
    left2D = cv_points(left2D, mask=notnans)
    right2D = cv_points(right2D, mask=notnans)
    (reprojection_error,
     _, _, _, _,  # camera intrinsics, which are fixed already, and not needed
     R1, t1, E, F) = cv2.stereoCalibrate(
        objectPoints=np.array(P3D),
        imagePoints1=np.array(left2D),
        imagePoints2=np.array(right2D),
        imageSize=cv_shape(left.imshape),
        cameraMatrix1=left.K,
        distCoeffs1=left.distortion,
        cameraMatrix2=right.K,
        distCoeffs2=right.distortion,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    if reprojection_error is None:  # Unsuccessful
        return np.full(2, None), None, None
    left, right = left.copy(), right.copy()
    right.R, right.t = R1, np.squeeze(t1)
    return left, right, E, F


@parallize('(),()->(),(),(3,3),(3,3)', **p_kwargs)
def stereorectify(left, right):
    c0, c1 = left, right
    R, t = c1.relative_to(c0)
    R0, R1, P0, P1 = cv2.stereoRectify(cameraMatrix1=c0.K.astype('f8'),
                                       distCoeffs1=c0.distortion.astype('f8'),
                                       cameraMatrix2=c1.K.astype('f8'),
                                       distCoeffs2=c1.distortion.astype('f8'),
                                       imageSize=cv_shape(c0.imshape),
                                       R=R.astype('f8'),
                                       T=t.astype('f8'),
                                       flags=0)[:4]
    return (camera_class.from_P(P0, imshape=left.imshape),
            camera_class.from_P(P1, imshape=right.imshape), R0, R1)
