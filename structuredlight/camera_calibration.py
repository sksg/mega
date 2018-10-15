import numpy as np
import cv2


class Camera:
    """
        Camera class.
        K = Intrinsic
        R = Rotation
        t = translation
        P = K[Rt]
        
        Camera position in world space is -R't
        
    """
    
    def PfromKRt(K, R, t):
        return np.array(np.mat(K) * np.mat(np.concatenate((R, t), axis=1)))
    
    def __init__(self, K=None, R=None, t=None):
        self._K = K
        self._R = R
        self._t = t
        self._P = PfromKRt(K, R, t)
        
    
    @property
    def K(self):
        return self._K
    
    @K.setter
    def K(self, K):
        self._K = K
    
    @property
    def R(self):
        return self._R
    
    @R.setter
    def K(self, R):
        self._R = R
    
    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, t):
        self._t = t
        
    @property
    def P(self):
        return self._P
    
    @P.setter
    def P(self, P):
        self._P = P
    
    

class CalibrationCheckerboard:
    """
        Class for calibration object - Checkerboard
    """
    
    def __init__(self, n_corners = (22, 13), checker_size_mm = 15):
        self.world_points_3D = self.create_real_world_grid(n_corners, checker_size_mm)
        self.n_corners = n_corners
        self.checker_size_mm = checker_size_mm
        self._term_criteria = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001 )
        self._check_criteria = ( cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS + cv2.CALIB_CB_FAST_CHECK )
        

    def create_real_world_grid(self, n_corners, checker_size_mm):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # Create array of zeros. 
        objp = np.zeros((n_corners[0] * n_corners[1], 3), np.float32)

        # Create the inner grid
        objp[:, :2] = np.mgrid[0:n_corners[0], 0:n_corners[1]].T.reshape(-1, 2)

        # Multiply with checker size
        objp = objp*checker_size_mm
        return objp
    
    
    def find_corners(self, image, resize_scale = None):
        """
            This function populates self._object_points and self._image_points.
            Image must be grayscale and 2D e.g. shape = [256, 256]
            
        """
     
        height, width = image.shape
        object_points_2D = None
        object_points_3D = None
        
        if resize_scale is not None:
            corner_image = cv2.resize(image, (int(width / resize_scale), int(height / resize_scale)), interpolation=cv2.INTER_CUBIC)
        else:
            corner_image = image
            
        ret, corners = cv2.findChessboardCorners(
            corner_image, self.n_corners, self._check_criteria)
        
        if ret == True:
            object_points_3D = self.world_points_3D.copy()
            
            if resize_scale is not None:
                corners = corners * resize_scale
                
            cv2.cornerSubPix(image, corners, (11, 11), (-1, -1), self._term_criteria)
            
            object_points_2D = corners.copy()
        
        return ret, object_points_2D, object_points_3D

    
    
class CameraCalibration:
    """
    Camera Calibration class
    """
    
    def __init__(self, calibration_object = CalibrationCheckerboard(), resize_scale = None):
        self.calibration_object = calibration_object
        self._object_points = []
        self._image_points = []
        self._resize_scale = resize_scale
        self.K = None
        self.distortion = []
        self.R = []
        self.t = []
        
        
    def calibrate_camera(self, images):
        for index, image in enumerate(images):
            
            shape = image.shape
            
            if len(shape) > 2:
                # case is RGB
                corrected_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                corrected_image = image
            
            # Find checkerboard corners in image
            ret, object_points_2D, object_points_3D = self.calibration_object.find_corners(corrected_image)
            
            self._image_points.append(object_points_2D)
            self._object_points.append(object_points_3D)
            
            
            if ret is False:
                print("Warning: Cannot find checkerboard in image {}".format(index))
               
        # Calibrate the camera given all images
        ret, K, dist, R_per_image, t_per_image = cv2.calibrateCamera(
            self._object_points, self._image_points, corrected_image.shape[::-1], None, None)
                
        self.K = K
        self.distortion = dist
        self.R = R_per_image
        self.t = t_per_image
        
        return (K, dist, R_per_image, t_per_image)
              
        
    def reprojection_error(self, printout = False):
        
        error = []
        
        for i in range(len(self._object_points)):
            reproj_image_points, _ = cv2.projectPoints(
                self._object_points[i], self.R[i], self.t[i], self.K, self.distortion)
            tmp_error = cv2.norm(self._image_points[i], reproj_image_points, cv2.NORM_L2) / len(reproj_image_points)
            error.append(tmp_error)
        
        min_error = np.min(error)
        mean_error = np.mean(error)
        max_error = np.max(error)
        
        if printout is True:
            print("Error - Min: {}, Mean: {}, Max: {}".format(min_error, mean_error, max_error))
            
        return {"min": min_error, "mean": mean_error, "max": max_error}

    
    def dump_data(self, path):
        print()
        

class StereoCalibration:
    """
        Stereo calibration class
    """
    
    def __init__(self):
        