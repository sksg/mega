import os, sys, glob
import numpy as np
import cv2
from camera_calibration import StereoCalibration, CalibrationCheckerboard


calibration_path = r"D:\brdf-painter\calibration"
index_of_number_start = 35 # Magic number for my folder.
camera_0_image_paths = sorted(glob.glob(os.path.join(calibration_path, "frame0_*.png")), key=lambda x: int(x[35:-4]))
camera_1_image_paths = sorted(glob.glob(os.path.join(calibration_path, "frame1_*.png")), key=lambda x: int(x[35:-4]))


camera_0_images = []
camera_1_images = []

for index, image_path in enumerate(camera_0_image_paths):
    tmp_img = cv2.imread(image_path)
    camera_0_images.append(tmp_img)
    
for index, image_path in enumerate(camera_1_image_paths):
    tmp_img = cv2.imread(image_path)
    camera_1_images.append(tmp_img)
    
    
resize_scale = 4
calibration_object = CalibrationCheckerboard()
stereo_calibration = StereoCalibration(calibration_object, camera_0_images, camera_1_images, resize_scale=resize_scale)
