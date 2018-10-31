import os, glob
import cv2
from camera_calibration import CameraCalibration


calibration_path = r"D:\brdf-painter\calibration"
index_of_number_start = 35  # Magic number for my folder.
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
camera_0_cali = CameraCalibration(resize_scale=resize_scale)
camera_0_cali.calibrate_camera(images=camera_0_images)
camera_1_cali = CameraCalibration(resize_scale=resize_scale)
camera_1_cali.calibrate_camera(images=camera_1_images)
print(camera_0_cali.reprojection_error(), camera_1_cali.reprojection_error())


# Stereo calibration
# Temporary solution
# See https://docs.opencv.org/3.4/d9/d0c/group__calib3d.html#ga91018d80e2a93ade37539f01e6f07de5
sret, slcammat, sldist, srcammat, srdist, R, T, E, F = cv2.stereoCalibrate(camera_0_cali._object_points, camera_0_cali._image_points, camera_1_cali._image_points, camera_0_cali.K, camera_0_cali.distortion, camera_1_cali.K, camera_1_cali.distortion, camera_0_images[0].shape[:-1])