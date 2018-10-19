import mega

# Single camera case
images = mega.read_images("calibration_set")
reference = mega.calibrate.checkerboard((19, 23), 5, corse_rescale=0.5)
calibration = mega.calibrate.cameras_from_images(reference, images)
print(calibration.cameras[0])

mega.save_cameras("camera.npy", calibration.cameras)
camera = mega.load_cameras("camera.npy")
print(camera)
