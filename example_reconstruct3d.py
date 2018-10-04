import mega
import mega.structuredlight as sl


path = "scans/20180913-subsurfacescattering-PTFE+Nylon66/"
calibration = path + "calibration.xml"
ply_file = "points.ply"


# Requires a special format for calibration file.
# TODO(sorgre): Need to some how generalize sl.calibration!
calib = sl.calibration(calibration)
raw_frames = mega.read_frame_sequence("path/to/frame_sequence/directory")
frames = calib.undistort_and_rectify(raw_frames)
# frames.shape = (cam_count == 2, frame_count, height, width, channels == 3)

primary_frame_count = 16
cue_frame_count = 8
wave_count = 40.0

lit = frames[:, 0]
dark = frames[:, 1]
primary = frames[:, 2:2 + primary_frame_count]
cue = frames[:, 2 + primary_frame_count:]
ply = sl.phaseshift.reconstruct(calib, lit, dark, primary, cue, wave_count)
ply.writePLY(ply_file)
print("Pointcloud exported to {}".format(ply_file))
