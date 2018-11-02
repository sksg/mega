import mega
import mega.structuredlight as sl
import numpy as np
import os


sequence_path = path + "sequence"
calibration_path = path + "calibration"
ply_file = "points.ply"

N = 16  # primary phaseshift step count
M = 8  # cue phaseshift step count
wn = 40.0  # phaseshift wave-number in inverse projector pixels

print("Processing calibration sequence")
frames = np.array([mega.read_images(calibration_path, 'frame0_[0-9]*.png'),
                   mega.read_images(calibration_path, 'frame1_[0-9]*.png')])

reference = mega.checkerboard((19, 12), 10, corse_rescale=0.5)
points3D, pixels, _ = reference.find_in_images(frames)
cameras, _, _ = mega.calibrate.stereo(points3D[0], pixels, shapes)
rectified = mega.calibrate.rectify_stereo(cameras, gray0.shape)
maps = mega.undistort_and_rectify(cameras, rectified, gray.shape)


def split_array(array, split):
    split = (0,) + tuple(np.cumsum(split))
    return [array[:, s0:s1] if s1 - s0 > 1 else array[:, s0]
            for s0, s1 in zip(split[:-1], split[1:])]


print("Processing capture sequence")
frames = [mega.read_images(sequence_path, 'frame0_[0-9]*.png'),
          mega.read_images(sequence_path, 'frame1_[0-9]*.png')]
frames = mega.remap_images(frames, maps)

color_frames = frames[:, 0]
grayscale = mega.rgb2gray(frames.astype(np.float32)) / 255
gray, dark, P, C = split_array(grayscale, (1, 1, N, M))
ph, dph, mask = sl.phaseshift.decode_with_cue(gray, dark, P, C, wn)
pixels = mega.match_epipolar_maps(ph, mask)

# The undistorted projector gradient
prj_gradient = np.array([2 * np.pi / prj_shape[1], 0.])[None, None, :]
prj_gradient = np.broadcast_to(prj_gradient, prj_shape + (2,))
cam_gradient = dph[..., 1, ::-1]  # row, col --> x, y
cam_gradient = ndimage.gaussian_filter(cam_gradient, sigma=(3, 3, 0), order=0)

colors = color_frame[(*cam_pixels.T,)]
points = sl.triangulate(camera, projector, (cam_pixels, prj_pixels),
                        image_shape=gray.shape)
normals, dev = sl.normals_from_gradients(*cameras, *pixels, *dph)

# Filter points with too large normal deviation
idx = (np.abs(dev) > 0.3).all(axis=0)
colors, points, normals = colors[idx], points[idx], normals[idx]
ply = mega.pointcloud(points, colors, normals)
ply.writePLY(ply_file)
print("Pointcloud exported to {}".format(ply_file))
