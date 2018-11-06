import mega
import mega.structuredlight as sl
import numpy as np
import os


path = "scans/20180913-subsurfacescattering-PTFE+Nylon66/"
sequence_path = path + "sequence_0"
calibration_path = path + "calibration"
ply_file = "points.ply"

N = 16  # primary phaseshift step count
M = 8  # cue phaseshift step count
wn = 40.0  # phaseshift wave-number in inverse projector pixels


def frmsort(filenames):
    names = [os.path.split(f)[1] for f in filenames]
    indices = [int(f.split('_')[1].split('.')[0]) for f in names]
    return [filenames[i] for i in np.argsort(indices)]

frexp = ['frame0_[0-9]*.png', 'frame1_[0-9]*.png']

print("Processing calibration sequence")
frames = np.stack([mega.read_images(calibration_path, frexp[0], frmsort),
                   mega.read_images(calibration_path, frexp[1], frmsort)])
reference = mega.checkerboard((19, 12), 10, corse_rescale=0.5)
points3D, pixels, _ = reference.find_in_images(frames)
cameras, _, _ = mega.calibrate.stereo(points3D[0], pixels, (frames.shape,
                                                            frames.shape))
rectified, Rs = mega.calibrate.rectify_stereo(cameras, frames.shape)
maps = mega.undistort_and_rectify(cameras, Rs, rectified, frames.shape)


def split_array(array, split):
    split = (0,) + tuple(np.cumsum(split))
    return [array[:, s0:s1] if s1 - s0 > 1 else array[:, s0]
            for s0, s1 in zip(split[:-1], split[1:])]


frexp = ['frames0_[0-9]*.png', 'frames1_[0-9]*.png']
print("Processing capture sequence")
frames = np.stack([mega.read_images(sequence_path, frexp[0], frmsort),
                   mega.read_images(sequence_path, frexp[1], frmsort)])
frames = mega.remap_images(frames, maps[:, None])

color_frames = frames[:, 0]
grayscale = mega.rgb2gray(frames.astype(np.float32)) / 255
gray, dark, P, C = split_array(grayscale, (1, 1, N, M))
ph, dph, mask = sl.phaseshift.decode_with_cue(gray, dark, P, C, wn)

pixels = sl.match_epipolar_maps(ph, mask)

if len(pixels) == 0:
    print("No points reconstructed!")
    exit()

colors = color_frames[(0, *pixels[0].astype(int).T,)]
points = sl.triangulate(*rectified, pixels, image_shape=color_frames.shape)
normals, dev = sl.normals_from_gradients(*rectified, points, *pixels, *dph)
# Alternatives:
# normals, dev = sl.normals_from_depth(rectified[0], points, pixels[0],
#                                      color_frames.shape)
# normals, dev = sl.normals_from_SVD(rectified[0], points, pixels[0],
#                                    color_frames.shape)

# Filter points with too large normal deviation
idx = (np.abs(dev) > 0.3).all(axis=0)
colors, points = colors[idx], points[idx]
normals, _normals = normals[idx], _normals[idx]
print(np.linalg.norm(normals - _normals, axis=1))
ply = mega.pointcloud(points, colors, _normals)
ply.writePLY(ply_file)
print("Pointcloud exported to {}".format(ply_file))
