import mega
import mega.structuredlight as sl
import numpy as np
import os


sequence_path = path + "sequence"
calibration_path = path + "calibration/sequence_{}"
N_calibration_sequences = 20
ply_file = "points.ply"

N = 16  # primary phaseshift step count
M = 8  # cue phaseshift step count
wn = 40.0  # phaseshift wave-number in inverse projector pixels
prj_shape = (1080, 1920)  # Projector resolution (height, width)


def split_array(array, split):
    split = (0,) + tuple(np.cumsum(split))
    return [array[s0:s1] if s1 - s0 > 1 else array[s0]  # avoid dimension == 1
            for s0, s1 in zip(split[:-1], split[1:])]

_gray, _prj_map, _mask = [], [], []
for s in range(N_calibration_sequences):
    print("Processing calibration sequence {}".format(s))
    path = calibration_path.format(s)
    frames = mega.read_images(path)
    grayscale = mega.rgb2gray(frames).astype(np.float32) / 255
    G, D, xP, xC, yP, yC = split_array(grayscale, (1, 1, N, M, N, M))
    ph, _, mask = sl.phaseshift.decode2D_with_cue(G, D, yP, yC, xP, xC, wn)
    # From phases to projector pixels
    prj_map = ph / (2 * np.pi) * np.array([[prj_shape]])
    _gray.append(G)
    _prj_map.append(prj_map)
    _mask.append(mask)
gray = (np.array(_gray) * 255).astype('u1')
prj_map = np.array(_prj_map)
mask = np.array(_mask)

reference = mega.checkerboard((19, 12), 10, corse_rescale=0.5)
# Alternative verison---not possible yet:
# p3D, cam_p2D, prj_p2D, mask = reference.find_mapped_in_images(gray, prj_map)
# Will replace below:
_points3D, _cam_pixels, _ = reference.find_in_images(gray)
_prj_pixels = mega.calibrate.interpolate_affine(prj_map, _cam_pixels, mask)
points3D, cam_pixels, prj_pixels = [], [], []
for _p3D, _c_pix, _p_pix in zip(_points3D, _cam_pixels, _prj_pixels):
    _p3D = [p3D for p3D, pix in zip(_p3D, _p_pix)
            if not np.isnan(pix).all()]
    _c_pix = [cpix for cpix, pix in zip(_c_pix, _p_pix)
              if not np.isnan(pix).all()]
    _p_pix = [pix for pix in _p_pix if not np.isnan(pix).all()]
    points3D.append(np.array(_p3D))
    cam_pixels.append(np.array(_c_pix))
    prj_pixels.append(np.array(_p_pix))
# Will replace until here (see comment above)
pixels = (cam_pixels, prj_pixels)
shapes = (gray.shape, prj_shape)
(camera, projector), _, _ = mega.calibrate.stereo(points3D, pixels, shapes)
c_undistort_map = mega.undistort(camera, gray.shape)


print("Processing capture sequence")
frames = mega.read_images(sequence_path, frame_rexp, frame_sorting)
frames = mega.remap_images(frames, c_undistort_map)

color_frame = frames[0]
grayscale = mega.rgb2gray(frames.astype(np.float32)) / 255
gray, dark, xP, xC, yP, yC = split_array(grayscale, (1, 1, N, M, N, M))
ph, dph, mask = sl.phaseshift.decode2D_with_cue(gray, dark, yP, yC, xP, xC, wn)
# From phases to projector pixels
prj_map = ph / (2 * np.pi) * np.array([[prj_shape]])

# Select from mask
cam_pixels = np.stack(np.where(mask)[:2], axis=0).T
prj_pixels = mega.undistort_points(projector, prj_map[(*cam_pixels.T,)])

# The undistorted projector gradient
prj_gradient = np.array([2 * np.pi / prj_shape[1], 0.])[None, None, :]
prj_gradient = np.broadcast_to(prj_gradient, prj_shape + (2,))
cam_gradient = dph[..., 1, ::-1]  # row, col --> x, y
cam_gradient = ndimage.gaussian_filter(cam_gradient, sigma=(3, 3, 0), order=0)

colors = color_frame[(*cam_pixels.T,)]
points = sl.triangulate(camera, projector, (cam_pixels, prj_pixels),
                        image_shape=gray.shape)
normals, dev = sl.normals_from_gradients(camera, projector, points,
                                         cam_pixels, prj_pixels,
                                         cam_gradient, prj_gradient)

# Filter points with too large normal deviation
idx = (np.abs(dev) > 0.3).all(axis=0)
colors, points, normals = colors[idx], points[idx], normals[idx]
ply = mega.pointcloud(points, colors, normals)
ply.writePLY(ply_file)
print("Pointcloud exported to {}".format(ply_file))
