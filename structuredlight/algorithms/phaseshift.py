import numpy as np
from scipy import ndimage
import os
from concurrent.futures import ThreadPoolExecutor as thread_pool
import mega
from ... import structuredlight as sl
from scipy.ndimage import map_coordinates


def decode_sequential(frames):
    dft = np.fft.fft(frames, axis=1)
    phase = 2.0 * np.pi - np.angle(dft[:, 1])
    amplitude = np.absolute(dft[:, 1])
    energy = np.absolute(dft[:, 1:]).sum(axis=1)
    return phase, amplitude, energy


def decode_parallel(frames, step=1):
    phase = np.empty((2, *frames.shape[2:]), dtype=frames.dtype)
    amplitude = np.empty_like(phase)
    energy = np.empty_like(phase)
    pool = thread_pool()
    futures = []
    for i in range(frames.shape[2] // step):
        j = slice(i * step, (i + 1) * step)
        futures.append(pool.submit(decode_sequential, frames[:, :, j]))
    for i in range(frames.shape[2] // step):
        j = slice(i * step, (i + 1) * step)
        phase[:, j], amplitude[:, j], energy[:, j] = futures[i].result()
    return phase, amplitude, energy


def decode(frames, step=1):
    if os.environ.get('MEGA_PARALLELIZE', default=False):
        return decode_parallel(frames, step)
    else:
        return decode_sequential(frames)


def unwrap_with_cue(primary, cue, wave_count):
    phase_cue = np.mod(cue - primary, 2 * np.pi)
    P = np.round(((phase_cue * wave_count) - primary) / (2 * np.pi))
    return (primary + (2 * np.pi * P)) / wave_count


def decode_with_cue(primary, cue, wave_count):
    primary = decode(primary)  # Heavily parallized, great speed-up
    cue = decode(cue)  # Heavily parallized, great speed-up
    primary_phase = primary[0]
    cue_phase = cue[0]
    # unwrap_with_cue(): Not parallized, very low speed-up
    phase = unwrap_with_cue(primary_phase, cue_phase, wave_count)
    dphase = np.stack(np.gradient(phase, axis=(-2, -1)), axis=-1)
    return phase, dphase, primary, cue


def stdmask(gray, background, phase, dphase, primary, cue, wave_count):
    # Threshold on saturation and under exposure
    adjusted = gray - background

    def print_masked(path, mask):
        frame = adjusted.copy()
        frame[mask] += 0.5
        frame = np.clip(frame, 0, 1) * 255
        mega.write_frame(path, frame)

    if adjusted.dtype == np.dtype('u1'):
        mask = np.logical_and(adjusted > 255 * 0.1, adjusted < 255 * 0.95)
    else:
        # Assume HDR acquisition, and that only valid values are present
        mask = np.ones_like(adjusted, dtype=bool)
    # Threshold on amplitude at primary frequency
    mask = np.logical_and(mask, primary[1] > 2.0 * wave_count)
    print_masked("mask01_0.png", mask[0])
    print_masked("mask01_1.png", mask[1])
    # Threshold on amplitudes versus total energy
    mask = np.logical_and(mask, primary[1] > 0.25 * primary[2])
    print_masked("mask02_0.png", mask[0])
    print_masked("mask02_1.png", mask[1])
    mask = np.logical_and(mask, cue[1] > 0.25 * cue[2])
    print_masked("mask03_0.png", mask[0])
    print_masked("mask03_1.png", mask[1])
    # Threshold on gradient of phase. Cannot be too large or too small
    mask = np.logical_and(mask, np.linalg.norm(dphase, axis=3) < 0.02)
    print_masked("mask04_0.png", mask[0])
    print_masked("mask04_1.png", mask[1])
    mask = np.logical_and(mask, np.linalg.norm(dphase, axis=3) > 1e-8)
    print_masked("mask05_0.png", mask[0])
    print_masked("mask05_1.png", mask[1])
    # Cannot must point at least slightly to the left side
    mask = np.logical_and(mask, dphase[:, :, :, 1] > 1e-8)
    # Remove border
    mask[:, :, [0, -1]] = 0
    mask[:, [0, -1], :] = 0
    # Remove pixels with no neighbors
    weights = np.ones((1, 3, 3), dtype=int)
    weights[0, 1, 1] = 0
    neighbors = ndimage.convolve(mask.astype(int), weights, mode='constant')
    mask = np.logical_and(mask, neighbors > 1)
    print_masked("mask06_0.png", mask[0])
    print_masked("mask06_1.png", mask[1])
    return mask


def reconstruct(calibration, lit, dark, primary, cue, wave_count, shift=None,
                estimate_projector=False):
    gray, background = mega.rgb2gray(lit), mega.rgb2gray(dark)
    primary, cue = mega.rgb2gray(primary), mega.rgb2gray(cue)
    phase, dphase, primary, cue = decode_with_cue(primary, cue, wave_count)
    if shift is not None:
        phase += shift
    mask_args = gray, background, phase, dphase, primary, cue, wave_count
    mask = stdmask(*mask_args)
    phase *= phase.shape[2] / (2 * np.pi)

    indices = sl.match_epipolar_maps(phase, mask)
    colors = np.mean([mega.bilinear_interpolate(lit[i],
                                                indices[i, :, 0],
                                                indices[i, :, 1],
                                                axes=(0, 1))
                      for i in range(len(lit))], axis=0)
    points = sl.triangulate_epipolar(calibration, indices)
    if calibration.projector is None and estimate_projector:
        calibration.projector = sl.calibration.camera()
        calibration.projector.position = calibration.cam0.position / 2
        calibration.projector.position += calibration.cam1.position / 2
        G = 1080 / wave_count
        calibration.projector.wave_vector = np.array([G, 0, 0])
    if calibration.projector is None:
        normals = mega.estimate_normals(points)
    else:
        def normalize(v):
            return v / np.linalg.norm(v, axis=-1, keepdims=True)
        p = normalize(points - calibration.projector.position)
        c0 = normalize(points - calibration.cam0.position)
        c1 = normalize(points - calibration.cam1.position)
        lambda_p = calibration.projector.wave_vector.copy()[None, :]
        p_focal_length = 20
        lambda_p = lambda_p * p[:, -1, None] / p_focal_length
        # orthonormal p, lambda_p, omega_p
        lambda_p -= (lambda_p * p).sum(axis=1)[:, None] * p
        omega_p = normalize(np.cross(p, lambda_p))
        print('p', p.mean(axis=0), np.linalg.norm(p, axis=1).mean())
        print('c0', c0.mean(axis=0), np.linalg.norm(c0, axis=1).mean())
        print('c1', c1.mean(axis=0), np.linalg.norm(c1, axis=1).mean())

        print('lambda_p', lambda_p.mean(axis=0),
              np.linalg.norm(lambda_p, axis=1).mean())
        print('omega_p', omega_p.mean(axis=0),
              np.linalg.norm(omega_p, axis=1).mean())
        omega_c = np.zeros((2, len(p), 3), dtype=p.dtype)
        lambda_c = np.zeros((2, len(p), 3), dtype=p.dtype)
        for i, c, cam in zip((0, 1), (c0, c1), (calibration.rectified_cam0,
                                                calibration.rectified_cam1)):
            idx = indices[i].T
            lambda_c[i, :, 0] = map_coordinates(dphase[i, ..., 1], idx)
            lambda_c[i, :, 1] = map_coordinates(dphase[i, ..., 0], idx)
            norm_2 = (lambda_c[i] * lambda_c[i]).sum(axis=1)[:, None]
            lambda_c[i] *= 2 * np.pi / norm_2
            lambda_c[i, :, :2] *= c[:, -1, None] / cam.focal_vector[None, :]
            lambda_c[i] = lambda_c[i].dot(cam.rotation)
            # orthonormal c, lambda_c, omega_c
            lambda_c[i] -= (lambda_c[i] * c).sum(axis=1)[:, None] * c
            omega_c[i] = normalize(np.cross(c, lambda_c[i]))
            print('lambda_c[{}]'.format(i), lambda_c[i].mean(axis=0),
                  np.linalg.norm(lambda_c[i], axis=1).mean())
            print('omega_c[{}]'.format(i), omega_c[i].mean(axis=0),
                  np.linalg.norm(omega_c[i], axis=1).mean())
        n0 = sl.normals_from_projection(lambda_p, lambda_c[0],
                                        omega_p, omega_c[0], p, c0)
        n1 = sl.normals_from_projection(lambda_p, lambda_c[1],
                                        omega_p, omega_c[1], p, c1)
        # n1 *= np.sign((n0 * n1).sum(axis=1))[:, None]
        # normals = n0  # (n0 + n1) / 2
        # align = np.array([((normals * p).sum(axis=1)),
        #                   ((normals * c0).sum(axis=1)),
        #                   ((normals * c1).sum(axis=1))])
        # i = np.argmax(np.abs(align), axis=0)
        # align = np.sign(align[i, np.arange(len(i))])[:, None]
        # n0 *= align
        # n1 *= align
        # normals *= align
        normals = normalize(n0)
    return mega.pointcloud(points, colors, normals)
