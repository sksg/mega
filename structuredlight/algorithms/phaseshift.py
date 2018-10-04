import numpy as np
from concurrent.futures import ThreadPoolExecutor as thread_pool
# from concurrent.futures import ProcessPoolExecutor as process_pool
import time  # to evaluate sppedup from parallelization
from ... import structuredlight as sl


def decode_sequential(frames):
    dft = np.fft.fft(frames, axis=1)
    phase = 2.0 * np.pi - np.angle(dft[:, 1])
    amplitude = np.absolute(dft[:, 1])
    energy = np.absolute(dft[:, 1:]).sum(axis=1)
    return phase, amplitude, energy


def decode(frames, step=1):
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
    mask = np.logical_and(adjusted > 30, adjusted < 250)
    # Threshold on amplitude at primary frequency
    mask = np.logical_and(mask, primary[1] > 2.0 * wave_count)
    # Threshold on amplitudes versus total energy
    mask = np.logical_and(mask, primary[1] > 0.25 * primary[2])
    mask = np.logical_and(mask, cue[1] > 0.25 * cue[2])
    # Threshold on gradient of phase. Cannot be too large or too small
    mask = np.logical_and(mask, np.linalg.norm(dphase, axis=3) < 0.02)
    mask = np.logical_and(mask, np.linalg.norm(dphase, axis=3) > 1e-8)
    # Cannot must point at least slightly to the left side
    mask = np.logical_and(mask, dphase[:, :, :, 1] > 1e-8)
    # Remove border
    mask[:, :, [0, -1]] = 0
    mask[:, [0, -1], :] = 0
    # Remove pixels with no neighbors
    K = np.ones((3, 3), dtype=np.float64)
    K[1, 1] = 0
    neighbors = sl.filter2D(mask.astype(np.float64), -1, K)
    mask = np.logical_and(mask, neighbors > 1)
    return mask


def reconstruct(calibration, lit, dark, primary, cue, wave_count, shift=None):
    gray, background = sl.grayscale(lit), sl.grayscale(dark)
    primary, cue = sl.grayscale(primary), sl.grayscale(cue)
    phase, dphase, primary, cue = decode_with_cue(primary, cue, wave_count)
    if shift is not None:
        phase += shift
    mask_args = gray, background, phase, dphase, primary, cue, wave_count
    mask = stdmask(*mask_args)
    phase *= phase.shape[2] / (2 * np.pi)

    indices = sl.match_epipolar_maps(phase, mask)
    colors = np.mean([sl.bilinear_interpolate(lit[i],
                                              indices[i, :, 0],
                                              indices[i, :, 1],
                                              axes=(0, 1))
                      for i in range(len(lit))], axis=0)
    points = sl.triangulate_epipolar(calibration, indices)
    normals = sl.estimate_normals(points)
    return sl.pointcloud(points, colors, normals)
