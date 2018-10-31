import numpy as np
from scipy import ndimage
import os
from concurrent.futures import ThreadPoolExecutor as thread_pool
import mega
from .. import structuredlight as sl


def _ndtake(indices, axis):  # simple slicing, much faster than np.take
    return (slice(None),) * (axis - 1) + (indices,)


def decode_vectorized(data, axis=-4):
    axis = axis + int(axis < 0) * data.ndim
    dft = np.fft.fft(data, axis=axis)
    order1st = _ndtake(1, axis)  # first order Fourier transform
    phase = 2.0 * np.pi - np.angle(dft[order1st])
    amplitude = np.absolute(dft[order1st])
    power = np.absolute(dft[_ndtake(slice(1, None), axis)]).sum(axis=axis)
    return phase, amplitude, power


def decode_sequential(data, axis=-4):
    phase = np.empty(data.shape[:axis] + data.shape[axis + 1:], data.dtype)
    amplitude, power = np.empty_like(phase), np.empty_like(phase)
    for i in np.ndindex(data.shape[:axis]):
        phase[i], amplitude[i], power[i] = decode_vectorized(data[i], axis)
    return phase, amplitude, power


def decode_parallel(data, axis=-4):
    phase = np.empty(data.shape[:axis] + data.shape[axis + 1:], data.dtype)
    amplitude, power = np.empty_like(phase), np.empty_like(phase)
    pool, futures = thread_pool(), {}
    for i in np.ndindex(data.shape[:axis]):
        futures[i] = pool.submit(decode_vectorized, data[i], axis)
    for i in np.ndindex(data.shape[:axis]):
        phase[i], amplitude[i], power[i] = futures[i].result()  # asynchronous
    return phase, amplitude, power


def decode(data, axis=-4):
    if 'MEGA_PARALLELIZE' in os.environ and not os.environ['MEGA_PARALLELIZE']:
        return decode_sequential(data, axis)
    else:
        return decode_parallel(data, axis)


def unwrap_phase_with_cue(phase, cue, wave_count):
    phase_cue = np.mod(cue - phase, 2 * np.pi)
    P = np.round(((phase_cue * wave_count) - phase) / (2 * np.pi))
    return (phase + (2 * np.pi * P)) / wave_count


def decode_with_cue_vectorized(primary, cue, wave_count, axis=-4):
    primary = decode_vectorized(primary, axis)
    cue = decode_vectorized(cue, axis)
    phase = unwrap_phase_with_cue(primary[0], cue[0], wave_count)
    dphase = np.stack(np.gradient(np.squeeze(phase), axis=(-2, -1)), axis=-1)
    return phase, dphase, primary, cue


def decode_with_cue_sequential(primary, cue, wave_count, axis=-4):
    primary = decode_sequential(primary, axis)
    cue = decode_sequential(cue, axis)
    phase = unwrap_phase_with_cue(primary[0], cue[0], wave_count)
    dphase = np.stack(np.gradient(np.squeeze(phase), axis=(-2, -1)), axis=-1)
    return phase, dphase, primary, cue


def decode_with_cue_parallel(primary, cue, wave_count, axis=-4):
    primary = decode_parallel(primary, axis)
    cue = decode_parallel(cue, axis)
    phase = unwrap_phase_with_cue(primary[0], cue[0], wave_count)
    dphase = np.stack(np.gradient(np.squeeze(phase), axis=(-2, -1)), axis=-1)
    return phase, dphase, primary, cue


def decode_with_cue(primary, cue, wave_count, axis=-4):
    if 'MEGA_PARALLELIZE' in os.environ and not os.environ['MEGA_PARALLELIZE']:
        return decode_with_cue_sequential(primary, cue, wave_count, axis)
    else:
        return decode_with_cue_parallel(primary, cue, wave_count, axis)


def stdmask(gray, background, phase, dphase, primary, cue, wave_count):
    # Threshold on saturation and under exposure
    adjusted = gray - background
    if adjusted.dtype == np.uint8:
        mask = np.logical_and(adjusted > 0.1 * 255, adjusted < 0.9 * 255)
    else:
        mask = np.logical_and(adjusted > 0.1, adjusted < 0.9)
    # Threshold on amplitude at primary frequency
    mask = np.logical_and(mask, primary[1] > 0.01 * wave_count)
    # Threshold on amplitudes; must be at least 1/4 of the power
    mask = np.logical_and(mask, primary[1] > 0.25 * primary[2])
    mask = np.logical_and(mask, cue[1] > 0.25 * cue[2])
    # Threshold on gradient of phase. Cannot be too large or too small
    dph = np.linalg.norm(dphase, axis=-1, keepdims=True)
    mask = np.logical_and(mask, dph < 1e-2)
    mask = np.logical_and(mask, dph > 1e-8)
    # Must point at least slightly to the left side
    mask = np.logical_and(mask, dphase[..., 1, None] > 1e-8)
    # Remove borders
    mask[..., [0, -1], :] = 0
    mask[..., [0, -1], :, :] = 0
    # Remove pixels with no neighbors
    weights = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)[..., None]
    weights = np.broadcast_to(weights, mask.shape[:-3] + weights.shape)
    neighbors = ndimage.convolve(mask.astype(int), weights, mode='constant')
    mask = np.logical_and(mask, neighbors > 1)
    return mask


def reconstruct(stereo, lit, dark, primary, cue, wave_count, shift=None):
    gray, background = mega.rgb2gray(lit), mega.rgb2gray(dark)
    primary = mega.rgb2gray(primary).astype(np.float64) / 255
    cue = mega.rgb2gray(cue).astype(np.float64) / 255
    phase, dphase, primary, cue = decode_with_cue(primary, cue, wave_count,
                                                  axis=-4, n=1)
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
    points = sl.triangulate_epipolar(stereo, indices)
    normals = mega.estimate_normals(points)
    return mega.pointcloud(points, colors, normals)
