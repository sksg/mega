import numpy as np
from scipy import ndimage
from scipy.fftpack import fft
import os
from concurrent.futures import ThreadPoolExecutor as thread_pool
import mega
from .. import structuredlight as sl


def _ndtake(indices, axis):  # simple slicing, much faster than np.take
    return (slice(None),) * (axis - 1) + (indices,)


def decode_vectorized(data, axis=-4):
    axis = axis + int(axis < 0) * data.ndim
    dft = fft(data, axis=axis)
    order1st = _ndtake(1, axis)  # first order Fourier transform
    phase = np.mod(-np.angle(dft[order1st]), 2 * np.pi)
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
    shape = data.shape[:axis] + data.shape[axis + 1:]
    print("decode_parallel. Shape =", shape, flush=True)
    phase = np.empty(shape, data.dtype)
    amplitude, power = np.empty_like(phase), np.empty_like(phase)
    pool, futures = thread_pool(), {}
    for i in np.ndindex(shape[:-2]):
        print(i, flush=True)
        exit()
        data_i = i[:axis] + (slice(None),) + i[axis:]
        futures[i] = pool.submit(decode_vectorized, data[data_i], axis)
    for i in np.ndindex(shape[:-2]):
        phase[i], amplitude[i], power[i] = futures[i].result()  # asynchronous
    return phase, amplitude, power


def decode(data, axis=-4):
    if 'MEGA_PARALLELIZE' in os.environ and not os.environ['MEGA_PARALLELIZE']:
        return decode_sequential(data, axis)
    else:
        return decode_parallel(data, axis)


def unwrap_phase_with_cue(phase, cue, wave_count):
    phase_cue = np.mod(cue - phase, 2 * np.pi)
    phase_cue = np.round(((phase_cue * wave_count) - phase) / (2 * np.pi))
    return (phase + (2 * np.pi * phase_cue)) / wave_count


def stdmask(gray, dark, phase, dphase, primary, cue, wave_count):
    # Threshold on saturation and under exposure
    adjusted = gray - dark
    if adjusted.dtype == np.uint8:
        mask = np.logical_and(adjusted > 0.1 * 255, adjusted < 0.9 * 255)
    else:
        mask = np.logical_and(adjusted > 0.1, adjusted < 0.9)
    # Threshold on phase
    mask = np.logical_and(mask, phase >= 0.0)
    mask = np.logical_and(mask, phase <= 2 * np.pi)
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


def decode_with_cue_vectorized(gray, dark, primary, cue, N, maskfn=stdmask):
    primary = decode_vectorized(primary, axis=-4)
    cue = decode_vectorized(cue, axis=-4)
    phase = unwrap_phase_with_cue(primary[0], cue[0], N)
    dphase = np.stack(np.gradient(np.squeeze(phase), axis=(-2, -1)), axis=-1)
    mask = maskfn(gray, dark, phase, dphase, primary, cue, N)
    return phase, dphase, mask


def decode_with_cue_sequential(gray, dark, primary, cue, N, maskfn=stdmask):
    primary = decode_sequential(primary, axis=-4)
    cue = decode_sequential(cue, axis=-4)
    phase = unwrap_phase_with_cue(primary[0], cue[0], N)
    dphase = np.stack(np.gradient(np.squeeze(phase), axis=(-2, -1)), axis=-1)
    mask = maskfn(gray, dark, phase, dphase, primary, cue, N)
    return phase, dphase, mask


def decode_with_cue_parallel(gray, dark, primary, cue, N, maskfn=stdmask):
    primary = decode_parallel(primary, axis=-4)
    cue = decode_parallel(cue, axis=-4)
    phase = unwrap_phase_with_cue(primary[0], cue[0], N)
    dphase = np.stack(np.gradient(np.squeeze(phase), axis=(-2, -1)), axis=-1)
    mask = maskfn(gray, dark, phase, dphase, primary, cue, N)
    return phase, dphase, mask


def decode_with_cue(gray, dark, primary, cue, N, maskfn=stdmask):
    if 'MEGA_PARALLELIZE' in os.environ and not os.environ['MEGA_PARALLELIZE']:
        return decode_with_cue_sequential(gray, dark, primary, cue, N, maskfn)
    else:
        return decode_with_cue_parallel(gray, dark, primary, cue, N, maskfn)


def decode2D_with_cue_sequential(gray, dark, P0, C0, P1, C1, N, Mfn=stdmask):
    ph0, dph0, mask0 = decode_with_cue_sequential(gray, dark, P0, C0, N, Mfn)
    ph1, dph1, mask1 = decode_with_cue_sequential(gray, dark, P1, C1, N, Mfn)
    phase = np.stack((ph0, ph1), axis=2)
    dphase = np.stack((dph0, dph1), axis=2)
    mask = mask0 & mask1
    return phase, dphase, mask


def decode2D_with_cue_parallel(gray, dark, P0, C0, P1, C1, N, Mfn=stdmask):
    ph0, dph0, mask0 = decode_with_cue_parallel(gray, dark, P0, C0, N, Mfn)
    ph1, dph1, mask1 = decode_with_cue_parallel(gray, dark, P1, C1, N, Mfn)
    phase = np.stack((ph0, ph1), axis=2)
    dphase = np.stack((dph0, dph1), axis=2)
    mask = mask0 & mask1
    return phase, dphase, mask


def decode2D_with_cue(gray, dark, P0, C0, P1, C1, N, Mfn=stdmask):
    if 'MEGA_PARALLELIZE' in os.environ and not os.environ['MEGA_PARALLELIZE']:
        return decode2D_with_cue_sequential(gray, dark, P0, C0, P1, C1, N, Mfn)
    else:
        return decode2D_with_cue_parallel(gray, dark, P0, C0, P1, C1, N, Mfn)
