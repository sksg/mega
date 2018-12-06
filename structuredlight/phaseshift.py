import numpy as np
from scipy import ndimage
from scipy.fftpack import fft
from ..utilities import ndtake
from ..parallize import parallize
from ..parallize import asparallel


@parallize('(x,t),_->(x),(x),(x)', isvec=True, default='sequential')
def decode(data, axis=-1):
    dft = fft(data, axis=axis)
    phase = np.mod(-np.angle(dft[ndtake(1, axis)]), 2 * np.pi)
    amplitude = np.absolute(dft[ndtake(1, axis)])
    power = np.absolute(dft[ndtake(slice(1, None), axis)]).sum(axis=-1)
    return phase, amplitude, power


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
    mask = np.squeeze(mask)
    # Threshold on phase
    mask = np.logical_and(mask, phase >= 0.0)
    mask = np.logical_and(mask, phase <= 2 * np.pi)
    # Threshold on amplitude at primary frequency
    mask = np.logical_and(mask, primary[1] > 0.01 * wave_count)
    # Threshold on amplitudes; must be at least 1/4 of the power
    mask = np.logical_and(mask, primary[1] > 0.25 * primary[2])
    mask = np.logical_and(mask, cue[1] > 0.25 * cue[2])
    # Threshold on gradient of phase. Cannot be too large or too small
    dph = np.linalg.norm(dphase, axis=-1)
    mask = np.logical_and(mask, dph < 1e-2)
    mask = np.logical_and(mask, dph > 1e-8)
    # Remove borders
    mask[..., [0, -1]] = 0
    mask[..., [0, -1], :] = 0
    # Remove pixels with no neighbors
    weights = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=int)
    weights = np.broadcast_to(weights, mask.shape[:-2] + weights.shape)
    neighbors = ndimage.convolve(mask.astype(int), weights, mode='constant')
    mask = np.logical_and(mask, neighbors > 1)
    return mask


@asparallel
def decode_with_cue(pinfo, gray, dark, primary, cue, N, maskfn=stdmask):
    primary = pinfo(decode)(np.squeeze(primary.swapaxes(-1, -4)), -1)
    cue = pinfo(decode)(np.squeeze(cue.swapaxes(-1, -4)), -1)
    phase = unwrap_phase_with_cue(primary[0], cue[0], N)
    dphase = np.stack(np.gradient(np.squeeze(phase), axis=(-2, -1)), axis=-1)
    mask = maskfn(gray, dark, phase, dphase, primary, cue, N)
    return phase, dphase, mask


@asparallel
def decode2D_with_cue(pinfo, gray, dark, P0, C0, P1, C1, N, Mfn=stdmask):
    ph0, dph0, mask0 = pinfo(decode_with_cue)(gray, dark, P0, C0, N, Mfn)
    ph1, dph1, mask1 = pinfo(decode_with_cue)(gray, dark, P1, C1, N, Mfn)
    phase = np.stack((ph0, ph1), axis=2)
    dphase = np.stack((dph0, dph1), axis=2)
    mask = mask0 & mask1
    return phase, dphase, mask
