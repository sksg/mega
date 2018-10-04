import numpy as np
import imageio, os, re


def write_frame(path, frame):
    return imageio.imwrite(path, frame)


def write_frame_sequence(directory, frames):
    path = directory + "/frames{}_{}.png"
    for i in range(frames.shape[0]):
        for j in range(frames.shape[1]):
            write_frame(path.format(i, j), frames[i, j])


def read_frame(frame):
    if not os.path.exists(frame):
        raise RuntimeError("Missing frame " + frame + " for stereo sequence.")
    return imageio.imread(frame)


def read_frame_sequence(directory):
    rexp = r'frames[0-1]_[0-9]*.png'
    N = len([f for f in os.listdir(directory) if re.match(rexp, f)]) // 2
    path = directory + "/frames{}_{}.png"
    frame00 = imageio.imread(path.format(0, 0))
    frames = np.empty((2, N, *frame00.shape), dtype=frame00.dtype)
    frames[0, 0] = frame00
    for i in (0, 1):
        for j in range(1 if i == 0 else 0, N):
            frames[i, j] = read_frame(path.format(i, j))
    return frames
