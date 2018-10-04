import numpy as np
import imageio, re, os, hashlib


def read_frames(directory, cache_id=None):
    if cache_id is not None:
        chk = hashlib.md5((directory + cache_id).encode('utf-8')).hexdigest()
        cache = "cached_frames_{}.npy".format(chk)
        if os.path.exists(cache):
            return np.load(cache)

    files = [path for path in os.listdir(directory)
             if re.match(r'frames[0-1]_[0-9]*.png', path)]
    N = len(files) // 2
    frame00 = imageio.imread(directory + "/frames0_0.png")
    h, w, c = frame00.shape
    frames = np.empty((2, N, h, w, c), dtype=frame00.dtype)
    frames[0, 0] = frame00
    for i in (0, 1):
        for j in range(1 if i == 0 else 0, N):
            path = directory + "/frames{}_{}.png".format(i, j)
            if not os.path.exists(path):
                raise RuntimeError("Missing .png's for stereo sequence.")
            frame = imageio.imread(path)
            frames[i, j] = frame
    if cache_id is not None:
        np.save(cache, frames)
    return frames
