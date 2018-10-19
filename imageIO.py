import numpy as np
import cv2
import os
import re


def imread(path):
    if not os.path.exists(path):
        raise FileNotFoundError("No such file: '" + path + "'.")
    return cv2.imread(path)


def read_images(path, rexp=r'.*\.png'):
    files = [os.path.join(path, f)
             for f in os.listdir(path)
             if re.match(rexp, f)]
    image0 = imread(files[0])
    images = np.empty((len(files), *image0.shape), dtype=image0.dtype)
    images[0] = image0
    for i, path in enumerate(files[1:], 1):
        images[i] = imread(path)
    return images
