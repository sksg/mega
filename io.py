import numpy as np
import cv2
import os
import re


def imread(path, color=True):
    if not os.path.exists(path):
        raise FileNotFoundError("No such file: '" + path + "'.")
    if color:
        return cv2.imread(path)[:, :, ::-1]  # BGR -> RGB
    else:
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)[:, :, None]


def imwrite(path, image):
    image = image.reshape((*image.shape[:2], -1))  # Make at least 3D.
    return cv2.imwrite(path, image[:, :, ::-1])  # RGB -> BGR


def read_images(path, rexp=r'.*\.png', sort=None, filter=None, color=True):
    files = [os.path.join(path, f)
             for f in os.listdir(path)
             if re.match(rexp, f)]
    if filter is not None:
        files = filter(files)
    if sort is not None:
        files = sort(files)
    if files == []:
        return []
    image0 = imread(files[0], color)
    images = np.empty((len(files), *image0.shape), dtype=image0.dtype)
    images[0] = image0
    for i, path in enumerate(files[1:], 1):
        images[i] = imread(path, color)
    return images


def write_images(directory, images, format=None):
    images = np.asarray(images)
    shape = images.shape[:-3]
    if format is None:
        format = "image" + ("_{}" * len(shape)) + ".png"
    path = os.path.join(directory, format)
    for idx in np.ndindex(shape):
        filepath = path.format(*idx)
        directory, _ = os.path.split(filepath)
        os.makedirs(directory, exist_ok=True)
        imwrite(filepath, images[idx])
