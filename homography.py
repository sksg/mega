import numpy as np
import os
import cv2


def _homography(x, y):
    H, _ = cv2.findHomography(x, y, cv2.LMEDS)
    return H

homography = np.vectorize(_homography, signature='(n,2),(n,2)->(3,3)')
