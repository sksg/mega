import cv2


def interpolate_affine(affine_image, points, mask=None, window=10, tol=50):
    # Using local homography to interpolate affine image
    return_array = np.empty_like(points)
    window = np.mgrid[0:window, 0:window].T.reshape(-1, 2)
    for idx in np.ndindex(points.shape[:-1]):
        p = points[idx]
        p_window = np.clip(p.astype(int) + window, 0, affine_image.shape)
        i_window = affine_image[p_window]
        m_window = mask[p_window]
        if m_window.sum() < tol:
            return_array[idx] = np.nan
        else:
            i_window = i_window[m_window]
            p_window = p_window[m_window]
            H = cv2.findHomography(p_window, i_window, cv2.LMEDS)
            p = H.dot(p)
            return_array[idx] = p[:2] / p[2]
    return return_array
