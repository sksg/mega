import numpy as np

default_assert_tolerance = 1e-12


def _length(v, *args, **kwargs):
    """Short hand for numpy.linalg.norm"""
    return np.linalg.norm(v, *args, **kwargs)


def _normalize(v, *args, **kwargs):
    """Short hand for v / numpy.linalg.norm(v, keepdims=True)"""
    return v / _length(v, keepdims=True, *args, **kwargs)


def _vectordot(u, v, *args, **kwargs):
    """Specilization of the dot-operator. u and v are ndarrays of vectors"""
    u, v = np.broadcast_arrays(u, v)
    return np.einsum('...i,...i ->...', u, v).reshape(*u.shape[:-1], 1)


def _assert_unitvectors(unitvectors, tolerance=None):
    """Short hand assertion"""
    if tolerance is None:
        tolerance = default_assert_tolerance
    err = np.abs(_length(unitvectors, axis=1) - 1)
    is_unitvectors = (err < tolerance).all()
    assert is_unitvectors, "max(err) = " + str(err.max())


def _refract_direction(unitvectors, eta, normal):
    _assert_unitvectors(unitvectors)
    # precompute
    n_v = _vectordot(normal, unitvectors)

    # result = x * (unitvector - y * normal)
    x = 1 / eta
    y = n_v * (1 - np.sqrt(1 + (eta**2 - 1) / (n_v**2)))
    return x * (unitvectors - y * normal)


def _project_2_surface(vector, normal, projector=None):
    x = _vectordot(vector, normal)
    if projector is not None:
        x /= _vectordot(normal, projector)
        x = x * projector
    else:
        x = x * normal
    return vector - x


def _reconstruct_phase_vectors(phase_gradients,
                               surface_points, surface_normals,
                               z0, z1, c0, c1, p):
    K0 = phase_gradients[0]
    K1 = phase_gradients[1]
    P0 = 2 * np.pi * K0 / _vectordot(K0, K0)
    P1 = 2 * np.pi * K1 / _vectordot(K1, K1)
    P0 *= z0
    P1 *= z1
    P0 = np.hstack((P0, np.zeros((len(P0), 1), dtype=P0.dtype)))
    P1 = np.hstack((P1, np.zeros((len(P1), 1), dtype=P1.dtype)))
    print(P0.shape)
    print(surface_normals.shape)
    print(c0.shape)
    P0 = _project_2_surface(P0, surface_normals, -c0)  # on to 3D surface
    P1 = _project_2_surface(P1, surface_normals, -c1)
    P0 = _project_2_surface(P0, -p, p)  # to projector image
    P1 = _project_2_surface(P1, -p, p)
    return (P0 + P1) / 2


def estimate_phase_error(extiction, refraction_index,
                         surface_points, surface_normals,
                         camera0_focal_length, camera1_focal_length,
                         camera0_position, camera1_position,
                         projector_position, phase_gradients):

    p = _normalize(surface_points - projector_position[None, :], axis=1)
    c0 = _normalize(camera0_position[None, :] - surface_points, axis=1)
    c1 = _normalize(camera1_position[None, :] - surface_points, axis=1)

    p_ss = _refract_direction(p, refraction_index, surface_normals)
    c0_ss = -_refract_direction(-c0, refraction_index, surface_normals)
    c1_ss = -_refract_direction(-c1, refraction_index, surface_normals)

    z0 = _length(surface_points - c0, axis=1, keepdims=True)
    z1 = _length(surface_points - c1, axis=1, keepdims=True)
    z0 /= camera0_focal_length
    z1 /= camera1_focal_length
    Lambda = _reconstruct_phase_vectors(phase_gradients,
                                        surface_points,
                                        surface_normals,
                                        z0, z1, c0, c1, p)
    K = Lambda / (2 * np.pi * _vectordot(Lambda, Lambda))
    rK = K / extiction

    c0_ss_n = _vectordot(c0_ss, surface_normals)
    c1_ss_n = _vectordot(c1_ss, surface_normals)
    p_ss_n = _vectordot(p_ss, surface_normals)
    delta0 = p_ss * c0_ss_n - c0_ss * p_ss_n
    delta0 /= p_ss_n - c0_ss_n
    delta1 = p_ss * c1_ss_n - c1_ss * p_ss_n
    delta1 /= p_ss_n - c1_ss_n
    shift0 = np.squeeze(np.arctan(_vectordot(rK, delta0)))
    shift1 = np.squeeze(np.arctan(_vectordot(rK, delta1)))

    return np.stack((shift0, shift1), axis=0)
