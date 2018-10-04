import numpy as np
import scipy.spatial
import plyfile
import warnings
from concurrent.futures import ThreadPoolExecutor as thread_pool
# from concurrent.futures import ProcessPoolExecutor as process_pool
# import time  # to evaluate speed-up from parallelization


class pointcloud(object):
    """pointcloud encapsulates positions, normals, and colors.

    The class can read and write Standford .ply files"""

    def __init__(self, positions=None, colors=None, normals=None):
        super(pointcloud, self).__init__()
        self.positions = positions
        self.colors = colors
        self.normals = normals

    def writePLY(self, filename, ascii=False):
        dtype = []
        N = -1
        if self.positions is not None:
            N = len(self.positions)
            dtype += [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
        if self.colors is not None:
            N = len(self.colors) if N == -1 else N
            dtype += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        if self.normals is not None:
            N = len(self.normals) if N == -1 else N
            dtype += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]

        error_msg = "Lengths of positions, colors, and normals must match."
        if self.positions is not None and N != len(self.positions):
            raise RuntimeError(error_msg)
        if self.colors is not None and N != len(self.colors):
            raise RuntimeError(error_msg)
        if self.normals is not None and N != len(self.normals):
            raise RuntimeError(error_msg)

        vertex = np.zeros((N,), dtype=dtype)

        if self.positions is not None:
            vertex['x'] = self.positions[:, 0].astype('f4')
            vertex['y'] = self.positions[:, 1].astype('f4')
            vertex['z'] = self.positions[:, 2].astype('f4')

        if self.colors is not None:
            # assuming RGB format
            vertex['red'] = self.colors[:, 0].astype('u1')
            vertex['green'] = self.colors[:, 1].astype('u1')
            vertex['blue'] = self.colors[:, 2].astype('u1')

        if self.normals is not None:
            vertex['nx'] = self.normals[:, 0].astype('f4')
            vertex['ny'] = self.normals[:, 1].astype('f4')
            vertex['nz'] = self.normals[:, 2].astype('f4')

        vertex = plyfile.PlyElement.describe(vertex, 'vertex')
        plyfile.PlyData([vertex], text=ascii).write(filename)
        return self

    def readPLY(self, filename):
        self.__init__()

        vertex = plyfile.PlyData.read(filename)['vertex']

        with warnings.catch_warnings():
            # numpy does not like to .view() into structured array
            warnings.simplefilter("ignore")

            if all([p in vertex.data.dtype.names for p in ('x', 'y', 'z')]):
                position_data = vertex.data[['x', 'y', 'z']]
                N = len(position_data.dtype.names)
                self.positions = position_data.view((position_data.dtype[0],
                                                     N))

            colored = all([p in vertex.data.dtype.names
                           for p in ('blue', 'green', 'red')])
            if colored:
                color_data = vertex.data[['blue', 'green', 'red']]
                N = len(color_data.dtype.names)
                self.colors = color_data.view((color_data.dtype[0], N))

            if all([p in vertex.data.dtype.names for p in ('nx', 'ny', 'nz')]):
                normal_data = vertex.data[['nx', 'ny', 'nz']]
                N = len(normal_data.dtype.names)
                self.normals = normal_data.view((normal_data.dtype[0], N))
        return self


def estimate_normals(positions, neighbor_count=10, max_distance=np.inf,
                     edge_tolerance=0.3):

    def find_neighbors(tree, indices=slice(None), neighbors=10):
        distances, neighbors = tree.query(tree.data[indices], k=neighbors)
        return neighbors, distances

    def fit_planes(points, mask=None):
        barycenters = points.mean(axis=-2)[..., None, :]
        baryvectors = (points - barycenters)
        if mask is not None:
            baryvectors[~mask] *= 0
        M = (baryvectors[..., None, :] * baryvectors[..., None]).sum(axis=-3)
        eig_values, eig_vectors = np.linalg.eigh(M)
        i = tuple(np.arange(0, eig_values.shape[i], dtype=int)
                  for i in range(0, len(eig_values.shape) - 1))
        indices = (*i, slice(None), np.abs(eig_values).argmin(axis=-1))
        return eig_vectors[indices]

    def align_normals(src_normals, ngb_normals):
        ngb_product = np.einsum('...i,...i ->...', src_normals, ngb_normals)
        flip = (1 - (ngb_product < 0).astype(int) * 2)[..., None]
        return ngb_normals * flip

    N = len(positions)
    K = neighbor_count
    step = 5000 // K

    tree = scipy.spatial.cKDTree(positions)
    normals = np.empty_like(positions)
    neighborhoods = np.empty((N, K), dtype=int)
    distances = np.empty((N, K), dtype=float)

    pool = thread_pool()
    points = [[j for j in range(i, min(N, i + step))]
              for i in range(0, N, step)]
    futures = [pool.submit(find_neighbors, tree, p, K) for p in points]
    for i, future in enumerate(futures):
        n, d = future.result()
        neighborhoods[i * step:i * step + step] = n
        distances[i * step:i * step + step] = d
    mask = distances < max_distance
    futures = [pool.submit(fit_planes, positions[neighborhoods[p]], mask[p])
               for p in points]
    for i, future in enumerate(futures):
        normals[i * step:i * step + step] = future.result()

    view = np.einsum('...i,...i ->...', normals, -positions)
    normals *= (1 - (view < 0).astype(int) * 2)[:, None]
    visited = np.zeros(N, dtype=bool)
    priority = np.abs(view)

    source = []
    while not all(visited):
        if len(source) == 0:
            source = np.array([priority.argmax()])
            priority[source] = -1
            visited[source] = True

        source_mask = mask[source]  # used to filter out false neighbors
        neighbors = neighborhoods[source][source_mask]
        source = source[np.nonzero(source_mask)[0][~visited[neighbors]]]
        neighbors = neighbors[~visited[neighbors]]
        edge = np.einsum('...i,...i ->...',
                         normals[source], normals[neighbors])
        source = source[edge > edge_tolerance]
        neighbors = neighbors[edge > edge_tolerance]
        step = max(1, len(source) // pool._max_workers)
        futures = []
        for i in range(min(len(source) // step, pool._max_workers)):
            j = slice(i * step, (i + 1) * step)
            src, ngb = source[j], neighbors[j]
            src, ngb = normals[src], normals[ngb]
            futures.append(pool.submit(align_normals, src, ngb))
        for i, future in enumerate(futures):
            j = slice(i * step, (i + 1) * step)
            normals[neighbors[j]] = future.result()
        source = np.unique(neighbors)
        priority[source] = -1
        visited[source] = True
    return normals
