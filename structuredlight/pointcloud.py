import numpy as np
import plyfile
import warnings
import scipy.spatial


class pointcloud(object):
    """pointcloud encapsulates an entire point cloud according to the Standford
    .PLY standard.

    The class can read and write .PLY files, and has limited function for
    convenience. For example:
    - Normal estimation based of neighborhood tangential planes."""

    def __init__(self, positions=None, colors=None, normals=None):
        super(pointcloud, self).__init__()
        self.positions = positions
        self.colors = colors
        self.normals = normals
        self.tree = None

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

        if self.positions is not None and N != len(self.positions):
            print("\npointcloud error: N =", N, ", len(positions) =",
                  len(self.positions))
            msg = "Lengths of positions, colors, and normals must match."
            raise RuntimeError(msg)
        if self.colors is not None and N != len(self.colors):
            print("\npointcloud error: N =", N, ", len(colors) =",
                  len(self.colors))
            msg = "Lengths of positions, colors, and normals must match."
            raise RuntimeError(msg)
        if self.normals is not None and N != len(self.normals):
            print("\npointcloud error: N =", N, ", len(normals) =",
                  len(self.normals))
            msg = "Lengths of positions, colors, and normals must match."
            raise RuntimeError(msg)

        vertex = np.zeros((N,), dtype=dtype)

        if self.positions is not None:
            vertex['x'] = self.positions[:, 0].astype('f4')
            vertex['y'] = self.positions[:, 1].astype('f4')
            vertex['z'] = self.positions[:, 2].astype('f4')

        if self.colors is not None:
            # BGR format
            vertex['blue'] = self.colors[:, 0].astype('u1')
            vertex['green'] = self.colors[:, 1].astype('u1')
            vertex['red'] = self.colors[:, 2].astype('u1')

        if self.normals is not None:
            # BGR format
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

    def buildTree(self):
        self.tree = scipy.spatial.cKDTree(self.positions)

    def generateNormals(self, neighbor_count=10, coherent_neighbor_count=8,
                        max_distance=np.inf, buffer_size=None):
        if self.tree is None:
            self.buildTree()

        if buffer_size is None:
            buffer_size = min(50000 // neighbor_count, len(self.positions))

        self.normals = np.zeros_like(self.positions)

        k = neighbor_count
        for step in range(0, len(self.positions), buffer_size):
            positions = self.positions[step:step + buffer_size]
            distances, neighbors = self.tree.query(positions, k=k)
            valid = (distances <= max_distance)[:, :, None]
            barycenters = (self.positions[neighbors]).mean(axis=1)
            # barycenters = (self.positions[neighbors] * valid).sum(axis=1)
            # barycenters /= valid.sum(axis=1)
            vectors = (self.positions[neighbors] - barycenters[:, None])
            # vectors = vectors * valid
            M = (vectors[:, :, None] * vectors[:, :, :, None]).sum(axis=1)

            eval, evec = np.linalg.eigh(M)
            eval = np.abs(eval)
            indices = (np.arange(0, len(eval), dtype=int),
                       slice(None),
                       eval.argmin(axis=1))
            normals = evec[indices]

            view_product = np.einsum('...i,...i ->...', normals, -positions)
            alignment = 1 - (view_product < 0).astype(int) * 2
            normals = normals * alignment[:, None]

            self.normals[step:step + buffer_size] = normals

        visited = np.zeros(len(self.positions), dtype=bool)
        k = coherent_neighbor_count
        while visited.sum() != len(self.positions):
            i = visited.argmin()  # first zero occurance
            visited[i] = True
            distances, neighbors = self.tree.query(self.positions[i], k=k)
            neighbors = neighbors[distances <= max_distance]
            normal = self.normals[i, None]
            ngb_normals = self.normals[neighbors]
            ngb_product = np.abs(np.einsum('...i,...i ->...',
                                           normal, ngb_normals))

            edge_tolerance = 0.3
            heap = [(i, neighbors[ngb_product > edge_tolerance])]

            while len(heap) > 0:
                (i, neighbors), heap = heap[-1], heap[:-1]
                if all(visited[neighbors]):
                    continue
                neighbors = neighbors[~visited[neighbors]]
                src_normal = self.normals[i, None]
                ngb_normals = self.normals[neighbors]
                ngb_positions = self.positions[neighbors]
                ngb_product = np.einsum('...i,...i ->...',
                                        src_normal, ngb_normals)
                view_product = np.einsum('...i,...i ->...',
                                         ngb_normals, -ngb_positions)
                alignment = np.logical_and(ngb_product < 0, view_product < 0)
                alignment = (1 - alignment.astype(int) * 2)[:, None]
                self.normals[neighbors] = ngb_normals * alignment
                distances, next_neighbors = self.tree.query(ngb_positions, k=k)
                ngb_normals = self.normals[neighbors, None]
                nngb_normals = self.normals[next_neighbors]
                ngb_product = np.abs(np.einsum('...i,...i ->...',
                                               ngb_normals, nngb_normals))
                valid = np.logical_and(distances <= max_distance,
                                       ngb_product > edge_tolerance)
                heap += [(n, nn[v])
                         for n, nn, v in zip(neighbors, next_neighbors, valid)]
                visited[neighbors] = True
        return self
