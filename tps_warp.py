import numpy as np
from matplotlib import pyplot as plt
import os
from warp_3d import warp_images

import pyvista


from numpy import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


save, displau = True, True

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)


def _get_regular_grid_mesh(vertices, points_per_dim, inner_only=False):
    m = np.min(vertices, axis=0)
    M = np.max(vertices, axis=0)

    if inner_only:
        points_per_dim += 2

    h = np.linspace(m[0], M[0], points_per_dim)
    w = np.linspace(m[1], M[1], points_per_dim)
    d = np.linspace(m[2], M[2], points_per_dim)

    if inner_only:
        rows, cols, layers = np.meshgrid(w[1:-1], h[1:-1], d[1:-1])
    else:
        rows, cols, layers = np.meshgrid(w, h, d)

    return np.dstack([cols.flat, rows.flat, layers.flat])[0]

def _get_regular_grid(image_stack, points_per_dim):
    nrows, ncols, nlayers = image_stack.shape[0], image_stack.shape[1], image_stack.shape[2]
    rows = np.linspace(0, nrows, points_per_dim)
    cols = np.linspace(0, ncols, points_per_dim)
    layers = np.linspace(0, nlayers, points_per_dim)
    rows, cols, layers = np.meshgrid(rows, cols, layers)
    return np.dstack([cols.flat, rows.flat, layers.flat])[0]


def _generate_random_vectors(src_points, scale):
    dst_pts = src_points + np.random.uniform(-scale, scale, src_points.shape)
    return dst_pts


def _thin_plate_spline_warp(image, src_points, dst_points, keep_corners=True):
    width, height = image.shape[:2]
    depth = width
    image_stack = image[..., np.newaxis, ::].repeat(depth, axis=2)
    if keep_corners:
        # corner_points = np.array(
        #     [[0, 0], [0, width], [height, 0], [height, width]])
        corner_points = np.array(
            [[0, 0, 0], [0, width, 0], [height, 0, 0], [height, width, 0],
            [0, 0, depth], [0, width, depth], [height, 0, depth], [height, width, depth]
             ])
        src_points = np.concatenate((src_points, corner_points))
        dst_points = np.concatenate((dst_points, corner_points))
    out = warp_images(src_points, dst_points,
                      np.moveaxis(image_stack, -1, 0),
                      (0, 0, 0, width - 1, height - 1, depth - 1))
    return np.moveaxis(np.array(out), 0, -1)


def _thin_plate_spline_warp_mesh(vertices, src_points, dst_points, keep_corners=False, keep_random_anchors=True, extend_region=True, scale=1):
    # width, height = image.shape[:2]
    # depth = width
    v = vertices.copy()

    #align around positive quadrant of 0,0,0
    shift = np.min(v, axis=0)
    v -= shift

    region_mins = []
    region_maxs = []
    for axis in range(3):
        region_mins.append(v[..., axis].min())
        region_maxs.append(v[..., axis].max())

    region = tuple(region_mins) + tuple(region_maxs)
        # corner_points = np.array(
        #     [[0, 0], [0, width], [height, 0], [height, width]])

    if extend_region:
        extension = 0.1 * np.max(v, axis=0)
        v += extension
        region = tuple(region_mins) + tuple(np.array(region_maxs) + extension)

    xmin, ymix, zmin, xmax, ymax, zmax = region
    corner_points = np.array(
        [[xmin, ymix, zmin], [xmin, ymax, zmin], [xmax, ymix, zmin], [xmax, ymax, zmin],
        [xmin, ymix, zmax], [xmin, ymax, zmax], [xmax, ymix, zmax], [xmax, ymax, zmax]
         ])

    if keep_random_anchors:
        anchor_points = np.random.uniform([xmin, ymix, zmin], [xmax, ymax, zmax], (3,3))
        src_points = np.concatenate((src_points, anchor_points))
        dst_points = np.concatenate((dst_points, anchor_points))


    if keep_corners:
        src_points = np.concatenate((src_points, corner_points))
        dst_points = np.concatenate((dst_points, corner_points))


    extended_region = np.round(np.array(region) * scale).astype(np.int)

    out = warp_images(src_points, dst_points,
                      None,
                      np.round(np.array(region) * scale).astype(np.int),
                      approximate_grid=10
                      )
    return out, corner_points #np.moveaxis(np.array(out), 0, -1)

def vectices_to_voxel(vertices, fill=False):
    v = vertices.copy()

    region_shape = []

    for axis in range(3):
        axis_ptp = v[..., axis].max() - v[..., axis].min()
        region_shape.append(axis_ptp + 1)

        #shift values to be between 0 and ptp
        v[..., axis] -= v[..., axis].min()

    region = np.zeros(tuple(np.round(region_shape).astype(np.int)))

    indexes = tuple(np.floor(v).astype(np.int).transpose())

    region[indexes] = 1
    return region


def find_threshold(res, length):
    high, low = 1, 0
    thresh = 0.4
    for i in range(100):
        s = np.sum(res > thresh)
        print(round(thresh, 2), high, low, s, length)

        prev = thresh
        if s > length:
            thresh += ((high - thresh) / 2)
            low = prev
        elif s < length:
            thresh -= ((thresh - low) / 2)
            high = prev
        else:
            return thresh
    return -1


def tps_warp(image, points_per_dim, scale):
    width, height = image.shape[:2]
    depth = width
    image_stack = image[..., np.newaxis, ::].repeat(depth, axis=2)

    src = _get_regular_grid(image_stack, points_per_dim=points_per_dim)
    dst = _generate_random_vectors( src, scale=scale*width)
    # out = _thin_plate_spline_warp(image, src, dst)
    out = _thin_plate_spline_warp(image, src, dst)
    return out

def tps_warp_2(image, dst, src):
    out = _thin_plate_spline_warp(image, src, dst)
    return out


def _get_random_source(vertices, length, scale=3, inside_only=True):
    m = np.min(vertices, axis=0)
    M = np.max(vertices, axis=0)

    centroid = (m + M) / 2

    M = np.max(np.abs(vertices), axis=0)
    # sources = np.random.uniform(m,  scale * M, (length, 3))
    sources = np.random.uniform(-M,  M, (length, 3))
    return sources + centroid


def tps_warp_mesh_rand(vertices, points_per_dim=5, overall_points=2, scale:float=1, grid=False):
    if grid:
        src = _get_regular_grid_mesh(vertices, points_per_dim=points_per_dim)
    else:
        src = _get_random_source(vertices, length=overall_points)
    dst = _generate_random_vectors(src, scale=scale*np.ptp(vertices, axis=0))
    return tps_warp_mesh(vertices, src, dst), src, dst

def tps_warp_mesh(vertices, dst, src):
    out = _thin_plate_spline_warp_mesh(vertices, src, dst)
    return out


def load_mesh(path, align=True, resolution=100):
    mesh = pyvista.read(path)

    v = mesh.points

    if align:
        v -= np.min(v, axis=0)

    v *= resolution / np.max(v)

    mesh.points = v

    return mesh

def save_mesh(path, mesh):
    pyvista.save_meshio(path, mesh, binary=True)



def display(corners, verts, src, dst, mesh, new_mesh):
    v = mesh.points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    m, M = np.min(corners[0]), np.max(corners[-1])
    ax.set_xlim3d(m-10, M+10)
    ax.set_ylim3d(m-10, M+10)
    ax.set_zlim3d(m-10, M+10)

    ax.scatter(v[..., 0], v[..., 1], v[..., 2], marker='.', color=(0.12156863, 0.34901961, 0.22352941, 0.6))
    ax.scatter(verts[..., 0], verts[..., 1], verts[..., 2], marker = '.', color=(0.3, 0.3, 1, 0.8))

    for s, d in zip(src, dst):
        # ax.plot([mean_x,v[0]], [mean_y,v[1]], [mean_z,v[2]], color='red', alpha=0.8, lw=3)
        # I will replace this line with:
        a = Arrow3D([s[0], d[0]], [s[1], d[1]],
                    [s[2], d[2]], mutation_scale=15,
                    lw=3, arrowstyle="-|>", color="r")
        ax.add_artist(a)

    ax.scatter(corners[..., 0], corners[..., 1], corners[..., 2], marker='x', color="black")
    # ax.scatter(dst[..., 0], dst[..., 1], dst[..., 2], marker = '^')

    plt.show()

    s = np.linalg.norm(v - verts, axis=1)
    plotter = pyvista.Plotter()  # instantiate the plotter
    plotter.add_mesh(mesh, style='wireframe', color=(0.8, 0.8, 0.8))  # add a mesh to the scene
    plotter.add_mesh(new_mesh, style='wireframe', scalars=s)
    for s, d in zip(src, dst):
        dir = d - s
        n = np.linalg.norm(dir)
        dir /= n
        plotter.add_arrows(s, dir, n )#, color='red')
    plotter.add_bounding_box()
    cpos = plotter.show()

def hash_deform(src, dst):
    return str(abs(hash(tuple(s for s in src.flatten()) + tuple(d for d in dst.flatten()))))

def run(display=display, save=save):


    # mesh objects can be created from existing faces and vertex data
    # mesh = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]],
    #                        faces=[[0, 1, 2]])
    # with open("/Users/shimonheimowitz/PycharmProjects/DeepSIM/datasets/simple.obj", 'rb') as f:
    filename = "/Users/shimonheimowitz/PycharmProjects/3d_tps/demo/data/mushroom_fixed_light.stl"
    mesh = load_mesh(filename, resolution=100)

    v = mesh.points

    (res, corners), src, dst = tps_warp_mesh_rand(v, points_per_dim=2, scale=0.5)

    vt = v.transpose()

    w_right, index_left = np.modf(vt)
    w_right = np.mean(w_right, axis=0)
    w_left = 1 - w_right
    index_right = tuple((index_left + 1).astype(np.int))
    index_left = tuple(index_left.astype(np.int))


    new_vert = np.array([
        res[0][index_left] * w_left + res[0][index_right] * w_right,
        res[1][index_left] * w_left + res[1][index_right] * w_right,
        res[2][index_left] * w_left + res[2][index_right] * w_right,
        ]).transpose()

    new_mesh = mesh.copy()
    new_mesh.points = new_vert

    if save:
        save_mesh(os.path.join("dataset", hash_deform(src, dst) + ".stl"), new_mesh)

    if display:
        display(corners, new_vert, src, dst, mesh, new_mesh)



def loop(iters=1000, display=False, save=True):
    for _ in range(iters):
        run(display=display, save=save)




def test():
    v = np.linspace(0, 10, 12).reshape(4,3)
    res = vectices_to_voxel(v)
    for layer in res:
        plt.imshow(layer)
        plt.show()
    pass

if __name__ == '__main__':
    loop()