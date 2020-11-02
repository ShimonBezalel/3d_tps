

from scipy import ndimage
import numpy
import numpy as np


def warp_images(from_points, to_points, stack, output_region, approximate_grid=2):
    """Define a thin-plate-spline warping transform that warps from the from_points
    to the to_points, and then warp the given images by that transform. This
    transform is described in the paper: "Principal Warps: Thin-Plate Splines and
    the Decomposition of Deformations" by F.L. Bookstein.
    Parameters:
        - from_points and to_points: Nx2 arrays containing N 2D landmark points.
        - stack: list of images to warp with the given warp transform.
        - output_region: the (xmin, ymin, xmax, ymax) region of the output
                image that should be produced. (Note: The region is inclusive, i.e.
                xmin <= x <= xmax)
        - interpolation_order: if 1, then use linear interpolation; if 0 then use
                nearest-neighbor.
        - approximate_grid: defining the warping transform is slow. If approximate_grid
                is greater than 1, then the transform is defined on a grid 'approximate_grid'
                times smaller than the output image region, and then the transform is
                bilinearly interpolated to the larger region. This is fairly accurate
                for values up to 10 or so.
    """
    transform = _make_inverse_warp(from_points, to_points, output_region, approximate_grid)
    return transform

def _make_inverse_warp(from_points, to_points, output_region, approximate_grid):
    x_min, y_min, z_min, x_max, y_max, z_max = output_region
    if approximate_grid is None: approximate_grid = 1
    x_steps = (x_max - x_min) / approximate_grid
    y_steps = (y_max - y_min) / approximate_grid
    z_steps = (z_max - z_min) / approximate_grid
    x, y, z = numpy.mgrid[x_min:x_max:x_steps*1j, y_min:y_max:y_steps*1j, z_min:z_max:z_steps*1j]

    # make the reverse transform warping from the to_points to the from_points, because we
    # do image interpolation in this reverse fashion
    transform = _make_warp(to_points, from_points, x, y, z)

    if approximate_grid != 1:
        # linearly interpolate the zoomed transform grid
        new_x, new_y, new_z = numpy.mgrid[x_min:x_max+1, y_min:y_max+1, z_min:z_max+1]
        x_fracs, x_indices = numpy.modf((x_steps-1)*(new_x-x_min)/float(x_max-x_min))
        y_fracs, y_indices = numpy.modf((y_steps-1)*(new_y-y_min)/float(y_max-y_min))
        z_fracs, z_indices = numpy.modf((z_steps-1)*(new_z-z_min)/float(z_max-z_min))
        x_indices = x_indices.astype(np.int16)
        y_indices = y_indices.astype(np.int16)
        z_indices = z_indices.astype(np.int16)
        x1 = 1 - x_fracs
        y1 = 1 - y_fracs
        z1 = 1 - z_fracs
        ix1 = (x_indices+1).clip(0, x_steps-1).astype(np.int16)
        iy1 = (y_indices+1).clip(0, y_steps-1).astype(np.int16)
        iz1 = (z_indices+1).clip(0, z_steps-1).astype(np.int16)

        transform_by_axis = []
        for axis in range(3):

            t000 = transform[axis][(x_indices, y_indices, z_indices)]
            t010 = transform[axis][(x_indices, iy1, z_indices)]
            t100 = transform[axis][(ix1, y_indices, z_indices)]
            t110 = transform[axis][(ix1, iy1, z_indices)]
            t001 = transform[axis][(x_indices, y_indices, iz1)]
            t011 = transform[axis][(x_indices, iy1, iz1)]
            t101 = transform[axis][(ix1, y_indices, iz1)]
            t111 = transform[axis][(ix1, iy1, iz1)]
            transform_axis = \
                t000 * x1 * y1 * z1 + \
                t010 * x1 * y_fracs * z1 + \
                t100 * x_fracs * y1 * z1 + \
                t110 * x_fracs * y_fracs * z1 + \
                t001 * x1 * y1 * z_fracs + \
                t011 * x1 * y_fracs * z_fracs + \
                t101 * x_fracs * y1 * z_fracs + \
                t111 * x_fracs * y_fracs * z_fracs
            transform_by_axis.append(transform_axis)
        # t00 = transform[1][(x_indices, y_indices)]
        # t01 = transform[1][(x_indices, iy1)]
        # t10 = transform[1][(ix1, y_indices)]
        # t11 = transform[1][(ix1, iy1)]
        # transform_y = t00*x1*y1 + t01*x1*y_fracs + t10*x_fracs*y1 + t11*x_fracs*y_fracs
        # transform = [transform_x, transform_y, transform_z]
        transform = transform_by_axis
    return transform

_small = 1e-100
def _U(x):
    return (x**2) * numpy.where(x<_small, 0, numpy.log(x))

def _interpoint_distances(points):
    xd = numpy.subtract.outer(points[:,0], points[:,0])
    yd = numpy.subtract.outer(points[:,1], points[:,1])
    zd = numpy.subtract.outer(points[:,2], points[:,2])
    return numpy.sqrt(xd**2 + yd**2 + zd**2)

def _make_L_matrix(points):
    n = len(points)
    K = _U(_interpoint_distances(points))
    P = numpy.ones((n, 4))
    P[..., 1:] = points
    #O = numpy.zeros((3, 3))
    O = numpy.zeros((4, 4))
    L = numpy.asarray(numpy.bmat([[K, P],[P.transpose(), O]]))
    return L

def _calculate_f(coeffs, points, x, y, z):
    w = coeffs[:-4]
    a1, ax, ay, az = coeffs[-4:]
    # The following uses too much RAM:
    # distances = _U(numpy.sqrt((points[:,0]-x[...,numpy.newaxis])**2 + (points[:,1]-y[...,numpy.newaxis])**2))
    # summation = (w * distances).sum(axis=-1)
    summation = numpy.zeros(x.shape)
    for wi, Pi in zip(w, points):
        summation += wi * _U(numpy.sqrt((x-Pi[0])**2 + (y-Pi[1])**2 + (z-Pi[2])**2))
    return a1 + ax*x + ay*y + az*z + summation

def _make_warp(from_points, to_points, x_vals, y_vals, z_vals):
    from_points, to_points = numpy.asarray(from_points), numpy.asarray(to_points)
    err = numpy.seterr(divide='ignore')
    L = _make_L_matrix(from_points)
    extra = 4 # 3
    V = numpy.resize(to_points, (len(to_points)+extra, 3))
    V[-4:, :] = 0
    coeffs = numpy.dot(numpy.linalg.pinv(L), V)
    x_warp = _calculate_f(coeffs[:,0], from_points, x_vals, y_vals, z_vals)
    y_warp =  _calculate_f(coeffs[:,1], from_points, x_vals, y_vals, z_vals)
    z_warp =  _calculate_f(coeffs[:,2], from_points, x_vals, y_vals, z_vals)
    numpy.seterr(**err)
    return [x_warp, y_warp, z_warp]