"""
Decimation
~~~~~~~~~~

Decimate a mesh

"""
# sphinx_gallery_thumbnail_number = 4
import pyvista as pv
from pyvista import examples
import numpy as np
import os
import random

def load_mesh(path, align=True, resolution=100):
    mesh = pv.read(path)

    v = mesh.points

    if align:
        v -= np.min(v, axis=0)

    v *= resolution / np.max(v)

    mesh.points = v

    return mesh


# Define a camera potion the shows this mesh properly


def decimate_comparison(n=10, lower=0.7, upper=0.999):
    cpos = [(0.4, -0.07, -0.31), (0.05, -0.13, -0.06), (-0.1, 1, 0.08)]
    dargs = dict(show_edges=True, style='wireframe')
    comparisons = n
    p = pv.Plotter(shape=(1, comparisons))
    for i, target_reduction in enumerate(np.linspace(lower, upper, n)):
        print(f"Reducing {target_reduction * 100.0} percent out of the original mesh")

        ###############################################################################
        decimated = mesh.decimate(target_reduction)
        # p.add_mesh(mesh, **dargs)
        # p.add_text("Input mesh", font_size=24)
        # p.camera_position = cpos
        # p.reset_camera()
        p.subplot(0, i)
        p.add_mesh(decimated, **dargs)
        p.add_text("Decimated mesh {}".format(round(target_reduction, 3), font_size=24))
        p.camera_position = cpos
        p.reset_camera()
        # p.subplot(i, 2)
        # p.add_mesh(pro_decimated, **dargs)
        # p.add_text("Pro Decimated mesh", font_size=24)
        p.camera_position = cpos
        p.reset_camera()
        p.link_views()
    p.show()

def display_dataset(n=50):

    rows = 5
    n = rows * (n // rows)

    p = pv.Plotter(shape=(rows, n//rows))
    data_dir = "dataset"

    complex_args = dict(show_edges=False, color='w')
    primitive_args = dict(show_edges=True, style='wireframe')


    files = random.sample(set(filter(lambda p: "o" in p, os.listdir(data_dir))), n)
    pos = 0
    for i in range(n // rows):
        for j in range(rows):
            file_path = files[pos]
            complex_mesh = load_mesh(os.path.join(data_dir, file_path))
            primitive_mesh = load_mesh(os.path.join(data_dir, file_path.replace("_o", "_p")))

            # separate meshes vertically
            # M = np.max(complex_mesh.points, axis=0)
            complex_mesh.points += [0, 50, 0]
            primitive_mesh.points -= [0, 50, 0]

            p.subplot(j, i)
            p.add_mesh(complex_mesh, **complex_args)
            p.add_mesh(primitive_mesh, **primitive_args)

            # p.add_text("".format(round(target_reduction, 3), font_size=24))
            # p.camera_position = cpos
            # p.reset_camera()
            # p.subplot(i, 2)
            # p.add_mesh(pro_decimated, **dargs)
            # p.add_text("Pro Decimated mesh", font_size=24)
            # p.camera_position = cpos
            # p.reset_camera()
            p.link_views()
            pos += 1
    p.show()



if __name__ == '__main__':
    display_dataset()
    # mesh = load_mesh("demo/data/mushroom_fixed_light.stl")

    # decimate_comparison()
    # decimated = mesh.decimate(0.999)

    # pv.save_meshio("demo/data/primitive/mushroom_fixed_light_p.stl", decimated, binary=True)