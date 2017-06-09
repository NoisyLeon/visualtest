import numpy as np


def write_unstructured_vtk(fname, x, y, z):
    with open(fname, 'rb') as f:
        