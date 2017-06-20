
# Author: Gael Varoquaux <gael dot varoquaux at normalesup.org>
# Copyright (c) 2008, Enthought, Inc.
# License: BSD style.

from numpy import array, random, linspace, pi, ravel, cos, sin, empty
from tvtk.api import tvtk, write_data
from mayavi.sources.vtk_data_source import VTKDataSource
import numpy as np
from mayavi import mlab


def rectilinear_grid():
    data = random.random((1000, 3, 3))
    r = tvtk.RectilinearGrid()
    r.point_data.scalars = data.ravel()
    r.point_data.scalars.name = 'scalars'
    r.dimensions = data.shape
    # r.x_coordinates = array((0, 0.7, 1.4))
    r.x_coordinates = np.arange(1000)
    r.y_coordinates = array((0, 1, 3))
    r.z_coordinates = array((0, .5, 2))
    return r


ugrid=rectilinear_grid()
outfname='./rectilinearGrid.vtk'
# outfname='./test_4_pts_structured.vtk'
write_data(ugrid, outfname)