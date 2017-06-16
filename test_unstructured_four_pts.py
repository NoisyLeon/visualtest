from sympy.ntheory import primefactors
from tvtk.api import tvtk, write_data
import h5py
import numpy as np
# 
# field = np.array([0,1, 2, 3])
# pts = np.empty((4, 3), dtype=float)
# pts[..., 0] = np.array([0, 0, 0, 0])
# pts[..., 1] = np.array([0, 3, 2, 1])
# pts[..., 2] = np.array([0, 1, 2, 3]) # assign grid point position


x= np.arange(0, 10, 1);
y= np.arange(0, 10, 1);
z= np.arange(0, 10, 0.1);

xx, yy =np.meshgrid(x, y)

zz = np.zeros(xx.size)

field = np.random.rand(xx.size)
pts = np.empty((xx.size, 3), dtype=float)
pts[..., 0] = xx.reshape(xx.size)
pts[..., 1] = yy.reshape(xx.size)
pts[..., 2] = zz.reshape(xx.size)


# ugrid = tvtk.StructuredGrid(dimensions=(xx.shape[0], xx.shape[1], 1), points=pts) # create a structured grid object

ugrid = tvtk.UnstructuredGrid(points=pts) # create a structured grid object
# tets = np.arange(xx.size)
# tet_type = tvtk.Tetra().cell_type
# ug = tvtk.UnstructuredGrid(points=points)
# ugrid.set_cells(tet_type, np.array([tets]))


# ugrid.point_data.scalars = (field).ravel(order='F') # assign snapshot value
ugrid.point_data.scalars = field # assign snapshot value

ugrid.point_data.scalars.name = 'vz' # name it

outfname='./test_4_pts_unstructured.vtk'
# outfname='./test_4_pts_structured.vtk'
write_data(ugrid, outfname)

# dset.close()
