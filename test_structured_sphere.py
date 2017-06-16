from sympy.ntheory import primefactors
from tvtk.api import tvtk, write_data
import h5py
import numpy as np


Radius = 1.

theta=np.arange(0,90.,1)/180.*np.pi; phi=np.arange(0,90,1.)/180.*np.pi


theta1, phi1 = np.meshgrid(theta, phi, indexing='ij') 



x = Radius * np.sin(theta1) * np.cos(phi1)
y = Radius * np.sin(theta1) * np.sin(phi1)
z = Radius * np.cos(theta1)

field = np.random.rand(theta1.shape[0], theta1.shape[1])
dims = (theta1.shape[0], theta1.shape[1], 1)



pts = np.empty(z.shape + (3,), dtype=float)
pts[..., 0] = x; pts[..., 1] = y; pts[..., 2] = z # assign grid point position

# pts = pts.transpose(2, 1, 0, 3).copy()
pts.shape = pts.size / 3, 3

# sgrid = tvtk.StructuredGrid(dimensions=dims, points=pts) # create a structured grid object
sgrid = tvtk.UnstructuredGrid(points=pts) # create a structured grid object
sgrid.point_data.scalars = (field).ravel(order='F') # assign snapshot value

sgrid.point_data.scalars.name = 'vz' # name it

outfname='./test_sphere.vtk'
write_data(sgrid, outfname)

# dset.close()