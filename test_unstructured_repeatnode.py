from sympy.ntheory import primefactors
from tvtk.api import tvtk, write_data
import h5py
import numpy as np

# dset = h5py.File('./snapshot_for_kuangdai.h5')

# subgroup = dset['vz/14000'] # vertical velocity for iteration of 14000
Radius = 1.
# theta, phi, r: data point for spherical coordinate
# field : data array
theta=np.arange(0,90.,1)/180.*np.pi; phi=np.arange(0,90,1.)/180.*np.pi


theta1, phi1 = np.meshgrid(theta, phi, indexing='ij') 

theta1=theta1.reshape(theta1.size)
phi1=phi1.reshape(phi1.size)


x = Radius * np.sin(theta1) * np.cos(phi1)
y = Radius * np.sin(theta1) * np.sin(phi1)
z = Radius * np.cos(theta1)

field = np.random.rand(theta.size*phi.size)
least_prime=primefactors(theta1.size)[0]

dims = (field.size/least_prime, least_prime, 1)
### Note: Ideally dims should be exactly (Nx, Ny, Nz), however, another process of sorting and reshaping the field data is needed, which can be time-consuming

pts = np.empty(z.shape + (3,), dtype=float)
pts[..., 0] = x; pts[..., 1] = y; pts[..., 2] = z # assign grid point position
# sgrid = tvtk.StructuredGrid(dimensions=dims, points=pts) # create a structured grid object
sgrid = tvtk.UnstructuredGrid(points=pts) # create a structured grid object
sgrid.point_data.scalars = (field).ravel(order='F') # assign snapshot value

sgrid.point_data.scalars.name = 'vz' # name it

outfname='./test.vtk'
write_data(sgrid, outfname)

# dset.close()