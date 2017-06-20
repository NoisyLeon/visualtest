import numpy as np


def write_unstructured_vtk_irregular(fname, x, y, z, v, dataname='scalars'):
    """
    """
    N = x.size
    with open(fname, 'wb') as fid:
        # header information
        fid.write('# vtk DataFile Version 4.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        fid.write('POINTS '+str(N)+' float\n')
        # grid coordinate
        for i in xrange(N): fid.write(str(x[i])+' '+str(y[i])+' '+str(z[i])+'\n')
        # connectivity
        fid.write('CELL_TYPES 0\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS ' +dataname+' float\n')
        fid.write('LOOKUP_TABLE default\n')
        for i in xrange(N): fid.write(str(v[i])+'\n')
        return

def write_polydata_vtk(fname, x, y, z, v, dataname='scalars'):
    """
    """
    N = x.size
    with open(fname, 'wb') as fid:
        # header information
        fid.write('# vtk DataFile Version 4.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET POLYDATA\n')
        fid.write('POINTS '+str(N)+' float\n')
        # grid coordinate
        for i in xrange(N): fid.write(str(x[i])+' '+str(y[i])+' '+str(z[i])+'\n')
        # connectivity
        fid.write('VERTICES %d %d\n' %(N, 2*N))
        for i in xrange(N): fid.write('1 %d\n' %i)
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS ' +dataname+' float\n')
        fid.write('LOOKUP_TABLE default\n')
        for i in xrange(N): fid.write(str(v[i])+'\n')
        return

def write_unstructured_vtk(fname, x, y, z, v, dataname='scalars', celltype=11):
    """
    celltype:
    http://www.vtk.org/doc/nightly/html/vtkCellType_8h_source.html
    """
    N = x.size*y.size*z.size # total number of grid
    nx=x.size; ny=y.size; nz=z.size
    with open(fname, 'wb') as fid:
        # header information
        fid.write('# vtk DataFile Version 4.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET UNSTRUCTURED_GRID\n')
        fid.write('POINTS '+str(N)+' float\n')
        # grid coordinate
        for i in xrange(nx):
            for j in xrange(ny):
                for k in xrange(nz):
                    fid.write(str(x[i])+' '+str(y[j])+' '+str(z[k])+'\n')
        # connectivity
        if celltype!=-1:
            n_cells=(nx-1)*(ny-1)*(nz-1)
            fid.write('\n')
            # CELLS n1 n2
            # n1: number of cells n2: number of cell list
            # e.g. 8 0 1 2 3 10 11 12 13
            # means this cell is connecting 8 points, with id 0 1 2 3 10 11 12 13
            fid.write('CELLS '+str(n_cells)+' '+str(9*n_cells)+'\n') 
            count=nx*ny*nz
            for i in range(1, nx):
                for j in range(1, ny):
                    for k in range(1, nz):
                        a=k+(j-1)*nz+(i-1)*ny*nz-1
                        b=k+(j-1)*nz+(i-1)*ny*nz 
                        c=k+(j)*nz+(i-1)*ny*nz-1
                        d=k+(j)*nz+(i-1)*ny*nz
                        e=k+(j-1)*nz+(i)*ny*nz-1
                        f=k+(j-1)*nz+(i)*ny*nz
                        g=k+(j)*nz+(i)*ny*nz-1
                        h=k+(j)*nz+(i)*ny*nz
                        fid.write('8 '+str(a)+' '+str(b)+' '+str(c)+' '+str(d)+' '+str(e)+' '+str(f)+' '+str(g)+' '+str(h)+'\n')
            #cell types
            fid.write('\n')
            fid.write('CELL_TYPES '+str(n_cells)+'\n')
            for i in xrange(nx-1):
                for j in xrange(ny-1):
                    for k in xrange(nz-1):
                        fid.write(str(celltype)+'\n')
        else: fid.write('CELL_TYPES 0\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS ' +dataname+' float\n')
        fid.write('LOOKUP_TABLE default\n')
        for i in xrange(nx):
            for j in xrange(ny):
                for k in xrange(nz):
                    fid.write(str(v[i,j,k])+'\n')
        return
    
def write_structured_vtk(fname, x, y, z, v, dataname='scalars'):
    """
    v.shape = nx, ny, nz
    """
    N = x.size*y.size*z.size # total number of grid
    nx=x.size; ny=y.size; nz=z.size
    with open(fname, 'wb') as fid:
        # header information
        fid.write('# vtk DataFile Version 4.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET STRUCTURED_GRID\n')
        fid.write('DIMENSIONS %d %d %d' %(nx, ny, nz))
        fid.write('POINTS '+str(N)+' float\n')
        # grid coordinate
        for k in xrange(nz):
            for j in xrange(ny):
                for i in xrange(nx):
                    fid.write(str(x[i])+' '+str(y[j])+' '+str(z[k])+'\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS ' +dataname+' float\n')
        fid.write('LOOKUP_TABLE default\n')
        for k in xrange(nz):
            for j in xrange(ny):
                for i in xrange(nx):
                    fid.write(str(v[i,j,k])+'\n')
    return

def write_rectilinear_vtk(fname, x, y, z, v, dataname='scalars'):
    """
    v.shape = nx, ny, nz
    """
    N = x.size*y.size*z.size # total number of grid
    nx=x.size; ny=y.size; nz=z.size
    with open(fname, 'wb') as fid:
        # header information
        fid.write('# vtk DataFile Version 4.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET RECTILINEAR_GRID\n')
        fid.write('DIMENSIONS %d %d %d' %(nx, ny, nz))
        # grid coordinate
        fid.write('X_COORDINATES %d double\n' %nx)
        for i in xrange(nx): fid.write(str(x[i])+'\n')
        fid.write('Y_COORDINATES %d double\n' %ny)
        for j in xrange(ny): fid.write(str(y[j])+'\n')
        fid.write('Z_COORDINATES %d double\n' %nz)
        for k in xrange(nz): fid.write(str(z[k])+'\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS ' +dataname+' float\n')
        fid.write('LOOKUP_TABLE default\n')
        for k in xrange(nz):
            for j in xrange(ny):
                for i in xrange(nx):
                    fid.write(str(v[i,j,k])+'\n')
    return



import os
# 
# """test 1
# 3D box with random values, 5 types of vtk grid
# The z grid points are shifted so that the three 3D box can be shown simutaneously.
# 
# unstructured_irregular_test_1.vtk can only be visualized by selecting Point Gaussian and change Gaussisian Radius to be 0.1
# """
# outdir='./example_1_dir/'
# if not os.path.isdir(outdir): os.makedirs(outdir)
# x= np.arange(0, 3, 0.5)
# y= np.arange(0, 5, 0.25)
# z= np.arange(0, 2, 1)
# data = np.random.rand(x.size, y.size, z.size)
# 
# write_unstructured_vtk(outdir+'unstructured.vtk', x,y,z, data)
# write_structured_vtk(outdir+'structured.vtk', x,y,z+2, data)
# write_rectilinear_vtk(outdir+'rectilinear.vtk', x,y,z+4, data)
# 
# xx, yy, zz1 =np.meshgrid(x, y, z+6, indexing='ij')
# xx, yy, zz2 =np.meshgrid(x, y, z+8, indexing='ij')
# N=xx.size
# write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', xx.reshape(N), yy.reshape(N) ,zz1.reshape(N), data.reshape(N))
# write_polydata_vtk(outdir+'polydata.vtk', xx.reshape(N), yy.reshape(N) ,zz2.reshape(N), data.reshape(N))


# 
# """test 2
# 3D box with random values, 5 types of vtk grid
# x = [0, 1, 1, 2, 3]
# the points correspond to the first 1 is set to -1, others are all 1
# """
# outdir='./example_2_dir/'
# if not os.path.isdir(outdir): os.makedirs(outdir)
# x= np.arange(1, 4, 1)
# y= np.arange(0, 5, 0.25)
# z= np.arange(0, 2, 1)
# x = np.append(1., x)
# x = np.append(0., x)
# data = np.ones((x.size, y.size, z.size))
# data[1, :, :] = -1.
# 
# write_unstructured_vtk(outdir+'unstructured.vtk', x,y,z, data)
# write_structured_vtk(outdir+'structured.vtk', x,y,z+2, data)
# write_rectilinear_vtk(outdir+'rectilinear.vtk', x,y,z+4, data)
# 
# xx, yy, zz1 =np.meshgrid(x, y, z+6, indexing='ij')
# xx, yy, zz2 =np.meshgrid(x, y, z+8, indexing='ij')
# N=xx.size
# write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', xx.reshape(N), yy.reshape(N) ,zz1.reshape(N), data.reshape(N))
# write_polydata_vtk(outdir+'polydata.vtk', xx.reshape(N), yy.reshape(N) ,zz2.reshape(N), data.reshape(N))
# 
# 
# 
# """test 3
# 3D box with random values, 5 types of vtk grid
# x = [0, 1, 1, 2, 3]
# the points correspond to the second 1 is set to -1, others are all 1
# """
# outdir='./example_3_dir/'
# if not os.path.isdir(outdir): os.makedirs(outdir)
# x= np.arange(1, 4, 1)
# y= np.arange(0, 5, 0.25)
# z= np.arange(0, 2, 1)
# x = np.append(1., x)
# x = np.append(0., x)
# data = np.ones((x.size, y.size, z.size))
# data[2, :, :] = -1.
# 
# write_unstructured_vtk(outdir+'unstructured.vtk', x,y,z, data)
# write_structured_vtk(outdir+'structured.vtk', x,y,z+2, data)
# write_rectilinear_vtk(outdir+'rectilinear.vtk', x,y,z+4, data)
# 
# xx, yy, zz1 =np.meshgrid(x, y, z+6, indexing='ij')
# xx, yy, zz2 =np.meshgrid(x, y, z+8, indexing='ij')
# N=xx.size
# write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', xx.reshape(N), yy.reshape(N) ,zz1.reshape(N), data.reshape(N))
# write_polydata_vtk(outdir+'polydata.vtk', xx.reshape(N), yy.reshape(N) ,zz2.reshape(N), data.reshape(N))
# 

"""test 4
Plot a circle
"""
outdir='./example_4_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
theta   = np.arange(0, 360, 1)/180.*np.pi
r       = np.arange(0, 1.1, 0.1)
z       = np.array([0.])
theta2, r2, z= np.meshgrid(theta, r, z.size, indexing='ij')
x       = r2*np.cos(theta2)
y       = r2*np.sin(theta2)

data    = np.random.rand(theta.size, r.size, 1)

# 
# write_unstructured_vtk(outdir+'unstructured.vtk', x,y,z, data)
# write_structured_vtk(outdir+'structured.vtk', x,y,z+2, data)
# write_rectilinear_vtk(outdir+'rectilinear.vtk', x,y,z+4, data)
# 
# xx, yy, zz1 =np.meshgrid(x, y, z+6, indexing='ij')
# xx, yy, zz2 =np.meshgrid(x, y, z+8, indexing='ij')
N=x.size
write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', x.reshape(N), y.reshape(N) ,z.reshape(N), data.reshape(N))
write_polydata_vtk(outdir+'polydata.vtk', x.reshape(N), y.reshape(N) ,z.reshape(N), data.reshape(N))


# import h5py
# outdir='./example_3_dir/'
# if not os.path.isdir(outdir): os.makedirs(outdir)
# dset = h5py.File('./snapshot_for_kuangdai.h5')
# subgroup = dset['vz/14000'] # vertical velocity for iteration of 14000
# Radius = 1.
# theta=np.array([]); phi=np.array([]); r=np.array([]); data = np.array([])
# # subgroup.keys() : processor id
# # e.g. subgroup['359'] : data for processor id=359
# # loop over processor id to get data
# for key in subgroup.keys():
#     subdset = subgroup[key]
#     data   = np.append(data, (subdset[...]))
#     theta1  = subdset.attrs['theta']
#     phi1    = subdset.attrs['phi']
#     theta1, phi1 = np.meshgrid(theta1, phi1, indexing='ij')
#     theta   = np.append(theta, theta1)
#     phi     = np.append(phi, phi1)
# # convert spherical coordinate to 3D Cartesian coordinate
# x = Radius * np.sin(theta) * np.cos(phi)
# y = Radius * np.sin(theta) * np.sin(phi)
# z = Radius * np.cos(theta)
# 
# write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', x,y,z, data)
# write_polydata_vtk(outdir+'polydata.vtk',  x,y,z, data)
# dset.close()

