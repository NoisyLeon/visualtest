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

def polar2cartesian(r, theta, z):
    """ Generate points for structured grid for a cylindrical annular
        volume.  This method is useful for generating a unstructured
        cylindrical mesh for VTK.
    """
    # Find the x values and y values for each plane.
    N=r.size*theta.size
    x_plane = (np.tensordot(r, np.cos(theta), axes=0)).reshape(N, order='F')
    y_plane = (np.tensordot(r, np.sin(theta), axes=0)).reshape(N, order='F')

    # Allocate an array for all the points.  We'll have len(x_plane)
    # points on each plane, and we have a plane for each z value, so
    # we need len(x_plane)*len(z) points.
    points = np.empty([len(x_plane)*len(z),3])

    # Loop through the points for each plane and fill them with the
    # correct x,y,z values.
    start = 0
    for z_plane in z:
        end = start+len(x_plane)
        # slice out a plane of the output points and fill it
        # with the x,y, and z values for this plane.  The x,y
        # values are the same for every plane.  The z value
        # is set to the current z
        plane_points = points[start:end]
        plane_points[:,0] = x_plane
        plane_points[:,1] = y_plane
        plane_points[:,2] = z_plane
        start = end
    return points


def write_structured_vtk_cylindrical(fname, r, theta, z, v, dataname='scalars'):
    """
    v.shape = nr, ntheta, nz
    """
    pts     = polar2cartesian(r, theta, z)
    nr      = r.size; ntheta = theta.size; nz = z.size
    N       = nr*ntheta*nz
    with open(fname, 'wb') as fid:
        # header information
        fid.write('# vtk DataFile Version 4.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET STRUCTURED_GRID\n')
        fid.write('DIMENSIONS %d %d %d' %(nr, ntheta, nz))
        fid.write('POINTS '+str(N)+' float\n')
        # grid coordinate
        for i in xrange(N): fid.write(str(pts[i, 0])+' '+str(pts[i, 1])+' '+str(pts[i, 2])+'\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS ' +dataname+' float\n')
        fid.write('LOOKUP_TABLE default\n')
        for k in xrange(nz):
            for j in xrange(ntheta):
                for i in xrange(nr):
                    fid.write(str(v[i,j,k])+'\n')
    return

def spherical2cartesian(r, theta, phi):
    """ Generate points for structured grid for a cylindrical annular
        volume.  This method is useful for generating a unstructured
        cylindrical mesh for VTK.
    """
    # Find the x values and y values for each plane.
    N=r.size*theta.size*phi.size
    r1, theta1, phi1 = np.meshgrid(r, theta, phi, indexing='ij')
    x = (r1 * np.sin(theta1) * np.cos(phi1)).reshape(N, order='F')
    y = (r1 * np.sin(theta1) * np.sin(phi1)).reshape(N, order='F')
    z = (r1 * np.cos(theta1)).reshape(N, order='F')
    points = np.empty([N,3])
    points[:, 0] = x; points[:, 1] = y; points[:, 2] = z
    return points

def write_structured_vtk_sphere(fname, r, theta, phi, v, dataname='scalars'):
    """
    v.shape = nr, ntheta, nz
    """
    pts     = spherical2cartesian(r, theta, phi)
    nr      = r.size; ntheta = theta.size; nphi = phi.size
    N       = nr*ntheta*nphi
    with open(fname, 'wb') as fid:
        # header information
        fid.write('# vtk DataFile Version 4.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET STRUCTURED_GRID\n')
        fid.write('DIMENSIONS %d %d %d' %(nr, ntheta, nphi))
        fid.write('POINTS '+str(N)+' float\n')
        # grid coordinate
        for i in xrange(N): fid.write(str(pts[i, 0])+' '+str(pts[i, 1])+' '+str(pts[i, 2])+'\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS ' +dataname+' float\n')
        fid.write('LOOKUP_TABLE default\n')
        for k in xrange(nphi):
            for j in xrange(ntheta):
                for i in xrange(nr):
                    fid.write(str(v[i,j,k])+'\n')
    return


def spherical2cartesian_2(fname, r, theta, phi, v, dataname='scalars'):
    """
    v.size = nr, ntheta, nz
    """
    pts     = spherical2cartesian(r, theta, phi)
    nr      = r.size; ntheta = theta.size; nphi = phi.size
    N       = nr*ntheta*nphi
    print 'here'
    with open(fname, 'wb') as fid:
        # header information
        fid.write('# vtk DataFile Version 4.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET STRUCTURED_GRID\n')
        fid.write('DIMENSIONS %d %d %d' %(nr, ntheta, nphi))
        fid.write('POINTS '+str(N)+' float\n')
        # grid coordinate
        for i in xrange(N): fid.write(str(pts[i, 0])+' '+str(pts[i, 1])+' '+str(pts[i, 2])+'\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS ' +dataname+' float\n')
        fid.write('LOOKUP_TABLE default\n')
        for i in xrange(N): fid.write(str(v[i])+'\n')
    return

def write_structured_vtk_sphere_2(fname, r, theta, phi, v, dataname='scalars'):
    """
    v.size = nr, ntheta, nz
    """
    pts     = spherical2cartesian(r, theta, phi)
    nr      = r.size; ntheta = theta.size; nphi = phi.size
    N       = nr*ntheta*nphi
    with open(fname, 'wb') as fid:
        # header information
        fid.write('# vtk DataFile Version 4.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET STRUCTURED_GRID\n')
        fid.write('DIMENSIONS %d %d %d' %(nr, ntheta, nphi))
        fid.write('POINTS '+str(N)+' float\n')
        # grid coordinate
        for i in xrange(N): fid.write(str(pts[i, 0])+' '+str(pts[i, 1])+' '+str(pts[i, 2])+'\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS ' +dataname+' float\n')
        fid.write('LOOKUP_TABLE default\n')
        for i in xrange(N): fid.write(str(v[i])+'\n')
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
        fid.write('DIMENSIONS %d %d %d\n' %(nx, ny, nz))
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

"""test 1
3D box with random values, 5 types of vtk grid
The z grid points are shifted so that the three 3D box can be shown simutaneously.

unstructured_irregular_test_1.vtk can only be visualized by selecting Point Gaussian and change Gaussisian Radius to be 0.1
"""
print 'running example 1'
outdir='./example_1_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
x= np.arange(0, 3, 0.5)
y= np.arange(0, 5, 0.25)
z= np.arange(0, 2, 1)
data = np.random.rand(x.size, y.size, z.size)

write_unstructured_vtk(outdir+'unstructured.vtk', x,y,z, data)
write_structured_vtk(outdir+'structured.vtk', x,y,z+2, data)
write_rectilinear_vtk(outdir+'rectilinear.vtk', x,y,z+4, data)

xx, yy, zz1 =np.meshgrid(x, y, z+6, indexing='ij')
xx, yy, zz2 =np.meshgrid(x, y, z+8, indexing='ij')
N=xx.size
write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', xx.reshape(N), yy.reshape(N) ,zz1.reshape(N), data.reshape(N))
write_polydata_vtk(outdir+'polydata.vtk', xx.reshape(N), yy.reshape(N) ,zz2.reshape(N), data.reshape(N))



"""test 2
5 types of vtk grid, repeated grid points for x = 1
x = [0, 1, 1, 2, 3]
the points correspond to the FIRST 1 is set to -1, others are all 1
"""
print 'running example 2'
outdir='./example_2_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
x= np.arange(1, 4, 1)
y= np.arange(0, 5, 0.25)
z= np.arange(0, 2, 1)
x = np.append(1., x)
x = np.append(0., x)
data = np.ones((x.size, y.size, z.size))
data[1, :, :] = -1.
# data[0, :, :] = -1.  # discontinuity

write_unstructured_vtk(outdir+'unstructured.vtk', x,y,z, data)
write_structured_vtk(outdir+'structured.vtk', x,y,z+2, data)
write_rectilinear_vtk(outdir+'rectilinear.vtk', x,y,z+4, data)

xx, yy, zz1 =np.meshgrid(x, y, z+6, indexing='ij')
xx, yy, zz2 =np.meshgrid(x, y, z+8, indexing='ij')
N=xx.size
write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', xx.reshape(N), yy.reshape(N) ,zz1.reshape(N), data.reshape(N))
write_polydata_vtk(outdir+'polydata.vtk', xx.reshape(N), yy.reshape(N) ,zz2.reshape(N), data.reshape(N))


"""test 3
5 types of vtk grid
x = [0, 1, 1, 2, 3]
the points correspond to the SECOND 1 is set to -1, others are all 1
"""
print 'running example 3'
outdir='./example_3_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
x= np.arange(1, 4, 1)
y= np.arange(0, 5, 0.25)
z= np.arange(0, 2, 1)
x = np.append(1., x)
x = np.append(0., x)
data = np.ones((x.size, y.size, z.size))
data[2, :, :] = -1.

write_unstructured_vtk(outdir+'unstructured.vtk', x,y,z, data)
write_structured_vtk(outdir+'structured.vtk', x,y,z+2, data)
write_rectilinear_vtk(outdir+'rectilinear.vtk', x,y,z+4, data)

xx, yy, zz1 =np.meshgrid(x, y, z+6, indexing='ij')
xx, yy, zz2 =np.meshgrid(x, y, z+8, indexing='ij')
N=xx.size
write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', xx.reshape(N), yy.reshape(N) ,zz1.reshape(N), data.reshape(N))
write_polydata_vtk(outdir+'polydata.vtk', xx.reshape(N), yy.reshape(N) ,zz2.reshape(N), data.reshape(N))


"""test 4
Plot a cylinder
unstructured_irregular.vtk can only be visualized by choosing point Gaussian
"""
print 'running example 4'
outdir='./example_4_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
theta   = np.arange(0, 361, 1)/180.*np.pi
r       = np.arange(0.1, 1.1, 0.1)
z       = np.array([0., 1.])
# z       = np.array([0.]) # slice

# 
data    = np.random.rand(r.size, theta.size, z.size)
write_structured_vtk_cylindrical(outdir+'structured.vtk', r, theta, z, data)

pts = polar2cartesian(r, theta, z)
x=pts[:,0]; y=pts[:, 1]; z=pts[:, 2]
N=x.size
write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', x, y, z+2, data.reshape(N, order='F'))
write_polydata_vtk(outdir+'polydata.vtk', x, y, z+4, data.reshape(N, order='F'))

"""test 5
Plot a circle with discontinuity, r = 0.5 is repeated
unstructured_irregular.vtk can only be visualized by choosing point Gaussian
"""
print 'running example 5'
outdir='./example_5_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
theta   = np.arange(0, 361, 1)/180.*np.pi
r1      = np.arange(0.5, 1.1, 0.1); r2 = np.arange(0.1, 0.6, 0.1);
r       = np.append(r2, r1)
z       = np.array([0.])
# 
# # 
data    = np.ones((r.size, theta.size, z.size))
data[:5, :,:,] = -1.
write_structured_vtk_cylindrical(outdir+'structured.vtk', r, theta, z, data)
# 
pts = polar2cartesian(r, theta, z)
x=pts[:,0]; y=pts[:, 1]; z=pts[:, 2]
N=x.size
write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', x, y, z+2, data.reshape(N, order='F'))
write_polydata_vtk(outdir+'polydata.vtk', x, y, z+4, data.reshape(N, order='F'))

"""test 6
Plot a circle with discontinuity, repeated twice
unstructured_irregular.vtk can only be visualized by choosing point Gaussian
"""
print 'running example 6'
outdir='./example_6_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
theta   = np.arange(0, 361, 1)/180.*np.pi
r1      = np.arange(0.5, 1.1, 0.1); r2 = np.arange(0.1, 0.6, 0.1);
r       = np.append(r2, 0.5)
r       = np.append(r, r1)
z       = np.array([0.])
# 
# # 
data    = np.ones((r.size, theta.size, z.size))
data[:5, :,:,] = -1.
data[5, :,:,] = 0.
write_structured_vtk_cylindrical(outdir+'structured.vtk', r, theta, z, data)
# 
pts = polar2cartesian(r, theta, z)
x=pts[:,0]; y=pts[:, 1]; z=pts[:, 2]
N=x.size
write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', x, y, z+2, data.reshape(N, order='F'))
write_polydata_vtk(outdir+'polydata.vtk', x, y, z+4, data.reshape(N, order='F'))


"""test 7
Plot a circle with discontinuity, TWO files
unstructured_irregular.vtk can only be visualized by choosing point Gaussian
"""
print 'running example 7'
outdir='./example_7_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
theta   = np.arange(0, 361, 1)/180.*np.pi
r1      = np.arange(0.5, 1.1, 0.1); r2 = np.arange(0.1, 0.6, 0.1);
z       = np.array([0.])
# 
# # 
data1   = np.ones((r1.size, theta.size, z.size))
data2   = -np.ones((r2.size, theta.size, z.size))

write_structured_vtk_cylindrical(outdir+'structured_1.vtk', r1, theta, z, data1)
write_structured_vtk_cylindrical(outdir+'structured_2.vtk', r2, theta, z, data2)
# 
pts = polar2cartesian(r1, theta, z)
x1=pts[:,0]; y1=pts[:, 1]; z1=pts[:, 2]
N=x1.size
write_unstructured_vtk_irregular(outdir+'unstructured_irregular_1.vtk', x1, y1, z1+2, data1.reshape(N, order='F'))
write_polydata_vtk(outdir+'polydata_1.vtk', x1, y1, z1+4, data1.reshape(N, order='F'))

pts = polar2cartesian(r2, theta, z)
x2=pts[:,0]; y2=pts[:, 1]; z2=pts[:, 2]
N=x2.size
write_unstructured_vtk_irregular(outdir+'unstructured_irregular_2.vtk', x2, y2, z2+2, data2.reshape(N, order='F'))
write_polydata_vtk(outdir+'polydata_2.vtk', x2, y2, z2+4, data2.reshape(N, order='F'))


"""test 8
Plot a sphere
unstructured_irregular.vtk can only be visualized by choosing point Gaussian
"""
print 'running example 8'
outdir='./example_8_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
theta   = np.arange(0, 181, 1)/180.*np.pi
phi     = np.arange(0, 361, 1)/180.*np.pi
r       = np.array([1.])
#
data    = np.random.rand(r.size, theta.size, phi.size)
write_structured_vtk_sphere(outdir+'structured.vtk', r, theta, phi, data)
# 
pts = spherical2cartesian(r, theta, phi)
x=pts[:,0]; y=pts[:, 1]; z=pts[:, 2]
N=x.size
write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', x, y, z+2, data.reshape(N, order='F'))
write_polydata_vtk(outdir+'polydata.vtk', x, y, z+4, data.reshape(N, order='F'))


"""test 9
Plot a wavefield snapshot
unstructured_irregular.vtk can only be visualized by choosing point Gaussian
"""
print 'running example 9'
import h5py
outdir='./example_9_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
dset = h5py.File('./snapshot_for_kuangdai.h5')
subgroup = dset['vz/14000'] # vertical velocity for iteration of 14000
Radius = 1.
theta=np.array([]); phi=np.array([]); r=np.array([]); data = np.array([])
# subgroup.keys() : processor id
# e.g. subgroup['359'] : data for processor id=359
# loop over processor id to get data
for key in subgroup.keys()[:5]:
    subdset = subgroup[key]
    data    = np.append(data, (subdset[...]))
    theta1  = subdset.attrs['theta']
    phi1    = subdset.attrs['phi']
    theta2, phi2 = np.meshgrid(theta1, phi1, indexing='ij')
    theta   = np.append(theta, theta2)
    phi     = np.append(phi, phi2)
    
# convert spherical coordinate to 3D Cartesian coordinate
x = Radius * np.sin(theta) * np.cos(phi)
y = Radius * np.sin(theta) * np.sin(phi)
z = Radius * np.cos(theta)

write_unstructured_vtk_irregular(outdir+'unstructured_irregular.vtk', x,y,z, data)
write_polydata_vtk(outdir+'polydata.vtk',  x,y,z, data)


i=0
for key in subgroup.keys()[:5]:
    i+=1
    subdset = subgroup[key]
    theta   = subdset.attrs['theta']
    phi     = subdset.attrs['phi']
    data    = subdset[...]
    r       = np.array([Radius])
    N=phi.size*theta.size
    write_structured_vtk_sphere_2(outdir+'structured.%d.vtk' %i, r, theta, phi, data.reshape(N, order='F'))
dset.close()


"""test 10
Plot a wavefield snapshot
unstructured_irregular.vtk can only be visualized by choosing point Gaussian
"""
print 'running example 10'
import h5py
outdir='./example_10_dir/'
if not os.path.isdir(outdir): os.makedirs(outdir)
dset    = h5py.File('./snapshot_for_kuangdai.h5')
subgroup= dset['vz/14000'] # vertical velocity for iteration of 14000
Radius  = 1.
for j in xrange(10):
    i=0
    for key in subgroup.keys()[:5]:
        i+=1
        subdset = subgroup[key]
        theta   = subdset.attrs['theta']
        phi     = subdset.attrs['phi']
        data    = subdset[...] + j*2.*1e-12
        r       = np.array([Radius])
        N=phi.size*theta.size
        write_structured_vtk_sphere_2(outdir+'structured.%d.%d.vtk' %(i,j), r, theta, phi, data.reshape(N, order='F'))
dset.close()


    


