import numpy as np
import numba

# @numba.jit
def write_unstructured_vtk(fname, x, y, z, v, dataname='scalars', celltype=9):
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
        if celltype!=0:
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
        fid.write('LOOKUP_TABLE mytable\n')
        for i in xrange(nx):
            for j in xrange(ny):
                for k in xrange(nz):
                    fid.write(str(v[i,j,k])+'\n')
        return

def write_structured_vtk(fname, x, y, z, v, dataname='scalars', connectivity=True):
    N = x.size*y.size*z.size # total number of grid
    nx=x.size; ny=y.size; nz=z.size
    with open(fname, 'wb') as fid:
        # header information
        fid.write('# vtk DataFile Version 4.0\n')
        fid.write('vtk output\n')
        fid.write('ASCII\n')
        fid.write('DATASET STRUCTURED_GRID\n')
        fid.write('POINTS '+str(N)+' float\n')
        # grid coordinate
        for i in xrange(nx):
            for j in xrange(ny):
                for k in xrange(nz):
                    fid.write(str(x[i])+' '+str(y[j])+' '+str(z[k])+'\n')
        # connectivity
        if connectivity:
            n_cells=(nx-1)*(ny-1)*(nz-1)
            fid.write('\n')
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
        else: fid.write('CELL_TYPES 0\n')
        for i in xrange(nx-1):
            for j in xrange(ny-1):
                for k in xrange(nz-1):
                    fid.write('11\n')
        #- write data
        fid.write('\n')
        fid.write('POINT_DATA '+str(N)+'\n')
        fid.write('SCALARS ' +dataname+' float\n')
        fid.write('LOOKUP_TABLE mytable\n')
        for i in xrange(nx):
            for j in xrange(ny):
                for k in xrange(nz):
                    fid.write(str(v[i,j,k])+'\n')

"""test 1
"""
x= np.arange(0, 3, 1);
x=np.append(0, x)
x=np.append(0, x)
y= np.arange(0, 5, 1);
z=np.array([0, 1])
xx, yy, zz =np.meshgrid(x, y, z, indexing='ij')

# field = np.random.rand(xx.shape[0], xx.shape[1], xx.shape[2])
field = np.ones(xx.shape)
field[0, :, :] = -1
field[1, :, :] = -1000
# field[2, :, :] = -1

field.reshape(x.size, y.size, z.size)

write_unstructured_vtk('unstructured.vtk', x,y,z,field)


