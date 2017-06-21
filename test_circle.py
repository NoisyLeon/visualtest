from numpy import pi, cos, sin, empty, linspace, random, ravel
from tvtk.api import tvtk,  write_data

def generate_annulus(r, theta, z):
    """ Generate points for structured grid for a cylindrical annular
        volume.  This method is useful for generating a unstructured
        cylindrical mesh for VTK.
    """
    # Find the x values and y values for each plane.
    x_plane = (cos(theta)*r[:,None]).ravel()
    y_plane = (sin(theta)*r[:,None]).ravel()

    # Allocate an array for all the points.  We'll have len(x_plane)
    # points on each plane, and we have a plane for each z value, so
    # we need len(x_plane)*len(z) points.
    points = empty([len(x_plane)*len(z),3])

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

dims = (3, 4, 3)
r = linspace(5, 15, dims[0])
theta = linspace(0, 0.5*pi, dims[1])
z = linspace(0, 10, dims[2])
pts = generate_annulus(r, theta, z)
sgrid = tvtk.StructuredGrid(dimensions=(dims[1], dims[0], dims[2]))
sgrid.points = pts
s = random.random((dims[0]*dims[1]*dims[2]))
sgrid.point_data.scalars = ravel(s.copy())
sgrid.point_data.scalars.name = 'scalars'

outfname='./test_circle.vtk'
write_data(sgrid, outfname)