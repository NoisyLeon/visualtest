
# Author: Prabhu Ramachandran <prabhu@aero.iitb.ac.in>
# Copyright (c) 2007 Prabhu Ramachandran.
# License: BSD Style.

import numpy as np
from mayavi.scripts import mayavi2
from tvtk.api import tvtk, write_data


pd = tvtk.PolyData()

x= np.arange(0, 10, 1)
y= np.arange(0, 5, 1)
z= np.arange(0, 3, 1)
xx, yy, zz =np.meshgrid(x, y, z, indexing='ij')
xx=xx.reshape(xx.size)
yy=yy.reshape(xx.size)
zz=zz.reshape(xx.size)


pd.points = np.array([xx, yy, zz]).T
verts = np.arange(0, xx.size, 1)
verts.shape = (xx.size, 1)
data = np.random.rand(x.size, y.size, z.size)
pd.verts = verts
pd.point_data.scalars = data.reshape(xx.size)
pd.point_data.scalars.name = 'scalars'
outfname='./polydata.vtk'
# outfname='./test_4_pts_structured.vtk'
write_data(pd, outfname)
# # @mayavi2.standalone
# def main():
#     # Create some random points to view.
#     pd = tvtk.PolyData()
#     pd.points = np.random.random((1000, 3))
#     verts = np.arange(0, 1000, 1)
#     verts.shape = (1000, 1)
#     pd.verts = verts
#     pd.point_data.scalars = np.random.random(1000)
#     pd.point_data.scalars.name = 'scalars'
#     outfname='./polydata.vtk'
#     # outfname='./test_4_pts_structured.vtk'
#     write_data(pd, outfname)
#     # 
#     # # Now visualize it using mayavi2.
#     # from mayavi.sources.vtk_data_source import VTKDataSource
#     # from mayavi.modules.outline import Outline
#     # from mayavi.modules.surface import Surface
#     # 
#     # mayavi.new_scene()
#     # d = VTKDataSource()
#     # d.data = pd
#     # mayavi.add_source(d)
#     # mayavi.add_module(Outline())
#     # s = Surface()
#     # mayavi.add_module(s)
#     # s.actor.property.set(representation='p', point_size=2)
#     # # You could also use glyphs to render the points via the Glyph module.
# 
# if __name__ == '__main__':
#     main()