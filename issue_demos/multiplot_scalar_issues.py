##
import copy

import pyvista as pv

sp1 = pv.Sphere(1)
# color index toi the value of the x coordinate
scalars = sp1.points[:,0]
# set two different c-lims
clim1 = [-1,1]
clim2 = [-.2,.2]
cmap1 = 'jet'
cmap2 = 'bwr'
##

# plot with clim1
pl = pv.Plotter()
pl.add_mesh(mesh=sp1.copy(True),scalars=scalars.copy(),clim=clim1)
pl.show()
##
# plot with clim2
pl2 = pv.Plotter()
pl2.add_mesh(mesh=sp1.copy(True),scalars=scalars.copy(),clim=clim2)
pl2.show()

# try to plot them in the same plotter window
pl3 = pv.Plotter(shape=(1,2))
for i,clim in enumerate([clim1,clim2]):
    pl3.subplot(0,i)
    sc = copy.deepcopy(scalars)
    pl3.add_mesh(mesh=pv.Sphere(1), scalars=sc, clim=clim)
pl3.show()

## if i do it again but explitly call 'update_scalar_bar_range' it seems to work
pl3 = pv.Plotter(shape=(1,2))
for i,(clim,cmap) in enumerate(zip(*[[clim1,clim2],[cmap1,cmap2]])):
    pl3.subplot(0,i)
    sc = copy.deepcopy(scalars)
    pl3.add_mesh(mesh=pv.Sphere(1), scalars=sc, clim=clim,cmap=cmap)
    #pl3.update_scalar_bar_range(clim)
pl3.show()

