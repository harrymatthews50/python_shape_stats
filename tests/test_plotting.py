##
import copy
import sys
sys.path.append('/Users/hmatth5/Documents/Projects/python_shape_stats/')
from python_shape_stats import helpers
import numpy as np
##
demo_face_path = helpers._get_path_to_demo_face()
pd,_,_ = helpers.load_shape(demo_face_path)

## plot without scalars

#pl = helpers.plot_shape(pd)
#pl.enable_trackball_actor_style()
#pl.show()

## plot with directed scalars
sc = pd.points[:,0]
vecs = pd.point_normals
helpers.animate_vector(pd,vecs,range(50))

#p = helpers.get_user_input_for_cam_view(pd)
#print(p)
#
#
# pl = helpers.make_plotter()
# helpers.add_shape(mesh=pd,plotter=pl)
# pl.show()
# #
# p = helpers.get_camera_properies(pl.camera)
#
# #
# #
# pl2 = helpers.make_plotter()
# helpers.add_shape(mesh=pd,plotter=pl2)
# helpers.set_camera_properties(pl2.camera,p)
# pl2.show()


