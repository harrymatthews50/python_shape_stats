##
import copy

from python_shape_stats import helpers
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os
import glob
#
# load full face
ff,_,_ = helpers.load_shape('/Users/hmatth5/Documents/Projects/python_shape_stats/python_shape_stats/data/facial_part_masks/full_face.obj')

# load nose
nose,_,_ = helpers.load_shape('/Users/hmatth5/Documents/Projects/python_shape_stats/python_shape_stats/data/facial_part_masks/nose.obj')
# load forehead plus orbits
fh_orbits,_,_ = helpers.load_shape('/Users/hmatth5/Documents/Projects/python_shape_stats/python_shape_stats/data/facial_part_masks/forehead_and_orbits3.obj')
# load lower face
lower_face,_,_ = helpers.load_shape('/Users/hmatth5/Documents/Projects/python_shape_stats/python_shape_stats/data/facial_part_masks/lower_face.obj')

#
order = [nose,fh_orbits]#,lower_face]
labels = ['nose','forehead_orbits']#,'lower_face']
mask = np.ones(ff.n_points)*-1
Knn = NearestNeighbors(n_neighbors=1).fit(ff.points)

for i,poly in enumerate(order):
    _,inds = Knn.kneighbors(poly.points)
    mask[np.squeeze(inds)]=i
#
#pl =helpers.add_shape(ff,scalars=mask)
#pl.show()
#
dst = '/Users/hmatth5/Documents/Projects/python_shape_stats/python_shape_stats/data/sim_faces'
source = helpers.get_path_to_simulated_faces()
objs = glob.glob(os.path.join(source,'*.obj'))




## load the objs as polydata
_,poly = helpers.load_shapes_to_array(objs)
obj_fn = [os.path.split(item)[1] for item in objs]
for i,l in enumerate(labels):
    destination = os.path.join(dst,l)
    if not os.path.exists(destination):
        os.mkdir(destination)
    m = mask!=i
    for k,poly_obj in enumerate(poly):
        poly_cop = copy.deepcopy(poly_obj)
        poly_cop.remove_points(m,inplace=True)
        poly_cop.save(os.path.join(destination,obj_fn[k]))

