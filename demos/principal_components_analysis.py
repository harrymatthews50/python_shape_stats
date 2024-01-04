
###################### Model Fitting ##################################################
## Step 1 Imports and load the data
from python_shape_stats import helpers, procrustes
from python_shape_stats.statistical_shape_models import ShapePCA
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import copy

# load the simulated faces as a demo dataset
face_path = helpers.get_path_to_simulated_faces() # get path to the relevant package data
face_obj_paths = glob.glob(os.path.join(face_path,'*.obj'))
if len(face_obj_paths)==0:
    Warning('Dataset not found')

# load the data into a numpy array and a list of polydatas
vertices,polydatas = helpers.load_shapes_to_array(face_obj_paths)
ref_polydata = copy.deepcopy(polydatas[0]) # will need this later for visualisation


## Step 2 remove variation due to location, translation and size using Generalized Procrustes Analysis
verts, mean_verts = procrustes.do_generalized_procrustes_analysis(vertices,scale=True,max_iter=5)

## Do PCA
PCAMod = ShapePCA()
PCAMod.fit_transform(verts,center_config=mean_verts,center=True)
PCAMod.reference_polydata = ref_polydata
########## Model Visualisation ##################
## Step 1 Plot variance explained by each PC
# Each successive principal component explains less and less variance in the data...
# plot the variation explained against the PC number.

ax = PCAMod.scree_plot()
plt.show()
# NOTE: python_shape_stats implements several methods for determining the number of real principal components. See 'demos/how_many_pcs.inpby'

##
# or plot the proportion of variation explained agains the PC number
ax = PCAMod.cumulative_variance_plot()
plt.show()

## Visualise the shape effects of the PCs as animations morphing between +/- 3 standard deviations of the scores
# save as a gif in the current folder
pcs_to_plot = [1,2,3,4]
PCAMod.animate_pc([1,2,3,4], max_sd=3)

## Visualise the shape effects of the PCs as color maps showing the displacements along the sirface normals
# red = outward
# blue = inward
PCAMod.colormap_pc(pcs_to_plot)

##

