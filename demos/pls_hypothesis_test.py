
## Step 1 Imports and load the data
from python_shape_stats import helpers, procrustes
from python_shape_stats.statistical_shape_models import ShapePCA,ShapePLSHypothesisTest
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
# load the simulated faces as a demo dataset
face_path = helpers.get_path_to_simulated_faces() # get path to the relevant package data
face_obj_paths = glob.glob(os.path.join(face_path,'*.obj'))
if len(face_obj_paths)==0:
    Warning('Dataset not found')
# load simulated patient data
MD = pd.read_excel(helpers.get_path_to_simulated_metadata(),index_col=0)

# make sure shapes and metadata are in the same order
fns = [os.path.splitext(os.path.split(item)[1])[0] for item in face_obj_paths]
isIn, locB = helpers.ismember(fns,MD.index)
assert(all(isIn))
MD = MD.iloc[locB,:]


# load the data into a numpy array and a list of polydatas
vertices,polydatas = helpers.load_shapes_to_array(face_obj_paths)
ref_polydata = copy.deepcopy(polydatas[0]) # will need this later for visualisation




## Step 2 remove variation due to location, translation and size using Generalized Procrustes Analysis
verts, mean_verts = procrustes.do_generalized_procrustes_analysis(vertices,scale=True,max_iter=5)

## Do PCA
PCAMod = ShapePCA()
PCAMod.fit_transform(verts,center_config=mean_verts,center=True)
PCAMod.reference_polydata = ref_polydata

# the purpose of PCA here is just to improve computational efficiency. It is not, from a statistical poiint of view, strictly necessary.
# for this reason I suggest to keep all PCs that explain up to 98 or 99% of the variation. This usually reduces the diemnsionality substantially
# but is unlikely to remove any biologically important signal.
PCAMod.trim_perc_var(98)
### Step 3 Fit PLS model
PLS = ShapePLSHypothesisTest()
PLS.fit(MD,PCAMod)
PLS.run_permutation_test(n_reps=1000)
## Step 4 check results
# we want to look at the p-values and the R-squared values to determine if the effect of a variable was significant




# value corresponds to the change in shape along the surface normals per unit of the predictor
PLS.plot_coefficients(MD.columns,title=MD.columns,direction='normal')

## plot regression coefficients as animations
PLS.animate_coefficients(MD.columns,title=MD.columns, max_sd=3)



##
k
