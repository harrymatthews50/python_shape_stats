##
from python_shape_stats import helpers, procrustes
from python_shape_stats.statistical_shape_models import ShapePCA,ShapePLS_2B
import glob
import os
import numpy as np



# load the nose and forehead datasets
nose_path = helpers.get_path_to_simulated_noses() # get path to the relevant package data
nose_obj_paths = glob.glob(os.path.join(nose_path,'*.obj'))
if len(nose_obj_paths)==0:
    Warning('Dataset not found')
forehead_path = helpers.get_path_to_simulated_foreheads()
forehead_obj_paths = glob.glob(os.path.join(forehead_path,'*.obj'))
if len(forehead_obj_paths)==0:
    Warning('Dataset not found')

# check the filenames are matching (specimen A in one dataset is the same as in the other)
assert all([os.path.split(nose_obj_paths[x])[1] == os.path.split(forehead_obj_paths[x])[1] for x in range(len(forehead_obj_paths))])

# load each dataset into a numpy array and a list of polydatas
x_vertices,x_polydatas = helpers.load_shapes_to_array(nose_obj_paths)
y_vertices,y_polydatas = helpers.load_shapes_to_array(forehead_obj_paths)
# Co-align specimens
SUPER_TYPE = 'joint'

if SUPER_TYPE == 'joint':
    n_landmarks_x = x_vertices.shape[0]
    n_landmarks_y = y_vertices.shape[0]
    # concatenate the vertices
    cat_lms = np.vstack([x_vertices,y_vertices])
    # do GPA on the landmarks together
    aligned_lms,mean_lms = procrustes.do_generalized_procrustes_analysis(cat_lms)
    # split them into two blocks again
    X = aligned_lms[:n_landmarks_x]
    Y = aligned_lms[n_landmarks_x:]
    Xmean = mean_lms[:n_landmarks_x]
    Ymean = mean_lms[n_landmarks_x:]
elif SUPER_TYPE == 'separate':
    X,Xmean = procrustes.do_generalized_procrustes_analysis(x_vertices)
    Y,Ymean = procrustes.do_generalized_procrustes_analysis(y_vertices)
else:
    raise ValueError('not a valid \'SUPER_TYPE\'')
# this is not a situation in which the number of PCs retained is especially crucial. The main risk
# is that if you exclude too many you might lose some interesting information. So set a high threshold is my advice...
PCT_VAR = 99.9 # retain PCs explaining up to 99.9% of the total variation

# make PCA model of x
PCA_X = ShapePCA()
PCA_X.fit_transform(X,center_config=Xmean,center=True)
# reduce dimensions
PCA_X.trim_perc_var(PCT_VAR)

#make PCA model of y
PCA_Y = ShapePCA()
PCA_Y.fit_transform(Y,center_config = Ymean,center=True)
PCA_Y.trim_perc_var(PCT_VAR)

# for later we will want to visualise meshes as surface, for this we need the faces of the polydata, which we will attach to each ShapePCA by giving it a reference polydata object (an example mesh from each dataset)

PCA_Y.reference_polydata = y_polydatas[0]
PCA_X.reference_polydata = x_polydatas[0]

PLSMod = ShapePLS_2B()
PLSMod.fit(PCA_X,PCA_Y)

PLSMod.compute_null_distribution(n_reps=1000)
PLSMod.permutation_test_plot(p_crit=.05)

if SUPER_TYPE == 'joint':
    same_coordinate_system = True
else:
    same_coordinate_system = False

# make colormap
dim = 1 # e.g. plot the first dimension
#PLSMod.colormap_latent_dim(dim,link_views=False,same_coordinate_system=same_coordinate_system,direction='normal')

# make animation
PLSMod.animate_latent_dim(1,link_views=False,same_coordinate_system=same_coordinate_system)
