##
import copy
import sys
#sys.path.append('/Users/hmatth5/Documents/Projects/python_shape_stats/')

import numpy as np
#
from python_shape_stats import helpers, procrustes
from python_shape_stats.statistical_shape_models import ShapePCA,ShapePLSHypothesisTest
import glob
import os
import pickle
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# import pathlib
# import glob
import pyvista as pv
# from python_shape_stats import procrustes
# # #
# paths = glob.glob(helpers.get_path_to_simulated_faces()+'/*.obj')
# vs,poly = helpers.load_shapes_to_array(paths)
# vs,_ = procrustes.do_generalized_procrustes_analysis(vs)
# PCA = ShapePCA()
# PCA.fit_transform(vs)
# PCA.reference_polydata = poly[0]
# pickle.dump(PCA,open(helpers.get_path_to_pickled_shape_pca(),'wb'))
##


vars = ['Sex','BMI','Age']
mod = helpers.load_shape_hypothesis_test()
PCA = helpers.load_shape_pca()
#%%
mod.fit(mod.x.copy(),PCA,method='pls',n_comp=3)
v = mod._get_point_r_squared(vars)
mod.plot_coefficients(vars,title=vars,clim=None)
mod.plot_r_squared(vars,title=vars)
mod.run_permutation_test(n_reps = 100)
mod.plot_p_values(vars,title=vars)
##
Y = PCA.transformed_training_data
X = MD.to_array()
##



##

# vecs
sc = mod._get_point_regression_coefs('normal', vars, reverse=True)
pl = pv.Plotter()
pl.add_mesh(PCA.average_polydata.copy(True),scalars = sc[0],cmap = parula_map)
#pl.show()
##


##


##

