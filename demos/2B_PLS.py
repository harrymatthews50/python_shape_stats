"""
Studying covariation using two-block partial least-squares

Partial least-squares (PLS) is a family of techniques(see: http://staff.ustc.edu.cn/~zwp/teach/Reg/overview_pls.pdf) for
studying and modelling relations among multivariate data-sets. The terminology can get confusing so, to be clear, when I
say 'two-block-pls' I refer those pls methods that:

    1. Model the relationship between exactly two blocks of variables (although extensions exist to model relationships among more than 2 blocks).
    2. Are symmetrical (the X and Y blocks could be interchanged without altering the results).
    3. Usually the goal of the analysis is to extract and interpret the patterns of covariation, as opposed to prediction (e.g. of Y from X). For prediction variants that emphasize explaining one block from the other are more appropriate.

Python_shape_stats PLS_2B and ShapePLS_2B classes implement implement the 'PLS-SB' algoriths in the terminology of the
review (http://staff.ustc.edu.cn/~zwp/teach/Reg/overview_pls.pdf), although 'PLS-Mode-A' is also symmetric and would also
 be appropriate here.  'PLS-SB' is the same algorithm as 'PLSSVD' in scipy
 (https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSSVD.html).

Two block pls in shape analysis (https://doi.org/10.1080/106351500750049806) is often used to study the patterns of
covariation that exist between related structures. For example Young et.al (https://doi.org/10.1016/j.ajodo.2015.09.028)
studied the relationship between the facial surface and the underlying skeleton. This demo uses the
example of studying covariation between the shape of the human nose and the human forehead.
"""

## Step 1 Imports and load the two data sets
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

## Step 2 Generalised Procrustes Analysis
"""
The first thing we need to do is align all the samples into a common coordinate system.  For this we have two options and 
the appropriate one depends on the goal of the study.

1. If how structure A is positioned relative to structure B is important then you want to treat the points representing 
A and A as single block for superimposition (set SUPER_TYPE to 'joint').

2. If not, and you are only interested in how the shape of one structure covaries with another, ignoring how they are 
positioned relative to each other, then you want to superimpose the blocks separately (SUPER_TYPE = 'separate')
"""
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

## Step 3 Principal Components Analysis
"""
Next we need to create two instances of the 'ShapePCA' class, one representing shape variation in the nose, and one 
representing shape variation in the forehead. The two blocks in 2B-PLS analysis will be the principal component scores of the two structures.
While doing PCA prior to 2B-PLS is not strictly necessary:
1. The implementation of 2B-PLS in python_shape_stats assumes this is how the shapes will be represented.
2. It is usually drastically more efficient when the analysis is done on PC scores rather than the raw data especially 
when the number of features is large.
"""
# How many PCs to keep?
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

# for later we will want to visualise the results on surfaces, for this we need the faces of the polydata, which we will attach to each ShapePCA by giving it a reference polydata object (an example mesh from each dataset)
PCA_Y.reference_polydata = y_polydatas[0]
PCA_X.reference_polydata = x_polydatas[0]

## Step 3 Fit the PLS-2B model
"""
Now we can fit the two block PLS model between X and Y. 
Strictly speaking this fits the model to the pc scores i.e. 
it fits the model between PCA_X.transformed_training_data and PCA_Y.transformed_training_data
"""
PLSMod = ShapePLS_2B()
PLSMod.fit(PCA_X,PCA_Y)

## Step 4 How many PLS components are interesting
"""
Two block PLS reveals how block X covaries with block Y. It finds combinations of the x and y variables that are 
maximally associated with each other. Each successive dimension explains less and less covariation. 
We need to know which of these are real signal and which are noise. One way to do this is via a permutation test:

The covariance explained by each pair of latent dimensions is kept in the attribute PLSMod.cov_explained. 
To determine if this is more than expected by chance we can see what would happen if there was no relationship between X and Y  
and compare the observed values to that. Essentially we permute the rows of the X-block n times and fit a PLS_2B model 
to the simulated data. By doing so we build up a 'null' distribution that can be compared to the observed values.
"""
PLSMod.compute_null_distribution(n_reps=1000)

# A 'p value' for each dimension can then be calculated.
# This is for what proportion of repetitions the null values were greater than the observed values.
# It is common to accept  only those dimensions with p values < .05.
print(PLSMod.perm_test_p_values)

# And we can also plot how the observed values sit in relation to the null.
ax=PLSMod.permutation_test_plot(p_crit=.05)

## Step 5 Visualising the dimensions
"""
All dimensions are significant, however we can see that the first two explain much much more covariation than the 
remainder. These are likely to be the most interesting to look at.
"""

## Plot as colormaps
if SUPER_TYPE == 'joint':
    same_coordinate_system = True
else:
    same_coordinate_system = False

dim = 1 # e.g. plot the first dimension
# press k to save in the current directory
PLSMod.colormap_latent_dim(dim,link_views=False,same_coordinate_system=same_coordinate_system,direction='normal')

## Plot as animation
# press k to save in the current directory
PLSMod.animate_latent_dim(1,link_views=False,same_coordinate_system=same_coordinate_system)


##

