import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.sparse import spdiags
from scipy.stats import chi2
from sklearn.cross_decomposition import PLSRegression
from python_shape_stats import helpers, procrustes
from sklearn.model_selection import KFold
import joblib
from joblib_progress import joblib_progress
from abc import ABC, abstractmethod, abstractproperty, abstractclassmethod, abstractstaticmethod
import os
from tqdm import tqdm
import pyvista


class PCA:
    """
    Implements a principal components analysis of a data matrix with several
    methods for determining the number of principal components.
    """
    def __init__(self):
        self._eig_vec = None
        self._eig_val = None
        self._transformed_training_data = None
        self._initial_var = None
        self._center_vec = None
        self._standardize_cols_vec = None
        self._n_train_features = None
        self._n_samples = None
        self._params = None
        self._initial_eig_val = None
        self._parallel_analysis_results = None
        self._cross_validation_results = None


    @property
    def eig_vec(self):
        """
        The right singular vectors (the PCs) of the data matrix . The first dimension corresponds to PCs, the second to features.

        :type: np.ndarray
        """
        return self._eig_vec

    @property
    def eig_val(self):
        """
        The total variance explained by each PC. This is computed from the singular values of the training data matrix

        :math:`\\frac{s^2}{k}` where :math:`s` is the singular values and
        :math:`k` is the number of observations (rows in the training data matrix). If x is column-mean centered in the call to 'fit' This is equal to the sample variance in each dimension of the transformed data.

        :type: np.ndarray
        """

        return self._eig_val

    @property
    def eig_std(self):
        """
        The square root of 'self.eig_val'. If x was column mean centered during the call to 'self.fit' these are the sample standard deviations of the transformed training data

        :type: np.ndarray
        """
        if self.eig_val is None:
            return None
        else:
            return np.sqrt(self.eig_val)

    @property
    def cumulative_perc_var(self):
        """
        The cumulative percentage of the total initial variance explained. The :math:`i_{th}` entry is the percentage of variace explained by the pcs 1-i

        :type: np.ndarray
        """
        if self.eig_val is not None:
            return (np.cumsum(self.eig_val) / self._initial_var) * 100
        else:
            return None

    @property
    def n_dim(self):
        """
        The number of dimensions (PCs) currently in the model.

        :type: int
        """
        if self.eig_val is None:
            return None
        else:
            return len(self.eig_val)

    @property
    def transformed_training_data(self):
        """
        Coordinates of the training data in the space spanned by the PCs. Rows correspond to observations, columns to PCs
        These are alternatively called 'PC scores'

        :type: np.ndarray
        """
        return self._transformed_training_data

    @property
    def center_vec(self):
        """
        The vector used to center the columns of x. Using default keyword arguments to 'fit' this is the column mean of
        the training data matrix

        :type: np.ndarray
        """
        if self._center_vec is not None:
            return self._center_vec
        elif self._n_train_features is not None:
            return np.zeros(self._n_train_features)
        else:
            return None

    @property
    def standardize_cols_vec(self):
        """
        The vector used to standardize the columns of x. If standardize_cols == True during the call to fit this will be the column root mean square deviation from 'center_vec'

        :type: np.ndarray
        """

        if self._standardize_cols_vec is not None:
            return self._standardize_cols_vec
        elif self._n_train_features is not None:
            return np.ones(self._n_train_features)
        else:
            return None

    @property
    def params(self):
        """
        Parameters used during the last call to self.fit

        :type: dict
        """
        return self._params


    @staticmethod
    def _fit(x, center=True, center_vec=None, standardize_cols=False):
        params = locals()
        if center == True:
            if center_vec is None:
                center_vec = np.mean(x, axis=0)
            x0 = x - center_vec
        else:
            x0 = x
            center_vec = None

        if standardize_cols == True:
            standardize_cols_vec = np.sqrt(np.mean(np.square(x0)), axis=0)
            x0 = x0 / standardize_cols_vec
        else:
            standardize_cols_vec = None

        u, s, vt = np.linalg.svd(x0, full_matrices=False)
        n = x0.shape[0]
        eig_val = s ** 2 / (n - 1)
        eig_vec = vt
        outputs = {'_center_vec': center_vec, '_standardize_cols_vec': standardize_cols_vec, '_eig_val': eig_val,
                   '_eig_vec': eig_vec}
        params['x0'] = x0
        return outputs, params

    @staticmethod
    def _parallel_analysis_one_iter(x0, seed):
        rng = np.random.default_rng(seed)
        shuff_x = helpers.randomize_matrix(x0, rng)
        out, _ = PCA._fit(shuff_x, center=False, standardize_cols=False)
        return out['_eig_val']

    def fit_transform(self, x: np.ndarray, center: bool = True, center_vec: np.ndarray = None,
                      standardize_cols: np.ndarray = False) -> None:
        """Fits the PCA model to training data x and applies the dimensionality reduction to x

        :param x: a k (observations) by n (features) matrix of training data
        :param center: if True the columns of x will be centered on center_vec prior to the SVD
        :param center_vec: if None the 'center_vec' defaults to the column means of x and thus performs column mean centering of x
        :param standardize_cols: if True the columns of x will be standardized to have unit variance prior to the svd
        """
        self.fit(x, center=center, center_vec=center_vec, standardize_cols=standardize_cols)
        self._transformed_training_data = self.transform(x)

    def fit(self, x: np.ndarray, center: bool = True, center_vec: np.ndarray = None, standardize_cols: bool = False):
        """Fits the PCA model to training data x - uses the singular value decomposition of (column centered and standardized) x.

        :param x: a k (observations) by n (features) matrix
        :param center: if True the columns of x will be centered on center_vec prior to the SVD
        :param center_vec: if None the 'center_vec' defaults to the column means of x and thus performs column mean centering of x
        :param standardize_cols: if True the columns of x will be standardized to have unit variance prior to the svd
        """

        outputs, params = self._fit(x, center=center, center_vec=center_vec, standardize_cols=standardize_cols)
        self._params = params
        self._center_vec = outputs.pop('_center_vec')
        self._standardize_cols_vec = outputs.pop('_standardize_cols_vec')
        self._eig_val = outputs.pop('_eig_val')
        self._eig_vec = outputs.pop('_eig_vec')
        self._n_samples, self._n_train_features = x.shape
        self._initial_var = np.sum(self.eig_val)
        self._initial_eig_val = self.eig_val

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Perform the dimensionality reduction on x

        :param x: an l (observations) by n (features) matrix
        :return: an l (observations) by self.n_dim array of co-ordinates in the lower dimensional space
        """
        x0 = (x - self.center_vec) / self.standardize_cols_vec
        return (self.eig_vec @ x0.T).T

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Reverses the dimensionality reduction to get back the original feature values from the transformed values

        :param x: an l (observations) x self.n_dim matrix (or a vector of length self.n_dim)
        :return: an array with l rows and n (features) columns
        """
        return (x @ self.eig_vec) * self.standardize_cols_vec + self.center_vec

    def parallel_analysis(self, n_reps : int=50, n_jobs : int=1, seed: int | np.random.Generator=None):
        """
        Computes a 'null' distribution of eigenvalue spectra derived from random matrices. Run this method then PCA.parallel_analysis_plot to visualise and interpret the results and determine the optimum number of principal components

        :param n_reps: number of repetitions to use in the parallel analysis
        :param n_jobs: number of jobs to run in parallel
        :param seed: random seed
        """

        rng = np.random.default_rng(seed)
        ss = rng.bit_generator._seed_seq
        child_states = ss.spawn(n_reps)
        with joblib_progress('Running parallel analysis...', n_reps):
            args = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
                joblib.delayed(self._parallel_analysis_one_iter)(self._params['x0'], child_states[x])
                for x in range(n_reps))
        self._parallel_analysis_results = np.vstack(args)



    def scree_plot(self, ax : plt.Axes=None) -> tuple:
        """
        Makes a scree plot of the eigenvalue spectrum

        :param ax: optionally an axis handle
        :return: a tuple containing 1. the axis handle on which the plot is plotted and 2. a NoneType object
        """
        return _eigen_value_plot(self._initial_eig_val, title='Scree Plot', ax=ax)

    def cumulative_variance_plot(self, ax : plt.Axes=None) -> tuple:
        """
        Plots the cumulative percentage of variance explained by each PC.

        :param ax: an axis on which to plot.
        :return: a tuple containing 1. the axis handle on which the plot is plotted and 2. a NoneType object
        """
        return _eigen_value_plot(np.cumsum(self._initial_eig_val) / self._initial_var * 100,
                                 eig_vals_label='Cumulative Var. Exp.',
                                 title='Cumulative Variance', ylabel='Variance\nExplained (%)', ax=ax)

    def parallel_analysis_plot(self, ax : plt.Axes=None, ci_level : float=95, threshold_level :float =95, n_reps: int=50, n_jobs : int=1,
                               recompute_parallel_analysis : bool=False, seed : int | np.random.Generator=None) -> tuple:
        """
        Plots the result of a parallel analysis to determine the number of principal components to retain.
        In a parallel analysis the observed eigenvalue spectrum is compared to a ('null') distribution of eigenvalue spectra
        obtained from random matrices. If the observed eigenvalue for the ith principal component is greater
        than the specified threshold it is considered 'significant'. The estimated number of components is the (k-1)th
        component where k is the index of the first non-significant component. The plot plots the threshold as a solid line.
        The null distribution is plotted as a shaded region that spans the 0-ci_level percentiles of this distribution.

        :param ax: an axis handle on which to plot
        :param ci_level: the upper limit of the null distribution (as a percentile) to plot as a shaded region
        :param threshold_level: the percentile of the null distribution, if the eigenvalue exceeds this it is considered significant
        :param n_reps: the number of repetitions to compute the null distribution (will be ignored if the null distribution has already been calculated by calling PCA.parallel_analysis and recompute_parallel_analysis==False)
        :param n_jobs: the number of jobs to run in parallel while computing the null distribution (will be ignored if the null distribution has already been calculated by calling PCA.parallel_analysis and recompute_parallel_analysis==False)
        :param recompute_parallel_analysis: if True will recompute the null distribution, even if ot has already been calculated
        :param seed: seed for the random number generator
        :return: a tuple containing 1. the axis on which the plot is plotted and 2. the estimated number of principal components
        """
        # determine whether the empirical null distribution needs to be recalculated
        if (self._parallel_analysis_results is None) | recompute_parallel_analysis:
            self.parallel_analysis(n_reps=n_reps, n_jobs=n_jobs, seed=seed)
        return _eigen_value_plot(self._initial_eig_val, distr=self._parallel_analysis_results,
                                 distr_label='Null Spectra', ci_level=ci_level,
                                 threshold_level=threshold_level, ax=ax, title='Parallel\nAnalysis')

    def broken_stick_plot(self, ax : plt.Axes=None, ci_level :float=95, threshold_level: float=95, n_reps: int=1000) -> tuple:
        """
        Plots the result of a broken stick analysis to determine the number of principal components to retain.
        In a broken stick analysis the observed eigenvalue spectrum is compared to a ('null') distribution of eigenvalue spectra
        produced by randomly uniformly splitting the total variance ito the maximum number of principal components possible. If the observed eigenvalue for the ith principal component is greater
        than the specified threshold it is considered 'significant'. The estimated number of components is the (k-1)th
        component where k is the index of the first non-significant component. The plot plots the threshold as a solid line.
        The null distribution is plotted as a shaded region that spans the 0-ci_level percentiles of this distribution.

        :param ax: an axis handle on which to plot
        :param ci_level: the upper limit of the null distribution (as a percentile) to plot as a shaded region
        :param threshold_level: the percentile of the null distribution, if the eigenvalue exceeds this it is considered significant
        :param n_reps: the number of repetitions to perform to estimate the broken stick distribution
        :return: a tuple containing 1. the axis on which the plot is plotted and 2. the estimated number of principal components

        """
        # get the empirical broken stick distribution
        N = len(self._initial_eig_val)
        lengths = helpers.broken_stick_empirical(N, n_reps) * self._initial_var
        return _eigen_value_plot(self._initial_eig_val, distr=lengths, distr_label='Null Spectra', ax=ax,
                                 ci_level=ci_level,
                                 threshold_level=threshold_level, title='Broken Stick')


    def trim_no_pcs(self, no_pcs: int):
        """
        Modifies the object in situ, removing the specified number of pcs

        :param no_pcs: the number of pcs to retain
        """
        self._eig_vec = self._eig_vec[0:no_pcs, :]
        self._eig_val = self._eig_val[0:no_pcs]
        self._transformed_training_data = self._transformed_training_data[:, 0:no_pcs]

    def trim_perc_var(self, pct_var: float) :
        """
        Modifies the object in situ, removing the number of pcs explaining up to the specified amount of variance

        :param pct_var: the percentage of variance s to retain

        """
        if (pct_var <= 0.) | (pct_var >= 100.):
            raise ValueError('pct_var must be between 0 and 100')
        x = np.nonzero(self.cumulative_perc_var >= pct_var)
        no_pcs = x[0][0] + 1
        if no_pcs > self.n_dim:
            no_pcs = self.n_dim
        self.trim_no_pcs(no_pcs)

    def weighted_transform_to_model(self, x: np.ndarray, weights: np.ndarray, max_m_dist: float = None) -> np.ndarray:
        """
        Performs the dimensionality reduction on a single observation with weights that increase/decrease the infleunce
        of a given feature on the solution. The solution can also be constrained to lie within a specified Mahalanobis
        distance of the origin.

        :param x: a single observation in the original feature space
        :param weights: a weight for each feature
        :param max_m_dist: the maximum Mahalanobis distance the transformed values can be from the origin
        :return: the transformed values of x
        """
        if weights is None:
            weights = np.ones_like(x)
        x = helpers.validate_vector(x, 'x')
        weights = helpers.validate_vector(weights, 'weights')
        x0 = (x - self.center_vec) / self.standardize_cols_vec
        a = spdiags(weights, 0, len(x), len(x))
        aq = a @ self.eig_vec.T @ np.diag(self.eig_val)
        u, s, v = np.linalg.svd(aq, full_matrices=False)
        w = s / (s ** 2)
        fit = np.diag(self.eig_val) @ v.T @ np.diag(w) @ u.T @ a @ x0
        if max_m_dist is not None:
            curr_m_dist = self.get_distance(fit)
            if curr_m_dist > max_m_dist:
                fit = self.scale_vec(fit, max_m_dist, metric='mahalanobis')
        return fit

    def scale_vec(self, x: np.ndarray, target_dist: float | np.ndarray, origin: np.ndarray = None,
                  metric: str = 'euclidean') -> np.ndarray:
        """
        Scales vectors in the transformed space to be the target distance from the origin

        :param x: a k (observations) x self.n_dim matrix of locations in the transformed space. This can also be a vector of length self.n_dim
        :param target_dist: the target distance from the origin - this can be a single float or a vector entries corresponding to the rows of x
        :param origin: the origin with respect to which to scale the vectors
        :param metric: the distance metric (can be 'mahalanobis' or 'euclidean'
        :return: the scaled vectors in a matrix the same size as x
        """
        if len(x.shape) == 1:
            x = helpers.reshape_vector(x, 'row')
        if origin is None:
            origin = np.zeros_like(self.eig_val)
        x0 = x - origin
        # use default origin as x0 is already centered
        d = self.get_distance(x0, origin=None, metric=metric)
        sc = d / target_dist
        return x0 / sc

    def maha_dist_to_p_value(self, md: float | np.ndarray) -> float | np.ndarray:
        """
        Returns the probability  an observation as or more extreme than the given
        Mahalanobis distance from the origin

        :param md: the mahalanobis distance, can be a float or an array of floats
        :return: the p-values corresponding to the elements in md
        """
        return 1 - chi2.cdf(md ** 2, self.n_dim)

    def p_value_maha_dist(self, p):
        """
        Returns the Mahalanobis distance for which the probability of an observation as or more extrem is equal to p

        :param p: the p value, can be a float or an array of floats
        :return: the Mahalanobis distance, corresponding to the elements in md
        """

        return np.sqrt(chi2.ppf(1 - p, self.n_dim))

    def get_distance(self, x: np.ndarray, origin: np.ndarray = None, metric: str = 'euclidean') -> np.ndarray:
        """
        Gets the distance of x from the origin, in the space of the transformed variables

        :param x: a vector of length l or an array k (observations) x l. l == self.n_dim
        :param origin: a vector of length l or an array k (observations) x l. l == self.n_dim. If origin is a matrix each row of x is comapred to the corresponding row of origin
        :param metric: the distance metric to use, must either be 'euclidean' or 'mahalanobis'
        :return: an array containing the distances of each row of x from
        """
        if origin is None:
            origin = np.zeros_like(self.eig_val)
        if len(x.shape) == 1:
            x = helpers.reshape_vector(x, 'row')
        origin = helpers.reshape_vector(origin, 'row')
        x0 = x - origin
        if str.lower(metric) == 'euclidean':
            pass
        elif str.lower(metric) == 'mahalanobis':
            x0 /= helpers.reshape_vector(self.eig_std, 'row')
        else:
            raise ValueError('Invalid distance type')
        return np.linalg.norm(x0, axis=1)


class ShapePCA(PCA):
    """
    Implements some additional properties and methods for doing principal components analysis on a sample of shapes
    """
    def __init__(self):
        super().__init__()
        self._average_polydata = None
        self._reference_polydata = None

    @property
    def average_polydata(self):
        """
        A polydata object of the average shape

        :type: helpers.TriPolyData
        """
        if self.center_vec is not None:
            if self._reference_polydata is not None:
                return helpers.TriPolyData(helpers.landmark_2d_to_3d(self.center_vec), self._reference_polydata.faces)
            return helpers.TriPolyData(helpers.landmark_2d_to_3d(self.center_vec))
        return None

    @property
    def reference_polydata(self):
        """
        This should be an example shape from your dataset.The reference polydata determines how the average_polydata in visualisation of the PCA model can be rendered.
        Specifically if the topology of the surface is known this should be kept in reference_polydata.faces
        The average and other shape visualisations can then be visualised as surfaces, not just points.

        :type: helpers.TriPolyData
        """
        return self._reference_polydata

    @reference_polydata.setter
    def reference_polydata(self, val):
        if isinstance(val, helpers.TriPolyData):
            self._reference_polydata = val
        else:
            raise ValueError('Reference polydata should be an instance of helpers.TriPolyData')

    def fit(self, x: np.ndarray, center: bool = True, center_config: np.ndarray = None):
        """Fits the PCA model to training data x - uses the singular value decomposition of (centered) x.

        :param x: an l (vertices/landmarks) x 3 dimensions x k (observations) array of homologous shapes
        :param center: if True x will be centered along the third dimension by center_config x  prior to the SVD
        :param center_config: if None the 'center_config' defaults to np.mean(x,axis=2) and thus centers the landm,rks on their mean shapes
        """
        if center is True:
            if center_config is None:
                center_config = np.mean(x, axis=2)
            center_vec = helpers.landmark_3d_to_2d(center_config)
        else:
            center_vec = None
        x = helpers.landmark_3d_to_2d(x)
        super().fit(x, center=center, center_vec=center_vec, standardize_cols=False)

    def transform(self, x: np.ndarray, apply_procrustes_transform: bool = True, procrustes_scale: bool = True,
                  n_jobs: int = 1):
        """
        Transforms the landmark configurations in x into the lower dimenisonal space of the PCA model.

        :param x: an l (vertices/landmarks) x 3 dimensions x k (observations) array of homologous shapes
        :param apply_procrustes_transform: if True each observation in x will be aligned first to the mean shape using the Procrustes transform. If false then the shapes must already be aligned to the mean shape before calling this function
        :param procrustes_scale: if apply_procrustes_transform is True then this will determine if the configuratioons are allowed to scale towards the mean shape
        :param n_jobs: the number of jobs to run in parallel (see joblib.Parallel documentation)
        :return: a tuple containing 1. a k observations x self.ndims array of coordinates in the lower dimensional space, 2. a list of the procrustes transformations applied to each  configuration
        """

        if len(x.shape) == 2:
            x = x[:, :, np.newaxis]
        n_configs = x.shape[2]

        if apply_procrustes_transform:
            r = joblib.Parallel(n_jobs=n_jobs, verbose=10)(
                joblib.delayed(procrustes.conform_p_to_q)(x[:, :, i], helpers.landmark_2d_to_3d(self.center_vec),
                                                          scale=procrustes_scale) for i in range(n_configs))
            v, t = zip(*r)
            x = np.stack(v, axis=2)
        else:
            t = [np.identity(4)] * n_configs  # no transformation is the same as tranforming by an identity matrix

        return super().transform(helpers.landmark_3d_to_2d(x)), t

    def fit_transform(self, x: np.ndarray, center: bool = True, center_config: np.ndarray = None):
        """Fits the PCA model to training data in x and transforms it into the lower dimensional space

        :param x: an l (vertices/landmarks) x 3 dimensions x k (observations) array of homologous shapes
        :param center: if True x will be centered along the third dimension by center_config x  prior to the SVD
        :param center_config: if None the 'center_config' defaults to np.mean(x,axis=2) and thus centers the landmarks on their mean shapes
        """
        self.fit(x, center=center, center_config=center_config)
        self._transformed_training_data, _ = self.transform(x, apply_procrustes_transform=False)

    def animate_pc(self, pc_num : int, max_sd : float = 3, n_frames=20,mode='write_gif', **kwargs):
        """
        Makes an animation illustrating the shape change associated a specified PC

        :param pc_num: the PC to visualise
        :param max_sd: The shape will be morphed between -max_sd - max_sd standard deviations
        :param n_frames: the number of frames to render in the animation
        :param mode: if 'write_gif' animation will be written to file, if NoneType it will not
        :param kwargs: see helpers.animate_vectors
        """

        if not helpers.my_is_iterable(pc_num):
            pc_num= [pc_num]

        vec = [helpers.landmark_2d_to_3d(self.eig_vec[i, :] * self.eig_std[i]) for i in pc_num]
        frame_scalars = helpers._generate_circular_sequence(max_sd, -max_sd, n_in_sequence=n_frames)
        frame_scalars = [frame_scalars]*len(pc_num)
        polydatas = [copy.deepcopy(self.average_polydata) for i in range(len(pc_num))]

        # set default values of some of the keyword arguments
        file_name = kwargs.pop('file_name', 'PC_' + str(pc_num).replace('[', '').replace(']','').replace(',', '_').replace(' ',''))
        title = kwargs.pop('title',['PC '+str(i) for i in pc_num])
        # make animation
        helpers.animate_vectors(polydatas, vec, frame_scalars, file_name=file_name,title=title, **kwargs)
    def colormap_pc(self,pc_num,**kwargs):
        """
        Make colormaps of the pcs in pc_num

        :param pc_num: a single, or list of n pcs to plot
        :param kwargs: see helpers.plot_colormaps

        """
        if not helpers.my_is_iterable(pc_num):
            pc_num = [pc_num]

        file_name = kwargs.pop('file_name', 'PC_' + str(pc_num).replace('[', '').replace(']', '').replace(',', '_').replace(' ',''))
        title = kwargs.pop('title', ['PC ' + str(i) for i in pc_num])
        direction = kwargs.pop('direction','normal')

        pd = self.average_polydata
        vec = [helpers.landmark_2d_to_3d(self.eig_vec[i, :] * self.eig_std[i]) for i in pc_num]

        point_scalars = helpers._vecs_to_scalars(vec,direction=direction, poly=pd)

        helpers.plot_colormaps(pd, point_scalars, file_name=file_name, title = title,direction=direction, **kwargs)


def _eigen_value_plot(eig_vals, eig_vals_label='Eigenvalue Spectrum', distr=None, distr_label='', ci_level=95.,
                      threshold_level=95., ax=None, xlabel='PC',
                      ylabel='Explained\nVariance', title=''):
    if ax is None:
        ax = plt.subplot()
    x = np.linspace(1, len(eig_vals), len(eig_vals))
    ax.plot(x, eig_vals, 'b+-', label=eig_vals_label)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)

    if distr is not None:
        # get CIs and threshold level
        pctiles = np.percentile(distr, [0, ci_level, threshold_level], axis=0)
        # plot a green filled region between the lower and upper CIs
        ax.fill_between(x, pctiles[0, :], pctiles[1, :], alpha=.2, facecolor='g', edgecolor='g', label=distr_label)
        ax.plot(x, pctiles[2, :])

        # plot the specified threshold as a solid line
        ax.plot(x, pctiles[2, :], c='g', label='Threshold')

        # find the first time that the explained variance is less than the threshold
        inds = np.nonzero(np.less(eig_vals, pctiles[2, :]))[0]
        if len(inds)==0:
            n_comps=len(eig_vals)
        else:
            n_comps = np.min(inds)
        ax.axvline(x=n_comps, c='k', ls=':', label='Estimated No. Comp.')
    else:
        n_comps = None
    ax.legend()
    return ax, n_comps


class PLS(ABC):
    """
    An abstract base class implementing properties and methods common to all classes implementing partial least-squares analyses
    """
    def __init__(self):
        super().__init__()
        self._x = None
        self._y = None
        self._standardize_x = False
        self._standardize_y = False
        self._center_x = True
        self._center_y = True
        self._observation_mask = None
        self._observation_weights = None
        self._x_mu = None
        self._y_mu = None
        self._x_std = None
        self._y_std = None

    @property
    def y(self):
        """
        The complete block of y variables, before exclusion mask or dummy variables are created as a pandas
        DataFrame. The y.setter, depending on the subclass, may take or expect an instance of
        :py:class:`PCA` or :py:class:`ShapePCA` in which case the getter will return the attribute
        :py:attr:`PCA.transformed_training_data`
        :type: pd.DataFrame
         """
        return self._convert_to_data_frame(self._y)

    @y.setter
    def y(self,val):
        if isinstance(val, pd.DataFrame):
            val = self._validate_data_frame(val)
        self._y = copy.deepcopy(val)
    @property
    def x(self):
        """
        The complete block of x variables, before exclusion mask or dummy variables are created as a pandas
        DataFrame. The x.setter, depending on the subclass, may take or expect an instance of
        :py:class:`PCA` or :py:class:`ShapePCA` in which case the getter will return the attribute
        :py:attr:`PCA.transformed_training_data`

        :type: pd.DataFrame
        """
        return self._convert_to_data_frame(self._x)

    @x.setter
    def x(self,val):
        if isinstance(val,pd.DataFrame):
            val = self._validate_data_frame(val)
        self._x = copy.deepcopy(val)

    @property
    def __n_obs(self):
        """hidden property keeping track of how many subjects are in the x block, before exclusion criteria are applied"""
        return self.x.shape[0]

    @property
    def center_x(self):
        """
        Indicates whether or not the x-block is to be column centered. Is intended to be set only via a call to the 'fit' method of the class

        :type: bool
        """

        return self._center_x

    # @center_x.setter
    # def center_x(self,val):
    #     self._center_x = val
    @property
    def center_y(self):
        """
        Indicates whether or not the y-block is to be column-centered. Is intended to be set only via a call to the 'fit' method of the class

        :type: bool
        """
        return self._center_y

    # @center_y.setter
    # def center_y(self,val):
    #     self._center_y = val

    @property
    def standardize_x(self):
        """
        Indicates whether or not the x-block is to be column standardized to have unit variance. Is intended to be set only via a call to the 'fit' method of the class

        :type: bool
        """
        return self._standardize_x

    # @standardize_x.setter
    # def standardize_x(self,val):
    #     self._standardize_x = val

    @property
    def standardize_y(self):
        """
        Indicates whether or not the x-block is to be column standardized to have unit variance. Is intended to be set only via a call to the 'fit' method of the class

        :type: bool
        """

        return self._standardize_y


    @property
    def n_obs(self):
        """
        Number of observations after exclusion mask have been applied

        :type: int
        """
        if self.observation_mask is None:
            return None
        return sum(self.observation_mask)

    @property
    def x_treated(self):
        """
        The block of x variables  after categorical variables are expanded to dummy variables
        and exclusion mask has been applied. If no categorical variables are in self.x and no
        observation mask has been applied then this is the same as self.x cast into a DataFrame

        :type: pd.DataFrame
        """
        return self._treat_var_block(self.x, self.observation_mask, self._are_categories_possible(self._x))[0]

    @property
    def y_treated(self):
        """
        The block of y variables  after categorical variables are expanded to dummy variables
        and exclusion mask has been applied. If no categorical variables are in self.y and no
        observation mask has been applied then this is the same as self.y

        :type: pd.DataFrame
        """
        return self._treat_var_block(self.y, self.observation_mask, self._are_categories_possible(self._y))[0]

    @property
    def _y_block_var_indices(self):
        if self._y is None:
            return None
        return self._treat_var_block(self.y, self.observation_mask, self._are_categories_possible(self._y))[1]

    @property
    def _x_block_var_indices(self):
        if self._x is None:
            return None
        return self._treat_var_block(self.x, self.observation_mask, self._are_categories_possible(self._x))[1]

    @property
    def _xblock_data_types(self):
        """
        This is potentially confusing...if there are no categorical variables in x or there is no exclusion mask set
        then this is identical to self.x.dtypes, if there is both an observation mask set and some categorical variables
        then the pd.CategoricalDtype objects of x.dtypes are 'squeezed' to remove reference that are not present
        after removing the relevant observations

        :type: pd.Series
        """
        if self._observation_mask is None:
            return self.x.dtypes
        else:
            return helpers.squeeze_categorical_dtypes(self.x[self.observation_mask])

    @property
    def _yblock_data_types(self):
        """
        This is potentially confusing...if there are no categorical variables in x or there is no exclusion mask set
        then this is identical to self.y.dtypes, if there is both an observation mask set and some categorical variables
        then the pd.CategoricalDtype objects of y.dtypes are 'squeezed' to remove reference that are not present
        after removing the relevant observations

        :type: pd.Series
        """
        if self._observation_mask is None:
            return self.y.dtypes
        else:
            return helpers.squeeze_categorical_dtypes(self.y[self._observation_mask])

    @property
    def _x0(self):
        if self.x is None:
            return None
        x0 = self.x_treated.to_numpy(dtype=float)
        if self.center_x:
            x0 = x0 - self.x_mu
        if self.standardize_x:
            x0 = x0 / self.x_std
        x0 *= self.observation_weights[self.observation_mask, np.newaxis]
        return x0

    @property
    def _y0(self):
        y0 = np.atleast_2d(self.y_treated.to_numpy(dtype=float))
        y0 = self._center_scale_y(y0,reverse=False)
        y0 *= self.observation_weights[self.observation_mask, np.newaxis]
        return y0

    @property
    def observation_weights(self):
        """
        A vector of weights (one for each observation) that control the influence of each observation on the fitted model

        :type: np.ndarray
        """
        if self._observation_weights is None:
            if self._x is None:
                return None
            return np.ones(self.__n_obs)
        else:
            if len(self._observation_weights) != self.__n_obs:
                raise ValueError('observation weights should be the same length as the number of observations')

            return self._observation_weights

    @observation_weights.setter
    def observation_weights(self, val):
        self._observation_weights = val.flatten()

    @property
    def observation_mask(self):
        """
        A boolean vector that can be used to control the observations that are included in fitting the model.
        Each element corresponds to an observation (row of x and y). If the value is true then that observation will be included
        otherwise it won't be.

        :type: np.ndarray
        """
        if self._observation_mask is None:
            if self._x is None:
                return None
            return np.ones(self.__n_obs, dtype=bool)
        else:
            # check is the same size as x
            if self._x is not None:
                if len(self._observation_mask) != self.__n_obs:
                    raise ValueError('observation mask should be the same length as the number of observations')
            return self._observation_mask.flatten()

    @observation_mask.setter
    def observation_mask(self, val):
        if not isinstance(val, np.ndarray):
            raise TypeError('observation maks should be an numpy.ndarray')
        val = val.astype(dtype=bool)
        self._observation_mask = val

    @property
    def no_y_features(self):
        """
        The number of features/variables (columns) of the y-block before categorical variables are expanded to dummy variables

        :type: int
        """
        if self._y0 is None:
            return None
        return self._y0.shape[1]

    @property
    def no_x_features(self):
        """
        The number of features/variables (columns) of the x-block before categorical variables are expanded to dummy variables

        :type: int
            """

        if self._x0 is None:
            return None
        return self._x0.shape[1]

    @property
    def x_mu(self):
        """
        The (weighted) column means of x after categorical variables are expanded to dummy variables and after the observation mask has been applied

        :type: np.ndarray
        """
        if self._x_mu is None:
            # if the mean has not been otherwise explicitly set
            if self.x_treated is None:
                return None
            return helpers.weighted_column_mean(self.x_treated,
                                                self.observation_weights[self.observation_mask]).to_numpy(dtype=float)
        else:
            return self._x_mu

    @property
    def x_std(self):
        """
        The (weighted) column standard deviations of x after categorical variables are expanded to dummy variables and after the observation mask has been applied

        :type: np.ndarray
        """

        if self._x_std is None:
            if self.x is None:
                return None
            x0 = self.x_treated
            w = self.observation_weights[self.observation_mask]
            if self.center_x:
                x0 = x0 - self.x_mu
            return helpers.weighted_rms(x0, w).to_numpy(dtype=float)
        else:
            return self._x_std

    @property
    def y_mu(self):
        """
        The (weighted) column means of y after categorical variables are expanded to dummy variables and after the observation mask has been applied

        :type: np.ndarray
        """

        if self._y_mu is None:
            # if the mean has not been otherwise explicitly set
            if self.y_treated is None:
                return None
            return helpers.weighted_column_mean(self.y_treated,
                                                self.observation_weights[self.observation_mask]).to_numpy(dtype=float)
        else:
            return self._y_mu

    @property
    def y_std(self):
        """
        The (weighted) column standard deviations of x after categorical variables are expanded to dummy variables and after the observation mask has been applied

        :type: np.ndarray
        """
        if self._y_std is None:
            if self.y_treated is None:
                return None
            y0 = self.y_treated  # [self.observation_mask, :]
            w = self.observation_weights[self.observation_mask]
            if self.center_y:
                y0 = y0 - self.y_mu
            return helpers.weighted_rms(y0, w).to_numpy(dtype=float)
        else:

            return self._y_std

    def _center_scale_x(self,x,reverse=False):
       return self._center_scale(x,self.x_mu,self.x_std,scale=self.standardize_x,center=self.standardize_x,reverse=reverse)

    def _center_scale_y(self, y, reverse=False):
        return self._center_scale(y,self.y_mu,self.y_std,scale=self.standardize_y,center=self.center_y,reverse=reverse)
    @staticmethod
    def _center_scale(x,mu,std,scale=True,center=True,reverse=False):
        if not reverse:
            if center:
                x-=mu
            if scale:
                x /=std
        else:
            if scale:
                x*=std
            if center:
                x+=mu
        return x

    @staticmethod
    def _treat_var_block(x, mask, search_for_cats):
        if not isinstance(x,
                          pd.DataFrame):  # it is expected that an end user will not be able to reach this error, this error indicates a bug
            raise TypeError('Expected pandas.DataFrame')
        x = x.iloc[mask]
        x, indices = PLS._expand_block(x, x.dtypes, search_for_categories=search_for_cats)
        return x, indices, helpers.squeeze_categorical_dtypes(x)

    @staticmethod
    def _are_categories_possible(in_):
        if isinstance(in_, pd.DataFrame):
            return True
        else:
            return False

    @staticmethod
    def _convert_to_data_frame(in_):
        if in_ is None:
            return None
        if isinstance(in_, pd.DataFrame):
            return in_
        if isinstance(in_, PCA):
            return pd.DataFrame(data=in_.transformed_training_data)
        if isinstance(in_, (np.ndarray, pd.Series)):
            return pd.DataFrame(in_)

    @staticmethod
    def _expand_block(x, dtypes=None, search_for_categories=False):
        if not isinstance(x, pd.DataFrame):
            raise TypeError('Expected a pandas data frame')
        # check if any categorical variables in dtypes
        is_cat = [str(item) == 'category' for item in dtypes]
        if np.all(np.equal(is_cat, False)) | (not search_for_categories):  # no expansion to be done
            return x, [i for i in range(x.shape[1])]
        x_chunks = []
        block_indices = []
        for col in range(x.shape[1]):
            if str(x.dtypes.iloc[col]) == 'category':
                dum = helpers.get_dummy(x.iloc[:, col], x.dtypes.iloc[col])
                x_chunks.append(dum)
                block_indices.extend([col] * dum.shape[1])
            else:
                x_chunks.append(x.iloc[:, col])
                block_indices.extend([col])
        x = pd.concat(x_chunks, axis=1)
        return x, block_indices

    @staticmethod
    def _validate_data_frame(val):
        # check all columns are unique
        if len(np.unique(val.columns)) != len(val.columns):
            raise ValueError('All columns of the data frame should have unique headers')
        out = copy.deepcopy(val)
        out.columns = [str(item) for item in val.columns]
        return out

    @abstractmethod
    def fit(self):
        ...


class PLS_2B(PLS):
    """
    Performs a symmetrical 2-block PLS via singular value decomposition of the cross-covariance matrix C
    """
    # :math:`C = X0^T \\cdot Y0` where X0 and Y0 are two n (observations) by k (x features) or q (y features) matrices.
    # These matrices are usually column mean-centered and optionally standardised so that column variance is equal to 1.
    # The singular value decomposition yields
    #
    # :math: `USV^T = C`
    #
    # where the ith column of U and V




    def __init__(self):
        super().__init__()
        self._x_projection_matrix = None
        self._y_projection_matrix = None
        self._cov_explained = None
        self._x_scores = None
        self._y_scores = None
        self._permutation_null_distribution = None
        self._inner_relation_coefs = None

    @property
    def x_projection_matrix(self):
        """
        An orthogonal projection matrix onto the lower-dimensional space of the x-scares.
        Rows correspond to the features (columns) in the x-block, columns correspond to the PLS dimensions.

        :type: np.ndarray
        """
        return self._x_projection_matrix

    @property
    def y_projection_matrix(self):
        """
        An orthogonal projection matrix onto the lower-dimensional space of the y-scares.
        Rows correspond to the features (columns) in the y-block, columns correspond to the PLS dimensions.

        :type: np.ndarray
        """
        return self._y_projection_matrix


    @property
    def cov_explained(self):
        """
        The amount of covariance explained by each latent dimension

        :type: np.ndarray
        """
        return self._cov_explained

    @property
    def x_scores(self):
        """
        The observations in the x-block represented in the lower-dimensional space.
        Rows correspond to observations, columns to dimensions.

        :type: np.ndarray
        """
        return self._x_scores

    @property
    def y_scores(self):
        """
        The observations in the y-block represented in the lower-dimensional space.
        Rows correspond to observations, columns to dimensions.

        :type: np.ndarray
        """
        return self._y_scores

    @property
    def n_dim(self):
        """
        The number of latent dimensions in the model

        :type: int
        """
        if self.cov_explained is None:
            return None
        else:
            return len(self.cov_explained)

    @property
    def inner_relation_coefs(self):
        """
        Coefficients modelling the relationship between the x-scores and the y scores.
        Each coefficient corresponds to a pair of dimensions. The ith coefficient is the coefficient of a major axis regression of self.y_scores[:,i] onto self.x_scores[:,i]

        :type: np.ndarray
        """
        return self._inner_relation_coefs

    @property
    def perm_test_p_values(self):
        """
        P values corresponding to each latent dimension. :py:meth:`PLS_2B.compute_null_distribution` before accessing this property

        :type: np.ndarray
        """
        if self._permutation_null_distribution is None:
            return None
        return np.sum((self._permutation_null_distribution - self.cov_explained[np.newaxis,:])>=0,axis=0) / self._permutation_null_distribution.shape[0]

    def transform_x(self, x : np.ndarray | pd.DataFrame, expand_categories : bool=True) -> np.ndarray:
        """
        Performs the dimension reduction on x. If expand_categories == True the function will search for categorical variables in x
        and expand them into dummy variables.

        :param x: an n (observations) by k (number of x features) np.ndarrdy or pd.DataFrame. If expand_categories==True, this must be a DataFrame
        :param expand categories: whether to search for categoricakl variables in x and expand them into dummy variables.

        :return: np.ndarray
        """
        if expand_categories:
            if not isinstance(x, pd.DataFrame):
                raise TypeError('when \'expand_categories\'==True x is expected to be a pandas data frame')
                # convert categorical data to dummy coded data if present
                x = self._expand_block(x, dtypes=self._xblock_data_types)

        if isinstance(x, (pd.DataFrame, pd.Series)):
            x = x.to_numpy(dtype=float)

        if self.center_x:
            x -= self.x_mu
        if self.standardize_x:
            x /= self.x_std

        return x @ self.x_projection_matrix

    def transform_y(self, y : np.ndarray | pd.DataFrame , expand_categories:bool=True)->np.ndarray:
        """
        Performs the dimension reduction on y. If expand_categories == True the function will search for categorical variables in x
        and expand them into dummy variables.

        :param x: an n (observations) by k (number of x features) np.ndarrdy or pd.DataFrame. If expand_categories==True, this must be a DataFrame
        :param expand categories: whether to search for categoricakl variables in x and expand them into dummy variables.

        :return: np.ndarray
        """

        if expand_categories:
            if not isinstance(y, pd.DataFrame):
                raise TypeError('when \'expand_categories\'==True x is expected to be a pandas data frame')
                # convert categorical data to dummy coded data if present
                y = self._expand_block(y, dtypes=self._yblock_data_types)

        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.to_numpy(dtype=float)
        if self.center_y:
            y -= self.y_mu
        if self.standardize_y:
            y /= self.y_std
        return y @ self.y_projection_matrix



    def fit(self, x : np.ndarray | pd.DataFrame | ShapePCA, y: np.ndarray | pd.DataFrame | ShapePCA, center_x:bool=True, center_y:bool=True, standardize_x:bool=False, standardize_y:bool=False,
            observation_weights:np.ndarray=None, observation_mask:np.ndarray=None):
        """
        Fits the two block PLS model to the data x and y

        :param x: the x-block of variables. This can be an n (observations) by k features np.ndarray or pd.DataFrame. It can also be an instance of :py:class:`PCA` or one of its subclasses. If it is an np.ndarray this simply becomes the x block. If it is a DataFrame, each column's dtype is checked if dtype=='category' these will be expanded to l-1 dummy variables, where l is the number of unique values in the column. If it is a :py:class:`PCA` (or a subclass thereof) the x block will be the PC scores :py:attr:`PCA.transformed_training_data`
        :param y: the y-block of variables. The same details apply as for 'x'
        :param center_x: boolean indicating whether to column mean center x
        :param center_y: boolean indicating whether to column mean center y
        :param standardize_x: boolean indicating whether to standardize the columns of x to have unit variance
        :param standardize_y:boolean indicating whether to standardize the columns of y to have unit variance
        :param observation_weights: a vector of length n (observations)
        :param observation_mask: a boolean array of length n (observations)
        :return:
        """

        self._center_x = center_x
        self._center_y = center_y
        self._standardize_x = standardize_x
        self._standardize_y = standardize_y
        self._observation_mask = observation_mask
        self._observation_weights = observation_weights
        self.x = x
        self.y = y
        # do the svd
        cov = self._x0.T @ self._y0
        rank_upper_bound = min(self.x.shape[0], self.x.shape[1], self.y.shape[1])
        [self._x_projection_matrix, self._y_projection_matrix, self._cov_explained] = self._do_svd(cov,
                                                                                                   self.observation_weights,
                                                                                                   rank_upper_bound)
        self._x_scores = self.transform_x(self.x_treated, expand_categories=False)
        self._y_scores = self.transform_y(self.y_treated, expand_categories=False)
        self._inner_relation_coefs = self._fit_inner_relation(self.x_scores, self.y_scores,
                                                             self.observation_weights[self.observation_mask])

    @staticmethod
    def _do_svd(cov, w, n_comps):
        [u, s, v] = np.linalg.svd(cov)
        u = u[:, :n_comps]
        v = v[:n_comps]
        s = s[:n_comps]
        return u, v.T, (s ** 2) / sum(w)

    @staticmethod
    def _fit_inner_relation(x_scores, y_scores, weights):
        """
        Fits the inner model between the x scores and the y scores. The relationship between each pair of latent variables self.xscores[:,i] self.y_scores[:,i]
        as a symmetrical (major axis) regression

        :return: coefs
        """
        n_obs, n_dims = x_scores.shape
        coefs = np.zeros(n_dims)
        for i in range(n_dims):
            xy = np.concatenate([np.atleast_2d(x_scores[:, i]), np.atleast_2d(y_scores[:, i])], axis=0)
            cov = np.cov(xy, aweights=weights, rowvar=True)
            w, v = np.linalg.eigh(cov)
            vec = v[:, -1]
            coefs[i] = vec[1] / vec[0]
        return coefs

    def reconstruct_y_from_scores(self, y_scores):
        y0 = y_scores @ self._y_projection_matrix
        y = self._center_scale_y(y0,reverse=True)
        return y

    def reconstruct_x_from_scores(self, x_scores):
        x0 = x_scores @ self._x_projection_matrix
        x = self._center_scale_x(x0,reverse=True)
        return x

    def predict_y_scores_from_x(self, x_scores):
        return x_scores * self.inner_relation_coefs[np.newaxis, :]

    def predict_x_scores_from_y(self, y_scores):
        return y_scores * 1 / self.inner_relation_coefs[np.newaxis, :]

    @staticmethod
    def _perm_test_one_iter(x0, y0, weights, n_comps, seed):
        rng = np.random.default_rng(seed)
        n_obs = x0.shape[0]
        # permute rows of x
        x0 = x0.copy()
        x0 = x0[rng.permutation(n_obs), :]
        cov = x0.T @ y0
        _, _, s = PLS_2B._do_svd(cov, weights, n_comps)
        return s

    def compute_null_distribution(self, n_reps=1000, n_jobs=1, seed=None):
        rng = np.random.default_rng(seed)
        ss = rng.bit_generator._seed_seq
        child_states = ss.spawn(n_reps)
        with joblib_progress('Running permutation test...', n_reps):
            args = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
                joblib.delayed(self._perm_test_one_iter)(self._x0, self._y0,
                                                         self.observation_weights[self.observation_mask], self.n_dim,
                                                         child_states[x])
                for x in range(n_reps))

        self._permutation_null_distribution = np.vstack(args)

    def permutation_test_plot(self, p_crit=.05, ax=None, recompute_null_distribution=False, n_reps=1000, seed=None,
                              n_jobs=1):
        CIpct = (1 - p_crit) * 100

        if (self._permutation_null_distribution is None) | recompute_null_distribution:
            self.compute_null_distributions(n_reps=n_reps, seed=seed, n_jobs=n_jobs)

        return _eigen_value_plot(self.cov_explained, eig_vals_label='Covariance Explained',
                          distr=self._permutation_null_distribution, distr_label='Null', ci_level=CIpct,
                          threshold_level=CIpct, ax=ax, xlabel='PLS Dim',
                          ylabel='Explained\nCovariance', title='')

class ShapePLS_2B(PLS_2B):
    def __init__(self):
        super().__init__()

    @property
    def _is_x_shape(self):
        return isinstance(self._x,ShapePCA)
    @property
    def _is_y_shape(self):
        return isinstance(self._y,ShapePCA)


    def _get_base_polydata(self):
        bp = []
        if self._is_x_shape:
            bp.append(copy.deepcopy(self._x.average_polydata))
        if self._is_y_shape:
            bp.append(copy.deepcopy(self._y.average_polydata))
        return bp

    def _get_latent_vectors(self,dim):
        vectors = []
        if self._is_x_shape:
            vectors.append(helpers.landmark_2d_to_3d(self._x.eig_vec.T @ self.x_projection_matrix[:,dim]))
        if self._is_y_shape:
            vectors.append(helpers.landmark_2d_to_3d(self._y.eig_vec.T @ self.y_projection_matrix[:,dim]))
        return vectors

    def _get_frame_scalars(self,dim,max_sd=3,n_frames=20):
        sc_x = helpers._generate_circular_sequence(-max_sd, max_sd, 0, n_in_sequence=n_frames) * np.std(
            self.x_scores[:, dim])
        sc_y = sc_x*self.inner_relation_coefs[dim]
        frame_scalars = [sc_x,sc_y]
        is_shape = [self._is_x_shape,self._is_y_shape]
        return [frame_scalars[i] for i in range(2) if is_shape[i]]

    def _get_point_scalars(self,dim,direction):
        pd = self._get_base_polydata()
        latent_vectors = self._get_latent_vectors(dim)
        if direction.lower() == 'normal':
            normals = [item.point_normals for item in pd]
            scalars = [np.sum(normals[i]*latent_vectors[i],axis=1) for i in range(len(pd))]
        elif direction.lower() == 'total':
            scalars = [np.linalg.norm(latent_vectors[i], axis=1) for i in range(len(pd))]
        return scalars
    # some methods specific for visualising paired latent dimensiona
    def animate_latent_dim(self,dim,max_sd=3,n_frames=20,**kwargs):
        """
        Animates the shape transformations of a single latent dimension.

        :param dim: the latent dimension to visualise
        :param max_sd: the animation will go between +/- max_sd of the scores of the latent dimension
        :param n_frames: the number of frames for the animation
        :param kwargs: see helpers.animate_vectors
        :return:
        """


        file_name = kwargs.pop('file_name', 'PLS_Dim' + str(dim)+'.gif')
        # collect the average polydatas
        base_polydata = self._get_base_polydata()
        vectors = self._get_latent_vectors(dim)
        frame_scalars = self._get_frame_scalars(dim,max_sd=max_sd,n_frames=n_frames)

        helpers.animate_vectors(base_polydata=base_polydata,point_vectors=vectors,frame_scalars=frame_scalars,**kwargs)

    def colormap_latent_dim(self,dim,direction='normal',**kwargs):
        """
        :param dim: the latent dimension to visualise
        :param direction: plots the shape transformtion along the surface normals (if set to 'normal') otherwise plots the total magnitude of the shape transformation per point (if set to 'total')
        :param kwargs: see helpers.plot_colormaps
        :return:
        """
        file_name = kwargs.pop('file_name','PLS_Dim' + str(dim) + '.pdf'  )
        pd = self._get_base_polydata()
        scalars = self._get_point_scalars(dim,direction)
        helpers.plot_colormaps(pd,scalars,file_name=file_name,**kwargs)


class PLSHypothesisTest(PLS):
    def __init__(self):
        super().__init__()
        self._n_comp = None
        self._coefs = None
        self._method = None
        self._var_r_squared = None
        self._null_r_squared = None

    @property
    def coefs(self):
        return self._coefs

    @property
    def method(self):
        return self._method


    @property
    def model_stats(self):
        return self._assemble_model_stats()



    def fit(self, x, y, method='simpls',n_comp = None,center_x=True, center_y=True, standardize_x=False, standardize_y=False,observation_mask=None,
            observation_weights = None):
        self._center_x = center_x
        self._center_y = center_y
        self._standardize_x = standardize_x
        self._standardize_y = standardize_y
        self._observation_mask = observation_mask
        self._observation_weights = observation_weights
        self.x = x
        self.y = y
        self._method = method
        self._n_comp = n_comp
        self._coefs = self._fit(self._x0,self._y0,n_comp,method=self.method)
        self._var_r_squared = self._compute_var_r_squared()



    def run_permutation_test(self,vars_to_test = None,test_full_model = True,seed=None,n_reps=1000,n_jobs=True):
        perm_models = self._fit_permuted_models(vars_to_test=vars_to_test,test_full_model=test_full_model,seed=seed,n_reps=n_reps,n_jobs=n_jobs)
        self._null_r_squared = self._unpack_null_distributions(perm_models)


    def _unpack_null_distributions(self,perm_models):
        out = dict()
        keys = [item for item in perm_models.keys()]
        for x in tqdm(range(len(keys)), desc="Unpacking…", ascii=False, ncols=75):
            key = keys[x]
            out[key] = self._compute_rsquared(perm_models[key][0], perm_models[key][1])
        return out

    def _fit_permuted_models(self,vars_to_test = None,test_full_model = True,seed=None,n_reps=1000,n_jobs=True):
        def _run_parallel(x0,y0):
            ss = rng.bit_generator._seed_seq
            child_states = ss.spawn(n_reps)
            with joblib_progress('Running permutation test...', n_reps):
                res = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
                    joblib.delayed(self._perm_test_one_iter)(x0,y0,n_comp=None,method=self.method,
                                                             seed=child_states[x]) for x in range(n_reps))
            _,res = zip(*res)
            return res

        rng = np.random.default_rng(seed)
        # which variables will be tested
        if vars_to_test is None:
            vars_to_test = [i for i in range(self.x.shape[1])]
        else:
            if not helpers.my_is_iterable(vars_to_test):
                vars_to_test = [vars_to_test]
        vars_to_test = [self._find_var_in_x(item,search_in_x_treated=False) for item in vars_to_test]
        # for each variable get the results
        results = dict()
        for var in vars_to_test:
            x0,y0 = self._get_reduced_xy(var)
            results[self.x.columns[var]] = (_run_parallel(x0,y0),y0)
        if test_full_model:
            results['Full Model'] = (_run_parallel(self._x0,self._y0),self._y0)
        return results

    def _compute_rsquared(self,res,y0):
        if not helpers.my_is_iterable(res):
            res = [res]
        tot_var = np.sum(y0.flatten()**2)
        res_var = np.array([np.sum(item.flatten()**2) for item in res])
        return (1 - res_var / tot_var,)

    def _compute_var_r_squared(self):
        out = dict()
        for var in self.x.columns:
            x0,y0 = self._get_reduced_xy(self._find_var_in_x(var,False))
            _,res = self._fit_residualize(x0, y0, n_comp=None, method=self.method)
            out[var] = self._compute_rsquared(res, y0)
        return out

    @staticmethod
    def _perm_test_one_iter(x0,y0,n_comp,method,seed = None):
        rng = np.random.default_rng(seed)
        # permute rows of x0
        shuff_x = copy.copy(x0)
        shuff_x = shuff_x[rng.permutation(x0.shape[0]), :]
        return PLSHypothesisTest._fit_residualize(shuff_x, y0, n_comp=n_comp, method=method)

    def _get_reduced_xy(self,var):
        var_cols = [item==var for item in self._x_block_var_indices]
        co_var_cols = [item!=var for item in self._x_block_var_indices]
        var = self._x0[:, var_cols]
        co_var = self._x0[:, co_var_cols]
        if sum(co_var_cols) == 0:
            return copy.copy(self._x0),copy.copy(self._y0)
        _,x0 = self._fit_residualize(co_var, var, method=self.method)
        _,y0 = self._fit_residualize(co_var,self._y0,method = self.method)
        return x0, y0

    def _assemble_model_stats(self):
        if self._var_r_squared is None:
            return None
        out = pd.DataFrame(data=np.ones([len(self._var_r_squared),2])*np.nan,index=self._var_r_squared.keys(),columns=['R_2','p'])
        for key in self._var_r_squared.keys():
            R2=self._var_r_squared[key][0]
            out.loc[key,'R_2'] = R2
            if self._null_r_squared is not None:
                if key in self._null_r_squared.keys():
                    null_r2 = self._null_r_squared[key][0]
                    p = sum(null_r2>R2) / len(null_r2)
                    out.loc[key,'p'] = p
        return out
        # get variable



    @staticmethod
    def _fit(x0,y0,n_comp=None,method='simpls'):
        if n_comp is None:
            n_comp = min(x0.shape)
        if method == 'simpls':
            return simpls(x0, y0, n_comp)
        if method == 'pls':
            PLSMod = PLSRegression(n_components=n_comp,scale=False)
            PLSMod.fit(x0,y0)
            return PLSMod.coef_

    @staticmethod
    def _fit_residualize(x0,y0,n_comp = None,method = 'simpls'):
        coefs = PLSHypothesisTest._fit(x0,y0,n_comp=n_comp,method = method)
        res = y0 - PLSHypothesisTest._predict(coefs,x0)
        return coefs, res

    @staticmethod
    def _predict(coefs,x0):
        return x0 @ coefs
    def _get_regression_vectors(self, x_vars,reverse=False):
        if not helpers.my_is_iterable(x_vars):
            x_vars = [x_vars]
        return [self._get_regression_vector(item,reverse=reverse) for item in x_vars]
    def _get_regression_vector(self,x_var,reverse=False):
        row = self._find_var_in_x(x_var)
        vec = self.coefs[row,:]
        if reverse:
            vec = vec*-1
        return vec

    def _find_var_in_x(self,x_var,search_in_x_treated=True):
        if search_in_x_treated:
            x = self.x_treated
        else:
            x = self.x
        # if integer assume it is a numeric index to  row of coefs
        if isinstance(x_var, int):
                if x_var > (x.shape[1]-1):
                    raise ValueError('x_var appears to be an integer index, but is greater than the variables in x')
                row = x_var
        else:  # try to see if you can find it
            is_match = np.nonzero([item == x_var for item in x.columns])[0]
            if len(is_match) == 0:
                raise ValueError('No match for x_var (' + str(x_var) + ') found values' + str(
                    [item for item in x.columns]))
            if len(is_match) > 1:
                raise ValueError(
                    'Multiple matches for x_var (' + str(x_var) + ') found values' + str(
                        [item for item in x.columns]))
            row = is_match[0]
        return row
    @staticmethod
    def _get_residuals(coefs,x0,y0):
        return y0 - PLSHypothesisTest._predict(coefs,x0)

class ShapePLSHypothesisTest(PLSHypothesisTest):
    def __init__(self):
        super().__init__()
    @property
    def y(self):
        return super().y
    @y.setter
    def y(self,val):
        if not isinstance(val,ShapePCA):
            raise ValueError('y is expected to be an instance of the ShapePCA class')
        super(__class__, self.__class__).y.__set__(self, val)

    def _compute_rsquared(self,res,y0):
        ef_rsq = super()._compute_rsquared(res, y0)[0]
        # compute r squared per point
        if not helpers.my_is_iterable(res):
            res = [res]
        # rotate back to the space of landmarks and reshape
        res = [helpers.landmark_2d_to_3d(item @ self._y.eig_vec) for item in res]
        y0 = np.atleast_3d(helpers.landmark_2d_to_3d(y0 @ self._y.eig_vec))
        ss_per_point = lambda x : np.squeeze(np.sum(np.sum(x**2,axis=1),axis=1))[:,np.newaxis]
        tot_var = ss_per_point(y0)
        res_var = np.hstack([ss_per_point(item) for item in res])
        point_rsqu = 1 - res_var / tot_var
        return (ef_rsq,point_rsqu)
    def _get_regression_vector(self, x_var,reverse=False):
        vec = super()._get_regression_vector(x_var,reverse=reverse)
        return helpers.landmark_2d_to_3d(vec @ self._y.eig_vec)
    def _get_point_regression_coefs(self,direction,x_vars,reverse=False):
        if not helpers.my_is_iterable(x_vars):
            x_vars = [x_vars]
        vecs = self._get_regression_vectors(x_vars,reverse=reverse)
        return helpers._vecs_to_scalars(vecs,direction=direction,poly=self._y.average_polydata)

    def _get_point_r_squared(self,x_vars):
        if not helpers.my_is_iterable(x_vars):
            x_vars = [x_vars]
        # check variable exists
        [self._find_var_in_x(item,search_in_x_treated=False) for item in x_vars]
        return [self._var_r_squared[item][1] for item in x_vars]

    def _get_point_p_values(self,x_vars):
        if not helpers.my_is_iterable(x_vars):
            x_vars = [x_vars]
            # check variable exists
        [self._find_var_in_x(item, search_in_x_treated=False) for item in x_vars]
        out = []
        for var in x_vars:
            r = self._var_r_squared[var][1]
            null_rsqu = self._null_r_squared[var][1]
            n_perms = null_rsqu.shape[1]
            p = np.sum((null_rsqu-r)>0,axis=1) / n_perms
            out.append(p)
        return out

    def plot_coefficients(self,x_vars=None,direction = 'normal', **kwargs):
        """
        Plot the regression coefficients as colormaps

        :param x_vars: the names of the x variables plot
        :param direction: either 'normal' or 'total' plotting the magnitude of the coefficients along the surface normals or their total magnitude
        :param kwargs: see helpers.plot_colormaps
        """
        file_name = kwargs.pop('file_name','regression_coeffs')  # +str(x_vars).replace('[','').replace.(']','').replace(',','_').replace(' ','')
        pd = self._y.average_polydata
        point_scalars = self._get_point_regression_coefs(direction=direction, x_vars=x_vars)
        helpers.plot_colormaps(pd,point_scalars,file_name=file_name,**kwargs)
    def plot_r_squared(self,x_vars=None, **kwargs):
        """
        Plots the partial r-sqaured for the specified variables for each point

        :param x_vars: the names of the x variables plot
        :param kwargs: see helpers.plot_colormaps
        """
        file_name = kwargs.pop('file_name', 'regression_r_squared')
        pd = self._y.average_polydata
        point_scalars = self._get_point_r_squared(x_vars=x_vars)
        helpers.plot_colormaps(pd,point_scalars,file_name=file_name,**kwargs)

    def plot_p_values(self,x_vars=None,p_crit=0.05, **kwargs):
        """
        Plots the significance of of the effect at each point as a binary colormap

        :param x_vars: the names of the x variables plot
        :param p_crit: the significance threshold
        :param kwargs: see helpers.plot_colormaps

        """
        file_name = kwargs.pop('file_name','regression_point_p_values')
        cmap = kwargs.pop('cmap','summer')
        clim = kwargs.pop('clim',[0,1])
        plot_color_bar = kwargs.pop('plot_color_bar',False)

        pd = self._y.average_polydata
        point_scalars = [(item<p_crit).astype(float) for item in self._get_point_p_values(x_vars=x_vars)]
        helpers.plot_colormaps(pd,point_scalars,file_name = file_name,cmap = cmap,clim=clim,plot_color_bar=plot_color_bar,**kwargs)

    def animate_coefficients(self, vars, max_sd : float = 3, n_frames : int =20, **kwargs):
        """
        Visualise the regression coefficients as animations and save as a gif.

        :param vars: the names of the x variables plot
        :param max_sd: animate between the predicted shapes at +/- max_sd of standrad deviation of the predictor
        :param n_frames: the number of frames for the animation
        :param kwargs: see helpers.animate_vectors
        :return:
        """
        if not helpers.my_is_iterable(vars):
            vars = [vars]

        file_name = kwargs.pop('file_name','regression')  # +str(x_vars).replace('[','').replace.(']','').replace(',','_').replace(' ','')
        title = kwargs.pop('title', [str(i) for i in vars])

        # get the standard deviations
        sds = [np.std(self.x_treated.iloc[:,self._find_var_in_x(vars[i],search_in_x_treated=True)]) for i in range(len(vars))]

        vec = self._get_regression_vectors(vars)
        frame_scalars = [helpers._generate_circular_sequence(max_sd*sds[i], -max_sd*sds[i], n_in_sequence=n_frames) for i in range(len(vars))]
        polydatas = [copy.deepcopy(self._y.average_polydata) for i in range(len(vars))]
        # set default values of some of the keyword arguments
             # make animation
        helpers.animate_vectors(polydatas, vec, frame_scalars, file_name=file_name, title=title,
                                 **kwargs)












def simpls(x0, y0, n_comp):
    # this is a straightforward port of the simpls algorithm implemented in MATLABs plsregress

    n, dx = x0.shape
    dy = y0.shape[1]

    x_loadings = np.zeros([dx, n_comp])
    y_loadings = np.zeros([dy, n_comp])

    x_scores = np.zeros([n, n_comp])
    y_scores = np.zeros([n, n_comp])

    weights = np.zeros([dx, n_comp])
    r = np.zeros([dx, n_comp])  # not part of the algorithm, will collect ri vectors for my own interest
    v = np.zeros([dx, n_comp])

    cov = x0.T @ y0
    for i in range(n_comp):
        u, s, vt = np.linalg.svd(cov, full_matrices=0)
       # try:
        ri = u[:, 0]
   #     except IndexError:
     #       k = aa
        ci = vt[0, :]
        si = s[0]

        r[:, i] = ri

        ti = np.dot(x0, ri)  # projection onto ri
        normti = np.linalg.norm(ti)
        ti = ti / normti
        x_loadings[:, i] = np.dot(np.transpose(x0), ti)

        qi = si * ci / normti
        y_loadings[:, i] = qi

        x_scores[:, i] = ti
        y_scores[:, i] = np.dot(y0, qi)

        weights[:, i] = ri / normti

        # update orthonormal basis with modified Gramm-Schmidt
        vi = x_loadings[:, i]
        for repeat in range(2):
            for j in range(i):
                vj = v[:, j]
                vi = vi - np.dot(vj, vi) * vj

        vi = vi / np.linalg.norm(vi)
        v[:, i] = vi


        # deflate cov with respect to current vector
        cov = cov - np.outer(vi, np.dot(vi, cov))

        # deflate cov again with respect to all previous vectors to ensure complete deflation
        vi = v[:, 0:(i + 1)]
        if i == 0:  # Vi will be a single column, numpy will only do this using np.outer
            cov = cov - np.outer(vi, np.dot(np.transpose(vi), cov))
        else:
            cov = cov - np.dot(vi, np.dot(np.transpose(vi), cov))

    # Orthogonalise Y-scores
    for i in range(n_comp):
        ui = y_scores[:, i]
        for repeat in range(2):
            for j in range(i):
                tj = x_scores[:, j]
                ui = ui - np.dot(tj, ui) * tj

        y_scores[:, i] = ui


    coefs = np.atleast_2d(np.dot(weights,np.transpose(y_loadings)))
    return coefs

def pls_svd(cross_cov):
    [u, s, v] = np.linalg.svd(cross_cov, full_matrices=False)



