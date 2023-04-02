import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import spdiags
from scipy.stats import chi2
from python_shape_stats import helpers, procrustes
from sklearn.model_selection import KFold
import joblib
from joblib_progress import joblib_progress
from abc import ABC,abstractmethod,abstractproperty,abstractclassmethod,abstractstaticmethod
import os
import pyvista


class PCA:
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
    # @property
    # def dim_log_likelihood(self):
    #     """
    #     The log likelihood of the data given models of differing dimensionalities. Borrows from the implementation of PCA in scikit learn,
    #     which in turn implements the method of `this paper <https://tminka.github.io/papers/pca/minka-pca.pdf>.
    #     This is intended as a method for determining the number of principal components to retain.
    #
    #     Note: the i_{th} element of the output array corresponds to the log likelihood of the data give a model with i-1 dimensions
    #           for example the first element of the output array will always be -inf as it corresponds to model with zero dimensions.
    #           The optimal number of dimensions under this criteria is given by np.argmax(self.dim_log_likelihood)
    #
    #     :type: np.ndarray | NoneType
    #     """
    #     if self._initial_eig_val is None:
    #         return None
    #     out = np.zeros_like(self.eig_val)
    #     out[0] = -np.inf
    #     for x in range(1,len(self._initial_eig_val)):
    #         out[x]=decomposition._pca._assess_dimension(self._initial_eig_val,x,self._n_samples)
    #     return out

    @property
    def eig_vec(self) -> np.ndarray | None:
        """
        The right singular vectors of the data matrix (the PCs). The first dimension corresponds to PCs, the second to features.

        :type: np.ndarray | NoneType
        """
        return self._eig_vec

    @property
    def eig_val(self):
        """
        The total variance explained by each PC. This is computed from the singular values of the training data matrix
        :math:`\\frac{s^2}{k}` where :math:`s` is the singular values and :math:`k` is the number of observations (rows in the training data matrix). If x is column-mean centered in the call to 'fit' This is equal to the sample variance in each dimension of the transformed data.

        :type: np.ndarray | NoneType
        """

        return self._eig_val

    @property
    def eig_std(self):
        """
        The square root of 'self.eig_val'. If x was column mean centered during the call to 'self.fit' these are the sample standard deviations of the transformed training data

        :type: np.ndarray | NoneType
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

        :type: np.ndarray | NoneType
        """
        return self._transformed_training_data


    @property
    def center_vec(self):
        """
        The vector used to center the columns of x. Using default keyword arguments to 'fit' this is the column mean of
        the training data matrix

        :type: np.ndarray | None
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

        :type: np.ndarray | None
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

    @property
    def cv_error_per_dim(self):
        """
        The mean squared error of cross-validation per model dimension
        The ith entry corresponds the error from a model with i-1 dimensions.
        For example the first entry contains the error of a model with zero dimensions (i.e. the mean only)
        -------
        :note: run self.cross_validation before trying to access this property

        :type: np.ndarray | NoneType
        """
        if self._cross_validation_results is None:
            UserWarning('Cross-validation needs to be run before the error can be calculated, use the \'cross validation\' of this class')
            return None
        return np.mean(np.mean(self._cross_validation_results**2,axis=0),axis=0)

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

    @staticmethod
    def _cross_validation_one_fold(train, test, params):
        mod = PCA()
        mod.fit(train, **params)
        sc = mod.transform(test)
        # generate predictions for all possible dimensionalities
        predictions = np.zeros([test.shape[0],test.shape[1],mod.n_dim])
        for d in range(1,mod.n_dim):
            for_sc = sc.copy()
            for_sc[:, d:] = 0  # suppress PCs d: end from influencing the prediction
            predictions[:, :, d] = mod.predict(for_sc)
        return np.abs(test[:,:,np.newaxis] - predictions)

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

    def parallel_analysis(self, n_reps=50, n_jobs=1, seed=None):
        # generate independent random number generators for parallel processing
        # following these recommendataions: https://albertcthomas.github.io/good-practices-random-number-generators/
        rng = np.random.default_rng(seed)
        ss = rng.bit_generator._seed_seq
        child_states = ss.spawn(n_reps)
        with joblib_progress('Running parallel analysis...', n_reps):
            args = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
                joblib.delayed(self._parallel_analysis_one_iter)(self._params['x0'], child_states[x])
                for x in range(n_reps))
        self._parallel_analysis_results = np.vstack(args)
        return self._parallel_analysis_results

    # def cross_validation(self, k=10, n_jobs=1, seed=None):
    #     if k == 'L00':  # do leave one out
    #         k = self._n_samples
    #     # create an instance of KFold
    #     kf=helpers._rng_kfold_split(k,seed)
    #     # get the original training data matrix
    #     data = self._params['x']
    #     train_inds, test_inds,train_x,test_x= zip(*[(train, test,data[train,:],data[test,:]) for i, (train, test) in enumerate(kf.split(data))])
    #
    #     # retrieve the necessary parameters
    #     params = copy.deepcopy(self._params)
    #     params.pop('x')
    #     params.pop('x0')
    #     with joblib_progress('Running cross-validation', k):
    #         errors = joblib.Parallel(n_jobs=n_jobs, verbose=0)(
    #             joblib.delayed(self._cross_validation_one_fold)(train_x[x], test_x[x],params) for x in range(k))
    #     test_i = np.squeeze(np.hstack(test_inds))
    #
    #     # the results might have slighlt different numbers of dimensions (as the number of observations per training fold may differ)
    #     # so we will normalise the size of these
    #     errors = helpers._trim_arrays_to_min_size(errors,axis=2)
    #     errors = np.vstack(errors)
    #     errors[test_i,:] = errors
    #     self._cross_validation_results = errors

    def scree_plot(self, ax=None):
        return _eigen_value_plot(self._initial_eig_val, title='Scree Plot', ax=ax)

    def cumulative_variance_plot(self, ax=None):
        return _eigen_value_plot(np.cumsum(self._initial_eig_val) / self._initial_var * 100,eig_vals_label='Cumulative Var. Exp.',
                                 title='Cumulative Variance', ylabel='Variance\nExplained (%)', ax=ax)

    def parallel_analysis_plot(self, ax=None, ci_level=95, threshold_level=97.5, n_reps=50, n_jobs=1,
                               recompute_parallel_analysis=False, seed=None):
        # determine whether the empirical null distribution needs to be recalculated
        if (self._parallel_analysis_results is None) | recompute_parallel_analysis:
            self.parallel_analysis(n_reps=n_reps, n_jobs=n_jobs, seed=seed)
        return _eigen_value_plot(self._initial_eig_val,distr= self._parallel_analysis_results,distr_label='Null Spectra', ci_level=ci_level,
                                 threshold_level=threshold_level, ax=ax, title='Parallel\nAnalysis')

    def broken_stick_plot(self, ax=None, ci_level=95, threshold_level=97.5, n_reps=1000):
        # get the empirical broken stick distribution
        N = len(self._initial_eig_val)
        lengths = helpers.broken_stick_empirical(N, n_reps) * self._initial_var
        return _eigen_value_plot(self._initial_eig_val, distr=lengths,distr_label='Null Spectra', ax=ax, ci_level=ci_level,
                                 threshold_level=threshold_level, title='Broken Stick')

    # def cross_validation_plot(self,ax = None,atol=1e8,rtol=1e5,recompute_cross_validation=False,k=10,seed=None,n_jobs=None):
    #     if recompute_cross_validation | (self._cross_validation_results is None):
    #         self.cross_validation(k=k,n_jobs=n_jobs,seed=seed)
    #
    #     cv_err = self.cv_error_per_dim
    #     ax,_ = _eigen_value_plot(cv_err[1:],eig_vals_label='Error',ylabel='Cross Validation\nMSE',title='Cross Validation')
    #     min_err = np.min(cv_err)
    #     # get the earliest error that is within the specified tolerance of the min value
    #     inds = np.nonzero(np.isclose(cv_err,np.tile(min_err,len(cv_err)),atol=atol,rtol=rtol))
    #     n_comps = inds[0][0]
    #     ax.axvline(x=n_comps, c='k', ls=':', label='Estimated No. Comp.')
    #     ax.legend()
    def trim_no_pcs(self, no_pcs: int):
        """
        Modifies the object in situ, removing the specified number of pcs

        :param no_pcs: the number of pcs to retain
        :return: None
        """
        self._eig_vec = self._eig_vec[0:no_pcs, :]
        self._eig_val = self._eig_val[0:no_pcs]
        self._transformed_training_data = self._transformed_training_data[:, 0:no_pcs]

    def trim_perc_var(self, pct_var: float) -> None:
        """
        Modifies the object in situ, removing the number of pcs explaining up to the specified amount of variance

        :param pct_var: the percentage of variance s to retain
        :return: None
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
        of a given feature on the solution. the solution can also be constrained to lie within a specified Mahalanobis
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

    def load_from_folder(self, path=None):
        pass

    def weighted_transform_to_model(self, x: np.ndarray, weights: np.ndarray, max_m_dist: float = None) -> np.ndarray:
        pass

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
        :param apply_procrustes_transform: if True each observation in x will be aligned first to the mean shape using the
        Procrustes transform. If false then the shapes must already be aligned to the mean shape before calling this function
        :param procrustes_scale: if apply_procrustes_transform is True then this will determine if the configuratioons are
        allowed to scale towards the mean shape
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

    def animate_pc(self, pc_num, max_sd, n_frames=20, **kwargs):
        # for safety make clone
        kwargs = copy.deepcopy(kwargs)
        vec = helpers.landmark_2d_to_3d(self.eig_vec[pc_num, :] * self.eig_std[pc_num])
        frame_scalars = helpers._generate_circular_sequence(max_sd, -max_sd, n_in_sequence=n_frames)
        mode = kwargs.pop('mode', 'write_gif')
        file_name = kwargs.pop('file_name', 'PC_' + str(pc_num))
        off_screen = kwargs.pop('off_screen', False)
        if mode == 'write_gif':
            ext = '.gif'
            file_name = os.path.splitext(file_name)[0] + ext
        helpers.animate_vector(self.average_polydata, vec, frame_scalars, mode=mode, file_name=file_name,
                               off_screen=off_screen, **kwargs)


def _eigen_value_plot(eig_vals,eig_vals_label='Eigenvalue Spectrum', distr=None, distr_label='', ci_level=95., threshold_level=95, ax=None, xlabel='PC',
                      ylabel='Explained\nVariance', title=''):
    if ax is None:
        ax = plt.subplot()
    x = np.linspace(1, len(eig_vals), len(eig_vals))
    ax.plot(x, eig_vals, 'b+-',label=eig_vals_label)

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
        inds = np.nonzero(np.less(eig_vals, pctiles[2, :]))
        n_comps = np.min(inds)
        ax.axvline(x=n_comps, c='k', ls=':', label='Estimated No. Comp.')
    else:
        n_comps = []
    ax.legend()
    return ax, n_com
class PLS(ABC):
    def __init__(self):
        super().__init__()
        self._x = None
        self._y = None
        self._standardize_x = None
        self._standardize_y = None
        self._center_x = None
        self._center_y = None
        self._observation_mask = None
        self._observation_weights = None
        self._x_mu
        self._y_mu
        self._x_std
        self._y_std

    @property
    @abstractmethod
    def x(self):
        return self._x

    @property
    @abstractmethod
    def y(self):
        return self._y

    @property
    def _x0(self):
        if self.x is None:
            return None
        x0 = self.x
        if self.center_x:
            x0 = x0-self.x_mu
        if self.standardize_x:
            x0 = x0 / self.x_std
        x0 *=self.observation_weights
        x0 = x0[self.observations_mask,:]
        return x0

    @property
    def _y0(self):
        y0 = self.y
        if self.center_y:
            y0 = y0 - self.y_mu
        if self.standardize_y:
            y0 = y0 / self.y_std
        y0 *= self.observation_weights
        y0 = y0[self.observation_mask, :]
        return y0

    @property
    def observation_weights(self):
        if self._observation_weights is None:
            if self.x is None:
                return None
            return np.ones(self.x.shape[0])
        else:
            return self._observation_weights

    @property
    def observation_mask(self):
        if self._observation_mask is None:
            if self.x is None:
                return None
            return np.ones(self.x.shape[0],dtype=bool)
        else:
            return self._observation_mask



    @property
    def no_y_features(self):
        if self._y0 is None:
            return None
        return self._y0.shape[1]


    @property
    def no_x_features(self):
        if self._x0 is None:
            return None
        return self._x0.shape[1]

    @property
    def x_mu(self):
        if self._x_mu is None: # if the mean has not been otherwise explicitly set
            return helpers.weighted_column_mean(self.x[self._observation_mask,:],self.observation_weights[self.observation_mask])
    @property
    def x_std(self):
        if self.
    @property
    def y_mu(self):

    @property
    def y_std(self):

    def

    @abstractmethod
    def fit(self):
        ...





class ShapePLSRegression():
    pass


# print(__name__)
if __name__ == '__main__':
    obj = PCA()
    obj2 = ShapePCA()
