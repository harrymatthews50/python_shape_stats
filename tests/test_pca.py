import unittest
import numpy as np
from python_shape_stats import helpers, procrustes,statistical_shape_models
import pathlib
import copy
import sklearn
import matplotlib.pyplot as plt


class TestSSM(unittest.TestCase):
    def setUp(self) -> None:
        path = helpers._get_path_to_simulated_population()
        obj_paths = pathlib.Path(path).glob('*.obj')
        r = helpers.load_shapes_to_array(obj_paths, n_jobs=1,verbose=0)
        r = [procrustes.apply_procrustes_transform(r[:,:,i],helpers._random_transformation()['matrix']) for i in range(r.shape[2])]
        r = np.stack(r,axis=2)
        self.original_shapes = copy.copy(r)
        r = procrustes.do_generalized_procrustes_analysis(r,scale=True,max_iter=5,verbose=0)
        self.shapes = r

    def test_animate_pc(self):
        mod = statistical_shape_models.ShapePCA()
        pd,_,_ = helpers.load_shape(helpers._get_path_to_demo_face())
        mod.reference_polydata = pd
        mod.fit(self.shapes['landmarks'], center=True, center_config=self.shapes['mean'])
        mod.animate_pc(1,3)
        self.assertTrue(True)

    def test_pca_fit_runs(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit(shapes_vec)
        self.assertTrue(True)

    def test_shape_pca_fit_runs(self):
        mod = statistical_shape_models.ShapePCA()
        mod.fit(self.shapes['landmarks'],center=True,center_config=self.shapes['mean'])
        self.assertTrue(True)

    def test_pca_transform_predict(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        p = mod.predict(mod.transformed_training_data)
        self.assertTrue(np.allclose(shapes_vec,p))


    def test_shape_pca_transform_predict_runs(self):
        mod = statistical_shape_models.ShapePCA()
        mod.fit_transform(self.shapes['landmarks'],center=True,center_config=self.shapes['mean'])
        self.assertTrue(True)

    def test_shape_pca_transform_runs(self):
        mod = statistical_shape_models.ShapePCA()
        mod.fit_transform(self.shapes['landmarks'],center=True,center_config=self.shapes['mean'])
        #assert(np.array_equal(self.shapes['mean'],helpers.landmark_2d_to_3d(mod.center_vec)))
        scores,_ = mod.transform(self.original_shapes,apply_procrustes_transform=True,procrustes_scale=True)
        self.assertTrue(np.array_equal(scores,mod.transformed_training_data))

    def test_agreement_between_eig_val_and_score_variance(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        self.assertTrue(np.allclose(mod.eig_val,np.var(mod.transformed_training_data,axis=0,ddof=1)))
    def test_log_lik(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        shapes_vec = shapes_vec[:,1:10]

        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        n_comp = np.argmax(mod.dim_log_likelihood)
        # compare to scikit
        mod2 = sklearn.decomposition.PCA(n_components='mle',svd_solver='full')
        mod2.fit_transform(shapes_vec)

    def test_trim_n_pcs(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        mod.trim_no_pcs(15)
        self.assertTrue(mod.n_dim==15)
        self.assertTrue(len(mod.eig_val)==15)
        self.assertTrue(mod.transformed_training_data.shape[1]==15)
        self.assertTrue(np.allclose(mod.eig_val, np.var(mod.transformed_training_data, axis=0, ddof=0)))

    def test_trim_strip_perc_var_removes_pcs(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        n_dim = mod.n_dim
        mod.trim_perc_var(78)
        n_dim2 = mod.n_dim
        self.assertTrue(sum(mod.cumulative_perc_var>=78)==1)
    def test_parallel_analysis_single_iter(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        out = mod._parallel_analysis_one_iter(mod.params['x0'])
        self.assertTrue(True)
    def test_maha_dist_to_p_value(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        mod.trim_no_pcs(1)
        p = mod.maha_dist_to_p_value(1.959963984540054)
        self.assertTrue(np.isclose(p,0.05))
        m = mod.p_value_maha_dist(p)
        self.assertTrue(np.isclose(m, 1.959963984540054))

    def test_weighted_transform_with_equal_weights_agrees_with_transform(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        sc1 = mod.transformed_training_data[0,:]
        w = np.ones(mod._n_train_features) / 3
        sc2 = mod.weighted_transform_to_model(shapes_vec[0,:],w)
        self.assertTrue(np.allclose(sc1,sc2))

    def test_scale_vec(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        sc1 = mod.transformed_training_data[0, :]
        sc2 = mod.scale_vec(sc1,target_dist=1,metric = 'euclidean')
        d1 = mod.get_distance(sc2,metric='euclidean')
        sc2 = mod.scale_vec(sc1, target_dist=1, metric='mahalanobis')
        d2 = mod.get_distance(sc2,metric='mahalanobis')
        self.assertTrue(np.allclose([d1,d2],[1.,1.]))

    # def test_eigenvalue_plot_runs(self):
    #     shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
    #     mod = statistical_shape_models.PCA()
    #     mod.fit_transform(shapes_vec)
    #     distr = helpers.broken_stick_empirical(mod.n_dim,10000,seed=1)*mod._initial_var
    #     statistical_shape_models._eigen_value_plot(mod.eig_val,distr=distr)
    #     plt.show()

    def test_broken_stick_plot_runs(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        mod.broken_stick_plot()
        plt.show()
    def test_parallel_analysis_runs(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        ax,np_pcs = mod.parallel_analysis_plot()
       # ax.set_ylim(np.array([-0,0.2])*1e-7)
        plt.show()

    def test_cross_validation_runs(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        mod.cross_validation()

    def test_cv_plot(self):
        shapes_vec = helpers.landmark_3d_to_2d(self.shapes['landmarks'])
        mod = statistical_shape_models.PCA()
        mod.fit_transform(shapes_vec)
        mod.cross_validation_plot(atol=1e-20)
        plt.show()


if __name__ == '__main__':
    unittest.main()
