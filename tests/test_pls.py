import unittest
from python_shape_stats.statistical_shape_models import PLSHypothesisTest,PCA,PLS_2B,ShapePLSHypothesisTest
from python_shape_stats import helpers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSSVD

class test_pls_hyp_test(unittest.TestCase):
    def setUp(self):
        x = np.random.uniform(0, 1, [20, 3])
        cats = np.array(['Blue', 'Red', 'Green'])
        n_cats = len(cats)
        cats = cats[np.random.randint(0, 3, 20, dtype=int)]
        mask = np.random.randint(0, 2, 20, dtype=bool)
        x_PCA = PCA()
        x_PCA.fit(x)
        x_df = pd.concat([pd.DataFrame(data=x), pd.DataFrame(data=cats, dtype='category')], axis=1)
        x_df.columns = [0,1,2,3]
        self.x_array = x
        self.x_PC = x_PCA
        self.mask = mask
        self.x_df = x_df
        self.n_cats = n_cats
        self.weights = np.random.uniform(0, 1, 20)
    def test_fit_runs(self):
        PLSH = PLSHypothesisTest
        PLSH.fit(self.x_df,self.x_df)
        self.assertTrue(True)

    def test_fit_residualise(self):
        PLSH = PLSHypothesisTest()
        PLSH.fit(self.x_df, self.x_df)
        for method in ['simpls','pls']:
            cf,res = PLSH._fit_residualize(PLSH._x0,PLSH._y0,method=method)
            self.assertTrue(np.allclose(res,0))


    def test_perm_test(self):
        PLSH = PLSHypothesisTest()
        PLSH.fit(self.x_df, self.x_df)
        perm_models = PLSH.run_permutation_test(vars_to_test=None, test_full_model=True, seed=None,
                                                n_reps=100, n_jobs=1)

    def test_r_squared(self):
        PLSH = PLSHypothesisTest()
        PLSH.fit(self.x_df, self.x_df)
        # since x any are identical all partial and total R-squared should be one
        rsqu = PLSH._var_r_squared
        self.assertTrue(np.allclose([item[1] for item in iter(rsqu.items())],1.))

    def test_assemble_model_stats_before_permutatation_test(self):
        PLSH = PLSHypothesisTest()
        PLSH.fit(self.x_df, self.x_df)
        st = PLSH.model_stats
        self.assertTrue(True)

    def test_assemble_model_stats_after_permutatation_test(self):
        PLSH = PLSHypothesisTest()
        PLSH.fit(self.x_df, self.x_df)
        PLSH.run_permutation_test(vars_to_test=['0'])
        st = PLSH.model_stats
        self.assertTrue()

    def test_r_squared_shape_hyp_test(self):
        pca_mod = helpers.load_shape_pca()
        x = pd.read_excel(helpers.get_path_to_simulated_metadata(),index_col=0)
        shapePLS = ShapePLSHypothesisTest()
        shapePLS.fit(x,pca_mod)
        shapePLS.run_permutation_test(n_reps = 10)
        shapePLS._get_point_p_values(shapePLS.x_treated.columns)






class test_pls_svd(unittest.TestCase):
    def setUp(self):
        X = np.array([[0., 0., 1.],
                      [1., 0., 0.],
                      [2., 2., 2.],
                      [2., 5., 4.]])
        Y = np.array([[0.1, -0.2],
                      [0.9, 1.1],
                      [6.2, 5.9],
                      [11.9, 12.3]])
        self.X = X
        self.Y = Y
    def test_compare_to_scipy(self):

        pls = PLSSVD(n_components=2,scale=False).fit(self.X, self.Y)
        pls_b = PLS_2B()
        pls_b.fit(self.X,self.Y)
        # check x_rotations agree up to the sign
        self.assertTrue(np.allclose(np.abs(pls.x_weights_), np.abs(pls_b.x_projection_matrix)))
        self.assertTrue(np.allclose(np.abs(pls.y_weights_), np.abs(pls_b.y_projection_matrix)))
        xsc,ysc = pls.transform(self.X,self.Y)
        self.assertTrue(np.allclose(np.abs(ysc), np.abs(pls_b.y_scores)))
        self.assertTrue(np.allclose(np.abs(xsc), np.abs(pls_b.x_scores)))

    def test_inner_relation_model(self):
        x_scores = self.X
        # simulate some y-scores that are simply a scaling of the x-scores - so ifd the code is correct one should be perfectly predictable
        # from the other
        n_obs,n_dims=x_scores.shape
        sc=np.random.lognormal(0,1,n_dims)
        y_scores = x_scores*sc[np.newaxis,:]
        mod = PLS_2B()
        mod._x_scores = x_scores
        mod._y_scores = y_scores
        mod._inner_relation_coefs = mod.fit_inner_relation(x_scores,y_scores,np.ones(n_obs))
        self.assertTrue(np.allclose(sc,mod.inner_relation_coefs))

        # check prediction of x scores from y_scores
        x_hat = mod.predict_x_scores_from_y(y_scores)
        self.assertTrue(np.allclose(x_hat,x_scores))

        # check prediction of y scores from x scores
        y_hat=mod.predict_y_scores_from_x(x_scores)
        self.assertTrue(np.allclose(y_hat,y_scores))


    def test_permutation_test(self):
        import numpy as np

        n = 500
        # 2 latents vars:
        l1 = np.random.normal(size=n)
        l2 = np.random.normal(size=n)

        latents = np.array([l1, l1, l2, l2]).T
        X = latents + np.random.normal(size=4 * n).reshape((n, 4))
        Y = latents + np.random.normal(size=4 * n).reshape((n, 4))

        X_train = X[: n // 2]
        Y_train = Y[: n // 2]

        pls_b = PLS_2B()
        pls_b.fit(X_train, Y_train)
        pls_b.compute_null_distribution(n_reps=100)
        pls_b.perm_test_p_values
        pls_b.permutation_test_plot()
        plt.show()




class TestAbstractDataHandling(unittest.TestCase):
    """Test the data handling implemented in the abstract _PLS class"""
    def setUp(self):
        # generate some data
        x = np.random.uniform(0,1,[20,3])
        cats = np.array(['Blue','Red','Green'])
        n_cats = len(cats)
        cats = cats[np.random.randint(0,3,20,dtype=int)]
        mask = np.random.randint(0,2,20,dtype=bool)

        x_PCA = PCA()
        x_PCA.fit(x)

        x_df = pd.concat([pd.DataFrame(data=x),pd.DataFrame(data=cats,dtype='category')],axis=1)

        self.x_array = x
        self.x_PC = x_PCA
        self.mask = mask
        self.x_df = x_df
        self.n_cats = n_cats
        self.weights = np.random.uniform(0,1,20)




    def test_init(self):
        obj = PLSHypothesisTest()
        self.assertTrue(True)  # add assertion here


    # foregoing function test the behaviour of the x and y getters
    def test_xy_getter_returns_data_frame_when_PC(self):
        obj = PLSHypothesisTest()
        obj.x = self.x_PC
        obj.y = self.x_PC

        self.assertTrue(isinstance(obj.x,pd.DataFrame))
        self.assertTrue(isinstance(obj.y,pd.DataFrame))

    def test_xy_getter_returns_data_frame_when_array(self):
        obj = PLSHypothesisTest()
        obj.x = self.x_array
        obj.y = self.x_array
        self.assertTrue(isinstance(obj.x, pd.DataFrame))
        self.assertTrue(isinstance(obj.y, pd.DataFrame))

    def test_xy_getter_returns_data_frame_when_data_frame(self):
        obj = PLSHypothesisTest()
        obj.x = self.x_df
        obj.y = self.x_df
        self.assertTrue(isinstance(obj.x, pd.DataFrame))
        self.assertTrue(isinstance(obj.y, pd.DataFrame))

    def test_x_treated_is_data_frame_when_data_frame(self):
        obj = PLSHypothesisTest()
        obj.x = self.x_df
        obj.y = self.x_df
        self.assertTrue(isinstance(obj.x_treated, pd.DataFrame))
        self.assertTrue(isinstance(obj.y_treated, pd.DataFrame))

    # test the behaviour of the data expansion in the presence of categorical variables
    def test_data_expands_when_variables_categorical(self):
        obj = PLSHypothesisTest()
        obj.x = self.x_df
        obj.y = self.x_df
        # get treated x
        v_x = obj.x_treated.shape[1]
        v_y = obj.y_treated.shape[1]
        exp_v = self.x_df.shape[1]-1+self.n_cats-1
        exp_indices = np.array([0,1,2,3,3])
        self.assertTrue(v_x==exp_v)
        self.assertTrue(v_y == exp_v)
        self.assertTrue(np.all(np.equal(obj._x_block_var_indices,exp_indices)))
        self.assertTrue(np.all(np.equal(obj._y_block_var_indices, exp_indices)))

    # test if user mask is listened to
    def test_user_mask_listened_to(self):
        obj = PLSHypothesisTest()
        obj.x = self.x_df
        obj.y = self.x_df
        obj.observation_mask = self.mask
        k = obj.n_obs
        o = sum(self.mask)
        #
        self.assertTrue(obj.n_obs == o)
        self.assertTrue(obj.x_treated.shape[0]==o)
        self.assertTrue(obj.y_treated.shape[0] == o)
        self.assertTrue(obj._x0.shape[0] == o)
        self.assertTrue(obj._y0.shape[0] == o)

    def test__x_and_yblock_cat_dtype_squeezed(self):
        obj = PLSHypothesisTest()
        obj.x = self.x_df
        obj.y = self.x_df
        obj.observation_mask=(self.x_df.iloc[:,-1]!='Red').to_numpy(dtype=bool)
        self.assertTrue(len(obj._xblock_data_types.iloc[-1].categories)==2)
        self.assertTrue(len(obj._yblock_data_types.iloc[-1].categories) == 2)

    def test_weighted_mean_and_std_computation_correct(self):
        obj = PLSHypothesisTest()
        obj.x = self.x_df
        obj.y = self.x_df
        obj.observation_weights = np.ones(obj.n_obs)*.5
        self.assertTrue(np.allclose(obj.x_mu,np.mean(obj.x_treated.astype(float))))
        self.assertTrue(np.allclose(obj.x_std, np.std(obj.x_treated.astype(float), ddof=0)))

    def test_weights_affect_computation(self):
        obj = PLSHypothesisTest()
        obj.x = self.x_df
        obj.y = self.x_df
        obj.observation_weights = self.weights
        self.assertFalse(np.allclose(obj.x_mu, np.mean(obj.x_treated.astype(float))))
        self.assertFalse(np.allclose(obj.x_std, np.std(obj.x_treated.astype(float), ddof=0)))

    def test_standardizing_and_centering_x0_y0(self):
        rms = lambda x : np.sqrt(np.mean(x**2,axis=0))
        for center_x in [True,False]:
            for center_y in [True,False]:
                for standardize_x in [True,False]:
                    for standardize_y in [True,False]:
                        obj = PLSHypothesisTest()
                        obj.x = self.x_df
                        obj.y = self.x_df
                        obj._center_x=center_x
                        obj._center_y=center_y
                        obj._standardize_x=standardize_x
                        obj._standardize_y=standardize_y

                        if center_x:
                            self.assertTrue(np.allclose(np.mean(obj._x0,axis=0),0))
                        else:
                            self.assertFalse(np.allclose(np.mean(obj._x0,axis=0),0))

                        if center_y:
                            self.assertTrue(np.allclose(np.mean(obj._y0,axis=0),0))
                        else:
                            self.assertFalse(np.allclose(np.mean(obj._y0,axis=0),0))

                        if standardize_x:
                            self.assertTrue(np.allclose(rms(obj._x0), 1))
                        else:
                            self.assertFalse(np.allclose(rms(obj._x0), 1))
                        if standardize_y:
                                self.assertTrue(np.allclose(rms(obj._y0), 1))
                        else:
                                self.assertFalse(np.allclose(rms(obj._y0), 1))
                        # test reverse function of self._center_scale


    def test_invert_center_scale(self):
        for center_x in [True,False]:
            for center_y in [True,False]:
                for standardize_x in [True,False]:
                    for standardize_y in [True,False]:
                        obj = PLSHypothesisTest()
                        obj.x = self.x_df
                        obj.y = self.x_df
                        obj._center_x=center_x
                        obj._center_y=center_y
                        obj._standardize_x=standardize_x
                        obj._standardize_y=standardize_y
                        y = obj.y_treated
                        y0 = obj._center_scale_y(y)
                        y_hat = obj._center_scale_y(y0, reverse=True)
                        self.assertTrue(np.allclose(y, y_hat))

                        X = obj.x_treated
                        X0 = obj._center_scale_x(X)
                        x_hat = obj._center_scale_x(X0, reverse=True)
                        self.assertTrue(np.allclose(X, x_hat))

if __name__ == '__main__':
    unittest.main()
