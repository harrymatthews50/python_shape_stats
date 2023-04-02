import unittest
from python_shape_stats import helpers
import pathlib
import numpy as np

class TestHelpers(unittest.TestCase):
    def test_landmarks_3d_to_2d_returns_2d_array_if_a_2_dim(self):
        a = np.random.uniform(0,1,[10,3])
        a_vec = helpers.landmark_3d_to_2d(a)
        self.assertTrue(np.all(np.equal(a_vec.shape,(1,30))))

    def test_landmark_3d_2d_handles_2_and_3d_arrays_equivalently(self):
        a_2 = np.tile([10,20,30],[9,1]) + np.tile(np.linspace(1,9,9)[:,np.newaxis],[1,3])
        a_3 = np.stack([a_2,a_2],axis=2)

        a_2_flat = helpers.landmark_3d_to_2d(a_2)
        a_3_flat = helpers.landmark_3d_to_2d(a_3)
        self.assertTrue(np.array_equal(a_2_flat.flatten(),a_3_flat[0,:]))

    def test_landmarks_3d_to_2d_returns_2d_array_if_a_3_dim(self):
        a = np.random.uniform(0, 1, [10, 3,20])
        a_vec = helpers.landmark_3d_to_2d(a)
        self.assertTrue(np.all(np.equal(a_vec.shape, (20, 30))))

    def test_landmarks_2d_to_3d_converts_back_correctly_if_a_2_dim(self):
        a = np.random.uniform(0, 1, [10, 3])
        a_vec = helpers.landmark_3d_to_2d(a)
        a_p = helpers.landmark_2d_to_3d(a_vec)
        self.assertTrue(np.all(np.equal(a,a_p)))

    def test_landmarks_2d_to_3d_converts_back_correctly_if_a_3_dim(self):
        a = np.random.uniform(0, 1, [10, 3,20])
        a_vec = helpers.landmark_3d_to_2d(a)
        a_p = helpers.landmark_2d_to_3d(a_vec)
        self.assertTrue(np.all(np.equal(a, a_p)))

    def test_load_pinnochio(self):
        shp,_,_ = helpers.load_shape(helpers._get_path_to_pinnochio_demo_face())
        self.assertEqual(shp.n_points, 7160)  # add assertion here

    def test_load_non_pinnochio(self):
        shp, _, _ = helpers.load_shape(helpers._get_path_to_demo_face())
        self.assertEqual(shp.n_points, 7160)  # add assertion here
    def test_generate_random_cov_matrix(self):
        rank = 10
        n_vars = 100
        cov, eig_vec, eig_vals = helpers._generate_random_cov_matrix(n_vars, rank, 15)
        self.assertTrue(np.array_equal(cov.shape,[n_vars,n_vars]))
        self.assertEqual(np.linalg.matrix_rank(cov),rank)
    def test_load_shapes_to_array(self):
        path = helpers._get_path_to_simulated_population()
        obj_paths=pathlib.Path(path).glob('*.obj')
        r = helpers.load_shapes_to_array(obj_paths,n_jobs = 1)


    def test_broken_stick_distribution(self):
        lengths=helpers.broken_stick_empirical(20,100000,seed=5)
        mu_lengths = np.mean(lengths,axis=0)
        bs = helpers.broken_stick_expectation(20)
        # check that the empirical values match closely analytical expectations with a slightly relaxed tolerance
        self.assertTrue(np.allclose(bs,mu_lengths,atol=1e-3))

    def test_rng_kfold_listens_rng(self):
        x = 90
        k = 3
        train_list =[]
        test_list = []
        for i in range(2):
            rng = np.random.default_rng(1441)
            ks = helpers._rng_kfold_split(k,rng)
            train,test = zip(*[(train,test) for _, (train, test) in enumerate(ks.split(np.linspace(0,1,x)))])
            train_list.append(train)
            test_list.append(test)
        # check the rng is being listened to and giving the same results over again
        train_equal = all([np.array_equal(train_list[0][i],train_list[1][i]) for i in range(k)])
        test_equal = all([np.array_equal(test_list[0][i], test_list[1][i]) for i in range(k)])
        self.assertTrue(all([train_equal,test_equal]))
    def test_rng_kfold_is_randomising(self):
        x = 90
        k = 3
        train_list = []
        test_list = []
        rng = np.random.default_rng(1441)
        for i in range(2):
            ks = helpers._rng_kfold_split(k, rng)
            train, test = zip(*[(train, test) for _, (train, test) in enumerate(ks.split(np.linspace(0, 1, x)))])
            train_list.append(train)
            test_list.append(test)
        # check the rng is being listened to and giving the same results over again
        train_equal = all([np.array_equal(train_list[0][i], train_list[1][i]) for i in range(k)])
        test_equal = all([np.array_equal(test_list[0][i], test_list[1][i]) for i in range(k)])
        self.assertFalse(all([train_equal, test_equal]))
if __name__ == '__main__':
    unittest.main()
