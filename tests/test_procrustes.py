import unittest
import numpy as np
import matplotlib.pyplot as plt
from python_shape_stats import procrustes, helpers
import copy


class TestProcrustes(unittest.TestCase):
    def setUp(self):
        # generate a random landmark configuration
        self.p = np.random.uniform(-10,10,[15,3])

        trans = helpers.random_transformation()
        t = trans['translation']
        s = trans['scaling']
        r = trans['rotation']

        q = copy.copy(self.p)
        center_q = np.mean(q,axis=0)
        q -= center_q
        q = q @ r
        q += s
        q += center_q + t

        self.q = q
        self.translation = t
        self.rotation = r
        self.scaling = s

    def test_is_rotation_orthonormal(self):
        r = procrustes.compute_rotation(self.p,self.q)
        self.assertTrue(np.allclose(r @ r.T,np.identity(4)))  # add assertion here

    def test_is_rotation_not_reflection(self):
        # transform by a reflection and rotation matrix
        trans = helpers.random_transformation(include_reflection=True)
        r = trans['rotation']
        p = self.p - np.mean(self.p,axis=0)
        p_ref = p @ r
        r_p = procrustes.compute_rotation(p,p_ref)
        #  check that the reflection part is discarded
        self.assertTrue(np.allclose(np.linalg.det(r_p),1.))

    def test_is_rotation_correct(self):
        r = procrustes.compute_rotation(self.p,self.q)
        self.assertTrue(np.allclose(self.rotation,r[0:3,0:3]))

    def test_apply_transform(self):
        # (only) rotate p
        p0 = (self.p-np.mean(self.p,axis=0))
        p_rot = p0 @ self.rotation
        r = procrustes.compute_rotation(p0,p_rot)
        pp = procrustes.apply_procrustes_transform(p0,r)
        self.assertTrue(np.allclose(pp,p_rot))

    def test_make_translation_matrix(self):
        destination = np.random.uniform(-10, 10, [1, 3])
        origin = np.random.uniform(-10, 10, [1, 3])
        p0 = (self.p - np.mean(self.p, axis=0)) + origin # center on origin
        t = procrustes.make_translation_matrix(origin=origin,destination=destination)
        pp = procrustes.apply_procrustes_transform(p0,t)
        pp_c =np.mean(pp,axis=0)
        self.assertTrue(np.allclose(destination,pp_c))

    def test_compute_procrustes_transform(self):
        t = procrustes.compute_procrustes_transform(self.p,self.q,scale = True)
        pp = procrustes.apply_procrustes_transform(self.p,t)
        self.assertTrue(np.allclose(pp,self.q))

    def test_scale_shape(self):
        p = self.p
        target_size = np.random.uniform(0,10)
        pp = procrustes.scale_shape(p,target_size=target_size)
        self.assertTrue(np.allclose(procrustes.compute_centroid_size(pp),target_size))

    def test_do_generalized_procrustes_analysis(self):
        landmarks = np.zeros([self.p.shape[0],3,100])
        for x in range(100):
            t = helpers.random_transformation()
            landmarks[:,:,x] = procrustes.apply_procrustes_transform(self.p,t['matrix'])

        res = procrustes.do_generalized_procrustes_analysis(landmarks)

        # check if the configurations match
        match = np.zeros([100,100],dtype = bool)
        for row in range(100):
            for col in range(row,100):
                a = landmarks[:,:,row]
                b = landmarks[:,:,col]
                match[row,col] = np.allclose(a,b)

        # check if are all the same now
        match = match[np.triu_indices(100)]
        self.assertTrue(np.all(match))





















if __name__ == '__main__':
    unittest.main()
