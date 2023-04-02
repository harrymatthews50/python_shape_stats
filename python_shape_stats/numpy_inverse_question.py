import numpy as np
from scipy.stats import ortho_group

def generate_random_cov_matrix(sz, rank, eig_val_scale):
    """
    Generate a random covariance matrix with known rank by building it from a specified number of eigenvectors and eigenvalues

    :param sz: the returned covariance matrix with be shaped sz x sz
    :param rank: the desired rank of the returned matrix
    :param eig_val_scale: eigenvalues will be randomly sampled from a chi2 distribution (df=1) and scaled by this parameter
    :return: the covariance matrix

    """
    if rank > sz:
        raise ValueError('rank cannot be greater than size')

    #
    eig_vecs = ortho_group.rvs(dim=sz)
    eig_vecs = eig_vecs[:, 0:rank]
    # randomly sample eigenvalues from chi square (df=1) just so that  smaller eigenvalues are more likely
    eig_vals = np.random.chisquare(1, rank) * eig_val_scale
    # sort in descending order
    eig_vals = np.sort(eig_vals)
    eig_vals = eig_vals[::-1]
    cov = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
    return cov, eig_vecs, eig_vals


# generate a singular 20 x 20 covariance matrix
n_variables = 20
rank = 10
cov_mat,e_vecs,_ = generate_random_cov_matrix(n_variables,rank,100)

# is it singular as expected
assert np.linalg.det(cov_mat)==0

# are the elements of the of the covariance matrix so small that their determinant could be >0 but zero due to machine precision
print(np.mean(cov_mat))


# does numpy compute its inverse without an exception
try:
    cov_inv = np.linalg.inv(cov_mat)
    print('No exception was raised')
except:
    print('An exception was raised')


# is the resulting inverse a proper inverse (is A*inv(A) equal to an identity matrix)

try:
    assert np.all(np.isclose(cov_mat @ cov_inv,np.identity(n_variables)))
except AssertionError:
    print('The computed inverse is not a proper inverse')


