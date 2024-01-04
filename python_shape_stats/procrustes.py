import numpy as np
import copy
import joblib
from joblib_progress import joblib_progress
from python_shape_stats.helpers import validate_landmark_configuration_and_weights


def compute_procrustes_transform(p: np.ndarray, q: np.ndarray, scale: bool = True, w: np.ndarray = None) -> np.ndarray:
    """
    Computes the optimal rotation, translation and (optionally) scaling to align
    two corresponding landmark configurations (aligns p to q).
    
    :param p: an n (landmarks) by 3 array of landmark coordinates
    :param q: an n (landmarks) by 3 array of landmark coordinates
    :param scale: if True the scaling of p to q will be
            calculated. Otherwise, only rotation and translation will be calculated
    :param w: if not None this should be a
            vector of length n (landmarks). This is a vector of weights, those
            landmarks with higher weights have more impact on the solution
    :return: a 4x4 transformation matrix
    """
    w = validate_landmark_configuration_and_weights(p, q=q, w=w)
    r = compute_rotation(p, q, w)  # compute rotation

    # compute scaling
    if scale:
        p_cs = compute_centroid_size(p, w)
        q_cs = compute_centroid_size(q, w)
        s = q_cs / p_cs
        r[0:3, 0:3] *= s  # add isotropic scaling to  the rotation matrix

    # compute translation  of p to origin
    center_p = compute_centroid(p, w)
    Ta = make_translation_matrix(origin=center_p, destination=np.array([0, 0, 0]))

    # compute translation from origin to center of q
    center_q = compute_centroid(q, w)
    Tb = make_translation_matrix(origin=np.array([0, 0, 0]), destination=center_q)

    # combine transformations
    t = Ta @ r @ Tb
    return t


def compute_rotation(p: np.ndarray, q: np.ndarray, w: np.ndarray = None) -> np.ndarray:
    """
    Calculates the optimal rotation from p to q

    :param p: an n (landmarks) by 3 array of landmark coordinates
    :param q: an n (landmarks) by 3 array of landmark coordinates
    :param w: if not None this should be a vector of length n (landmarks). This is a vector of weights, those landmarks with higher weights have more impact on the solution
    :return: a 4x4 transformation matrix, modelling only rotation
    """
    w = validate_landmark_configuration_and_weights(p, q=q, w=w)
    center_p = compute_centroid(p, w=w)
    center_q = compute_centroid(q, w=w)

    # center p and q onto their centroids
    p0 = p - center_p
    q0 = q - center_q

    h = (p0 * w).T @ q0  # get the covariance matrix
    # get the rotation matrix
    u, s, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    # check for reflection and correct if present
    if np.linalg.det(r) < 0:
        vt[:, 2] *= -1
        r = vt.T @ u.T

    t = np.identity(4)
    t[0:3, 0:3] = r.T
    return t


def make_translation_matrix(origin=np.array([0, 0, 0]), destination=np.array([0, 0, 0])) -> np.ndarray:
    """
    Constructs a translation matrix for translating from the origin to the destination

    :param origin: a vector of length 3 specifying the origin of the translation, defaults to [0,0,0]
    :param destination: a vector of length 3 specifying the destination of the translation; defaults to [0,0,0]
    :return: a 4x4 transformation matrix modelling only the translation from origin to destination
    """
    v = destination - origin
    t = np.identity(4)
    t[3, 0:3] = v
    return t


def compute_centroid_size(p: np.ndarray, w: np.ndarray = None) -> float:
    """
    Computes the (weighted) centroid size of the landmark configuration p

    :param p: an n (landmarks) by 3 array of landmark coordinates
    :param w: if not None this should be a vector of length n (landmarks). This is a vector of weights, those landmarks with higher weights have more impact on the solution
    :return: the centroid size of the landmark configuration p
    """
    w = validate_landmark_configuration_and_weights(p, w=w)
    p0 = p - compute_centroid(p, w=w)  # center p on centroid
    weighted_ss = np.sum((np.linalg.norm(p0, axis=1) ** 2) * w)
    return np.sqrt(weighted_ss)


def compute_rms_size(p: np.ndarray, w: np.ndarray = None) -> float:
    """
    Computes the (weighted) root mean squared distance of the landmarks in configuration p from the centroid

    :param p: an n (landmarks) by 3 array of landmark coordinates
    :param w: if not None this should be a vector of length n (landmarks). This is a vector of weights, those landmarks with higher weights have more impact on the solution
    :return: the rms size of the landmark configuration p
    """
    w = validate_landmark_configuration_and_weights(p, w=w)
    p0 = p - compute_centroid(p, w=w)  # center p on centroid
    weighted_ss = np.sum((np.linalg.norm(p0, axis=1) ** 2) * w)
    return np.sqrt(weighted_ss / np.sum(w))


def compute_centroid(p: np.ndarray, w: np.ndarray = None) -> np.ndarray:
    """
    Computes the (weighted) centroid of the landmark configuration p

   :param p: an n (landmarks) by 3 array of landmark coordinates.
   :param w: if not None this should be a vector of length n (landmarks). This is a vector of weights, those landmarks with higher weights have more impact on the solution.
   :return: the centroid of the landmark configuration p.
   """
    w = validate_landmark_configuration_and_weights(p, w=w)
    return np.sum(p * w, axis=0) / np.sum(w)  # weighted column mean


def scale_shape(p: np.ndarray, target_size: float = 1., size_type='centroid', w: np.ndarray = None) -> np.ndarray:
    """
    Scales the landmark configuration to the target centroid size.

    :param p: an n (landmarks) by 3 array of landmark coordinates.
    :param target_size: the target centroid size to scale p to
    :param w: if not None this should be a vector of length n (landmarks). This is a vector of weights, thoselandmarks with higher weights have more impact on the solution.
    :param target_size: the size to scale p to
    :param size_type: which definition of size to use, can be 'centroid' or 'rms'
    :param w: the landmark configuration p scaled to the target size
    """
    w = validate_landmark_configuration_and_weights(p, w=w)
    center_p = compute_centroid(p, w)
    if size_type == 'centroid':
        p_size = compute_centroid_size(p, w)
    elif size_type == 'rms':
        p_size = compute_rms_size(p, w)
    scaling = target_size / p_size
    return (p - center_p) * scaling + center_p


def apply_procrustes_transform(p: np.ndarray, t: np.ndarray, invert: bool = False) -> np.ndarray:
    """
    Applies a procrustes transformation to the landmark configuration p

    :param p: an n (landmarks) by 3 array of landmark coordinates.
    :param t: a 4x4 transformation matrix
    :param invert: if True the function will apply the inverse of the transformation in t
    :return: the landmark configuration p after applying the transformation
    """

    p_hom = np.concatenate([p, np.ones([p.shape[0], 1])], axis=1)  # transform p into homogeneous coordinates
    # apply transformation to get p'
    if invert:
        t = np.linalg.inv(t)
    p_prime = p_hom @ t
    return p_prime[:, 0:3]

def conform_p_to_q(p : np.ndarray,q : np.ndarray,w=None,scale=True) -> tuple:
    """
    Computes the optimal rotation, translation and (optionally) scaling to align
    two corresponding landmark configurations (aligns p to q) and applies the transformation to p.

    :param p: an n (landmarks) by 3 array of landmark coordinates
    :param q: an n (landmarks) by 3 array of landmark coordinates
    :keyword scale: if True the scaling of p to q will be
            calculated. Otherwise, only rotation and translation will be calculated
    :keyword w: if not None this should be a
            vector of length n (landmarks). This is a vector of weights, those
            landmarks with higher weights have more impact on the solution
    :return: a tuple containing a copy of p and the 4x4 transformation matrix
    """
    w = validate_landmark_configuration_and_weights(p,q=q,w=w)
    t = compute_procrustes_transform(p,q,w=w,scale=scale)
    return apply_procrustes_transform(p,t),t

def do_generalized_procrustes_analysis(landmarks: np.ndarray, scale: bool = True, max_iter: int = np.inf,
                                       init_landmarks: np.ndarray = None,n_jobs=1) -> dict:
    """
    Aligns a sample of landmark configurations to their mean by generalized Procrustes analysis

    :param landmarks: an n (vertices) x 3 (dimensions) x k (observations) matrix of landmark coordinates
    :param scale: if True all landmark configurations will be scaled to unit centroid size
    :param max_iter: the maximum number of iterations if not specified the algorithm will continue until convergence, however long it takes
    :param init_landmarks: an n (vertices) x 3 (dimensions) array of landmark coordinates to initialise the algorithm. if unspecified
    :param n_jobs: the number of jobs to run in parallel - see joblib.Parallel documentation
    :return: a tuple with two elements: 1.  the landmarks after alignment to the sample mean and 2. the sample mean to which they are aligned
    """

    def assemble_output():
        return aligned_landmarks, prev_avg
    # check size of array
    if landmarks.shape[1] != 3:
        raise ValueError('shapes should be n landmarks x 3 dimensions x k observations')
    n_configs = landmarks.shape[2]

    # initialise algorithm
    if init_landmarks is not None:
        curr_avg = init_landmarks
    else:
        curr_avg = landmarks[:, :, 0]
    if scale:
        curr_avg = scale_shape(curr_avg,target_size=1.)
    iter = 0
    track_convergence = []

    print('Begin generalized Procrustes analysis')
    while True:
        print('Iteration ' + str(iter))
        with joblib_progress('Iteration ' + str(iter),n_configs):
            r = joblib.Parallel(n_jobs=n_jobs)(joblib.delayed(conform_p_to_q)(landmarks[:,:,x],curr_avg,scale=scale) for x in range(n_configs))
        r,_=zip(*r)
        aligned_landmarks = np.stack(r,axis=2)
        prev_avg = copy.copy(curr_avg)
        curr_avg = np.mean(aligned_landmarks, axis=2)

        # assess convergence
        if iter == (max_iter-1):
            print('Maximum number of iterations reached without converging')
            return assemble_output()
        if iter > 0:
            diff = np.sqrt(np.mean(np.square(prev_avg-curr_avg)))
            print('RMS difference ='+str(diff))
            track_convergence.append(diff)
        if iter > 1:
            grad = np.gradient(track_convergence)
            if np.isclose(grad[-1],0.) | (grad[-1]>0):
                print('algorithm converged')
                return assemble_output()
        iter += 1



