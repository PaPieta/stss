"""2D structure tensor module."""
import logging

import numpy as np
from scipy import ndimage

from st2ss import util


def structure_tensor_2d(image, sigma, ring_filter=True, rho=None, out=None, truncate=4.0):
    """Structure tensor for 2D image data.

    Arguments:
        image: array_like
            A 2D array. Pass ndarray to avoid copying image.
        sigma: scalar
            Derivative Gaussian filter size, correlated to feature size if ring_filter=True.
        ring_filter: bool
            If True, runs the algorithm version with ring filter instead of the integration filter
        rho: scalar
            Only if ring_filter=False. An integration scale giving the size over the neighborhood in which the
            orientation is to be analysed.
        out: ndarray, optinal
            A Numpy array with the shape (3, volume.shape) in which to place the output.
        truncate: float
            Truncate the filter at this many standard deviations. Default is 4.0.

    Returns:
        S: ndarray
            An array with shape (3, image.shape) containing elements of structure tensor
            (s_xx, s_yy, s_xy).

    Authors:
        vand@dtu.dk, 2019; niejep@dtu.dk, 2020, papi@dtu.dk, 2023
    """

    # Make sure it's a Numpy array.
    image = np.asarray(image)

    # Check data type. Must be floating point.
    if not np.issubdtype(image.dtype, np.floating):
        logging.warning('image is not floating type array. This may result in a loss of precision and unexpected behavior.') 

    # Check user input (ring filter vs integration filter).
    if ring_filter==True and rho is not None:
        logging.warning('rho is set with active ring filter. Rho value will have no effect.')
    elif ring_filter==False and rho is None:
        logging.warning('rho is not set while ring filter is disabled. Rho value will be set to 2*sigma.')
        rho=2*sigma
    
    # Compute derivatives (Scipy implementation truncates filter at 4 sigma).
    Ix = ndimage.gaussian_filter(image, sigma, order=[1, 0], mode='nearest', truncate=truncate)
    Iy = ndimage.gaussian_filter(image, sigma, order=[0, 1], mode='nearest', truncate=truncate)

    if out is None:
        # Allocate S.
        S = np.empty((3, ) + image.shape, dtype=image.dtype)
    else:
        # S is already allocated. We assume the size is correct.
        S = out
    
    tmp = np.empty(image.shape, dtype=image.dtype)

    if ring_filter:
        sigma_r = 0.9506 * (sigma)
        # Integrate elements of structure tensor with the ring filter.
        np.multiply(Ix, Ix, out=tmp)
        S[0] = util.ring_convolve(tmp, sigma_r, mode='nearest', truncate=truncate)
        np.multiply(Iy, Iy, out=tmp)
        S[1] = util.ring_convolve(tmp, sigma_r, mode='nearest', truncate=truncate)
        np.multiply(Ix, Iy, out=tmp)
        S[2] = util.ring_convolve(tmp, sigma_r, mode='nearest', truncate=truncate)
 
    else:
        # Integrate elements of structure tensor (Scipy uses sequence of 1D).
        np.multiply(Ix, Ix, out=tmp)
        ndimage.gaussian_filter(tmp, rho, mode='nearest', output=S[0], truncate=truncate)
        np.multiply(Iy, Iy, out=tmp)
        ndimage.gaussian_filter(tmp, rho, mode='nearest', output=S[1], truncate=truncate)
        np.multiply(Ix, Iy, out=tmp)
        ndimage.gaussian_filter(tmp, rho, mode='nearest', output=S[2], truncate=truncate)

    # Normalize S scale response
    S = S*(sigma**(2*1.2))

    return S


def eig_special_2d(S):
    """Eigensolution for symmetric real 2-by-2 matrices.

    Arguments:
        S: ndarray
            A floating point array with shape (3, ...) containing structure tensor.

    Returns:
        val: ndarray
            An array with shape (2, ...) containing sorted eigenvalues.
        vec: ndarray
            An array with shape (2, ...) containing eigenvector corresponding
            to the smallest eigenvalue (the other is orthogonal to the first).

    Authors:
        vand@dtu.dk, 2019; niejep@dtu.dk, 2020
    """

    # Save original shape and flatten.
    input_shape = S.shape
    S = S.reshape(3, -1)

    # Calculate val.
    val = np.empty((2, S.shape[1]), dtype=S.dtype)
    np.subtract(S[0], S[1], out=val[1])
    val[1] *= val[1]
    np.multiply(S[2], S[2], out=val[0])
    val[0] *= 4
    val[1] += val[0]
    np.sqrt(val[1], out=val[1])
    np.negative(val[1], out=val[0])
    val += S[0]
    val += S[1]
    val *= 0.5

    # Calcualte vec, y will be positive.
    vec = np.empty((2, S.shape[1]), dtype=S.dtype)
    np.negative(S[2], out=vec[0])
    np.subtract(S[0], val[0], out=vec[1])

    # Deal with diagonal matrices.
    aligned = S[2] == 0

    # Sort.
    vec[:, aligned] = 1 - np.argsort(S[:2, aligned], axis=0)

    # Normalize.
    vec_norm = np.einsum('ij,ij->j', vec, vec)
    np.sqrt(vec_norm, out=vec_norm)
    vec /= vec_norm

    # Reshape and return.
    val = val.reshape(val.shape[:1] + input_shape[1:])
    vec = vec.reshape(vec.shape[:1] + input_shape[1:])
    return val, vec
