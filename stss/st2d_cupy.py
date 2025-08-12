"""2D structure tensor module using CuPy."""
from typing import Union, Tuple

import cupy as cp
import cupy.typing as cpt
import cupyx as cpx
from cupyx.scipy import ndimage

from stss import util_cupy as util

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format="%(asctime)s : %(levelname)s : %(module)s : %(message)s",
    datefmt="%I:%M:%S",
)
logger = logging.getLogger(__name__)


def structure_tensor_2d(
    image: cpt.ArrayLike,
    sigma: float,
    ring_filter: bool = True,
    rho: Union[None, float] = None,
    out: Union[None,cpt.NDArray[cp.floating]] = None,
    truncate: float = 4.0,
) -> cpt.NDArray:
    """Structure tensor for 2D image data using CuPy.

    Arguments:
        image: cpt.ArrayLike
            A 2D array. Pass ndarray to avoid copying image.
        sigma: float
            Derivative Gaussian filter size, correlated to feature size if ring_filter=True.
        ring_filter: bool
            If True, runs the algorithm version with ring filter instead of the integration filter
        rho: float
            Only if ring_filter=False. An integration scale giving the size over the neighborhood in which the
            orientation is to be analysed.
        out: cpt.ArrayLike, optinal
            A Numpy array with the shape (3, volume.shape) in which to place the output.
        truncate: float
            Truncate the filter at this many standard deviations. Default is 4.0.

    Returns:
        S: cpt.ArrayLike
            An array with shape (3, image.shape) containing elements of structure tensor
            (s_xx, s_yy, s_xy).

    Authors:
        vand@dtu.dk, 2019; niejep@dtu.dk, 2020, papi@dtu.dk, 2023
    """

    # Make sure it's a Cupy array.
    image = cp.asarray(image)

    # Check data type. Must be floating point.
    if not cp.issubdtype(image.dtype, cp.floating):
        logging.warning(
            "image is not floating type array. This may result in a loss of precision and unexpected behavior."
        )

    # Check user input (ring filter vs integration filter).
    if ring_filter is True and rho is not None:
        logging.warning(
            "rho is set with active ring filter. Rho value will have no effect."
        )
    elif ring_filter is False and rho is None:
        logging.warning(
            "rho is not set while ring filter is disabled. Rho value will be set to 2*sigma."
        )
        rho = 2 * sigma

    # Compute derivatives (Scipy implementation truncates filter at 4 sigma).
    Ix = ndimage.gaussian_filter(
        image, sigma, order=[1, 0], mode="nearest", truncate=truncate
    )
    Iy = ndimage.gaussian_filter(
        image, sigma, order=[0, 1], mode="nearest", truncate=truncate
    )

    if out is None:
        # Allocate S.
        S = cp.empty((3,) + image.shape, dtype=image.dtype)
    else:
        # S is already allocated. We assume the size is correct.
        S = out

    tmp = cp.empty(image.shape, dtype=image.dtype)

    if ring_filter:
        sigma_r = 0.9506 * (sigma)
        # Integrate elements of structure tensor with the ring filter.
        cp.multiply(Ix, Ix, out=tmp)
        S[0] = util.ring_convolve(tmp, sigma_r, mode="nearest", truncate=truncate)
        cp.multiply(Iy, Iy, out=tmp)
        S[1] = util.ring_convolve(tmp, sigma_r, mode="nearest", truncate=truncate)
        cp.multiply(Ix, Iy, out=tmp)
        S[2] = util.ring_convolve(tmp, sigma_r, mode="nearest", truncate=truncate)

    else:
        # Integrate elements of structure tensor (Scipy uses sequence of 1D).
        cp.multiply(Ix, Ix, out=tmp)
        ndimage.gaussian_filter(
            tmp, rho, mode="nearest", output=S[0], truncate=truncate
        )
        cp.multiply(Iy, Iy, out=tmp)
        ndimage.gaussian_filter(
            tmp, rho, mode="nearest", output=S[1], truncate=truncate
        )
        cp.multiply(Ix, Iy, out=tmp)
        ndimage.gaussian_filter(
            tmp, rho, mode="nearest", output=S[2], truncate=truncate
        )

    return S


def eig_special_2d(S: cpt.ArrayLike) -> Tuple[cpt.NDArray[cp.floating], cpt.NDArray[cp.floating]]:
    """Eigensolution for symmetric real 2-by-2 matrices.

    Arguments:
        S: cpt.ArrayLike
            A floating point array with shape (3, ...) containing structure tensor.

    Returns:
        val: cpt.NDArray
            An array with shape (2, ...) containing sorted eigenvalues.
        vec: cpt.NDArray
            An array with shape (2, ...) containing eigenvector corresponding
            to the smallest eigenvalue (the other is orthogonal to the first).

    Authors:
        vand@dtu.dk, 2019; niejep@dtu.dk, 2020
    """

    # Save original shape and flatten.
    input_shape = S.shape
    S = S.reshape(3, -1)

    # Calculate val.
    val = cp.empty((2, S.shape[1]), dtype=S.dtype)
    cp.subtract(S[0], S[1], out=val[1])
    val[1] *= val[1]
    cp.multiply(S[2], S[2], out=val[0])
    val[0] *= 4
    val[1] += val[0]
    cp.sqrt(val[1], out=val[1])
    cp.negative(val[1], out=val[0])
    val += S[0]
    val += S[1]
    val *= 0.5

    # Calcualte vec, y will be positive.
    vec = cp.empty((2, S.shape[1]), dtype=S.dtype)
    cp.negative(S[2], out=vec[0])
    cp.subtract(S[0], val[0], out=vec[1])

    # Deal with diagonal matrices.
    aligned = S[2] == 0

    # Sort.
    vec[:, aligned] = 1 - cp.argsort(S[:2, aligned], axis=0)

    # Normalize.
    vec_norm = cp.einsum("ij,ij->j", vec, vec)
    cp.sqrt(vec_norm, out=vec_norm)
    vec /= vec_norm

    # Reshape and return.
    val = val.reshape(val.shape[:1] + input_shape[1:])
    vec = vec.reshape(vec.shape[:1] + input_shape[1:])
    return val, vec
