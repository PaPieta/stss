import cupy as cp
import cupy.typing as cpt
import cupyx as cpx
from cupyx.scipy.ndimage import convolve1d


def gauss_no_norm(t:float, truncate:float=4.0) -> cpt.NDArray:
    """Returns a 1D Gaussian function without the normalizing constant.

    Arguments:
        t: float
            Gaussian variance.
        truncate: float
            Truncate the filter at this many standard deviations. Default is 4.0.

    Returns:
        g: ndarray
            A 1D array containing values of the unnormalized Gaussian.

    Authors:
        papi@dtu.dk, 2023-2024, based on the code by abda@dtu.dk
    """

    s = cp.sqrt(t)
    x = cp.arange(int(-cp.round(s * truncate)), int(cp.round(s * truncate)) + 1)
    g = cp.exp(-(x**2) / (2 * t))
    return g


def ring_convolve(image:cpt.ArrayLike, sigma_r: float, truncate:float=4.0, mode:str="nearest", cval:float=0.0, origin:int=0) -> cpt.NDArray:
    """Convolves an image with a ring filter.

    Arguments:
        image: ndarray
            A 2D or 3D array containing the image.
        sigma_r: float
            Ring filter size based on Gaussian variance.
        truncate: float
            Truncate the filter at this many standard deviations. Default is 4.0.
        mode, cval, origin:
            see scipy.ndimage.convolve1d

    Returns:
        image: ndarray
            A 2D array containing the convolved image.

    Authors:
        papi@dtu.dk, 2023-2024
    """

    output = cp.copy(image)
    temp = cp.copy(output)

    # Prepeare components for the ring filter.
    g1 = gauss_no_norm(sigma_r**2, truncate=truncate)
    g2 = gauss_no_norm((sigma_r * 0.999) ** 2, truncate=truncate / 0.999)

    # Normalize the ring filter components.
    norm = cp.sum(g1 - g2)
    g1 = g1 / norm
    g2 = g2 / norm

    for i in range(image.ndim):
        # Integrate elements of structure tensor with the ring filter.
        output = convolve1d(
            output, g1, axis=i, mode=mode, cval=cval, origin=origin
        )
        temp = convolve1d(temp, g2, axis=i, mode=mode, cval=cval, origin=origin)

    output = output - temp
    return output
