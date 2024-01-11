import numpy as np
from scipy.ndimage import convolve1d

# Constant values for scale correction
C = 1.07
C_LIN = 0.65
C_PLAN = 1
C_SPH = 0.5


def gauss_no_norm(t, truncate=4.0):
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
        papi@dtu.dk, 2023, based on the code by abda@dtu.dk
    """

    s = np.sqrt(t)
    x = np.arange(int(-np.round(s * truncate)), int(np.round(s * truncate)) + 1)
    g = np.exp(-(x**2) / (2 * t))
    return g


def ring_convolve(image, sigma_r, truncate=4.0, mode="nearest", cval=0.0, origin=0):
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
        papi@dtu.dk, 2023
    """

    output = np.copy(image)
    temp = np.copy(output)

    # Prepeare components for the ring filter.
    g1 = gauss_no_norm(sigma_r**2, truncate=truncate)
    g2 = gauss_no_norm((sigma_r * 0.999) ** 2, truncate=truncate / 0.999)

    # Normalize the ring filter components.
    norm = np.sum(g1 - g2)
    g1 = g1 / norm
    g2 = g2 / norm

    for i in range(image.ndim):
        # Integrate elements of structure tensor with the ring filter.
        convolve1d(
            output, g1, axis=i, output=output, mode=mode, cval=cval, origin=origin
        )
        convolve1d(temp, g2, axis=i, output=temp, mode=mode, cval=cval, origin=origin)

    output = output - temp
    return output


def correct_scale(scale, val):
    """Corrects the scale values assigned to each pixel, to better reflect feature sizes.

    Arguments:
        scale: float
            Scale of the structure tensor features.
        val: ndarray
            Eigenvalues of the structure tensor.

    Returns:
        scale:
            Corrected scale of the structure tensor features.

    Authors:
        papi@dtu.dk, 2023
    """

    if scale.ndim == 2:
        scale = scale / (1.067 * (1 - 0.43 * val[0] / val[1])) / 0.372
    elif scale.ndim == 3:
        lin = (val[1] - val[0]) / val[2]
        plan = (val[2] - val[1]) / val[2]
        sph = val[0] / val[2]
        scale = (scale / (C * (C_LIN * lin + C_PLAN * plan + C_SPH * sph))) / 0.372
    else:
        raise ValueError("Scale must be 2D or 3D.")

    return scale