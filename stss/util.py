from typing import Union

import numpy as np
import numpy.typing as npt

from scipy.ndimage import convolve1d

# Constant values for scale correction
C_SIGMA_XF = 0.372

C_2D_ANIS = 0.0675
C_2D_ISO = -1 / 3

C_3D = 0.529
C_3D_LIN = 0.327
C_3D_PLAN = 1
C_3D_SPH = 0.0158


def gauss_no_norm(t: float, truncate: float = 4.0) -> npt.NDArray[np.floating]:
    """Returns a 1D Gaussian function without the normalizing constant.

    Arguments:
        t: float
            Gaussian variance.
        truncate: float
            Truncate the filter at this many standard deviations. Default is 4.0.

    Returns:
        g: npt.NDArray
            A 1D array containing values of the unnormalized Gaussian.

    Authors:
        papi@dtu.dk, 2023, based on the code by abda@dtu.dk
    """

    s = np.sqrt(t)
    x = np.arange(int(-np.round(s * truncate)), int(np.round(s * truncate)) + 1)
    g = np.exp(-(x**2) / (2 * t))
    return g


def ring_convolve(image: npt.NDArray[np.floating], 
                  sigma_r: float, 
                  truncate : float = 4.0, 
                  mode: str = "nearest", 
                  cval: float = 0.0, 
                  origin: int = 0
) -> npt.NDArray[np.floating]:
    """Convolves an image with a ring filter.

    Arguments:
        image: npt.NDArray
            A 2D or 3D array containing the image.
        sigma_r: float
            Ring filter size based on Gaussian variance.
        truncate: float
            Truncate the filter at this many standard deviations. Default is 4.0.
        mode, cval, origin:
            see scipy.ndimage.convolve1d

    Returns:
        image: npt.NDArray
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


def correct_scale(scale: Union[float, npt.NDArray[np.floating]], val: npt.NDArray[np.floating]) -> Union[float, npt.NDArray[np.floating]]:
    """Corrects the scale values assigned to each pixel, to better reflect feature sizes.

    Arguments:
        scale: float|npt.NDArray
            Scale of the structure tensor features.
        val: npt.NDArray
            Eigenvalues of the structure tensor.

    Returns:
        scale: float|npt.NDArray
            Corrected scale of the structure tensor features.

    Authors:
        papi@dtu.dk, 2023
    """

    if scale.ndim == 2:
        iso = val[0] / val[1]
        scale = scale / ((1 + C_2D_ANIS * (1 - iso)) * (1 + C_2D_ISO * iso))
        scale = scale / C_SIGMA_XF
    elif scale.ndim == 3:
        lin = (val[1] - val[0]) / val[2]
        plan = (val[2] - val[1]) / val[2]
        sph = val[0] / val[2]

        scale = scale / (
            C_3D * (1 + C_3D_SPH * sph) * (1 + C_3D_PLAN * plan) * (1 + C_3D_LIN * lin)
        )
        scale = scale / C_SIGMA_XF
    else:
        raise ValueError("Scale must be 2D or 3D.")

    return scale
