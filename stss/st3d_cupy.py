"""3D structure tensor module using CuPy."""

import logging
from typing import Literal

import cupy as cp
import cupy.typing as cpt
import cupyx as cpx
from cupyx.scipy import ndimage

from stss import util_cupy as util


def structure_tensor_3d(
    volume: cpt.ArrayLike,
    sigma: float,
    ring_filter:bool=True,
    rho:float=None,
    out: cpt.NDArray | None = None,
    truncate: float = 4.0,
) -> cpt.NDArray:
    """Calculate the structure tensor for 3D image data.

    Args:
        volume: A 3D array. Pass `cupy.ndarray` to avoid copying volume.
        sigma: A noise scale, structures smaller than sigma will be removed by smoothing.
        rho: An integration scale giving the size over the neighborhood in which the orientation is to be analysed.
        out: An array with the shape `(6, ...)` in which to place the output. If `None`, a new array is created.
        truncate: Truncate the filter at this many standard deviations.
    Returns:
        S: An array with shape `(6, ...)` containing elements of structure tensor `(s_xx, s_yy, s_zz, s_xy, s_xz, s_yz)`.

    Authors: vand@dtu.dk, 2019; niejep@dtu.dk, 2019-2024, papi@dtu.dk, 2022-2024
    """

    # Make sure it's an array.
    volume = cp.asarray(volume)

    # Check data type. Must be floating point.
    if not cp.issubdtype(volume.dtype, cp.floating):
        logging.warning(
            "volume is not floating type array. This may result in a loss of precision and unexpected behavior."
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

    # Computing derivatives (scipy implementation truncates filter at 4 sigma).
    Vx = ndimage.gaussian_filter(
        volume, sigma, order=(1, 0, 0), mode="nearest", truncate=truncate
        )  # type: ignore
    Vy = ndimage.gaussian_filter(
        volume, sigma, order=(0, 1, 0), mode="nearest", truncate=truncate
        )  # type: ignore
    Vz = ndimage.gaussian_filter(
        volume, sigma, order=(0, 0, 1), mode="nearest", truncate=truncate
        )  # type: ignore

    if out is None:
        # Allocate S.
        S = cp.empty((6,) + volume.shape, dtype=volume.dtype)
    else:
        # S is already allocated. We assume the size is correct.
        S = out

    tmp = cp.empty(volume.shape, dtype=volume.dtype)

    if ring_filter:
        sigma_r = 0.9506 * (sigma)
        # Integrate elements of structure tensor with the ring filter.
        cp.multiply(Vx, Vx, out=tmp)
        S[0] = util.ring_convolve(tmp, sigma_r, mode="nearest", truncate=truncate)
        cp.multiply(Vy, Vy, out=tmp)
        S[1] = util.ring_convolve(tmp, sigma_r, mode="nearest", truncate=truncate)
        cp.multiply(Vz, Vz, out=tmp)
        S[2] = util.ring_convolve(tmp, sigma_r, mode="nearest", truncate=truncate)
        cp.multiply(Vx, Vy, out=tmp)
        S[3] = util.ring_convolve(tmp, sigma_r, mode="nearest", truncate=truncate)
        cp.multiply(Vx, Vz, out=tmp)
        S[4] = util.ring_convolve(tmp, sigma_r, mode="nearest", truncate=truncate)
        cp.multiply(Vy, Vz, out=tmp)
        S[5] = util.ring_convolve(tmp, sigma_r, mode="nearest", truncate=truncate)

    else:
        # Integrating elements of structure tensor (scipy uses sequence of 1D).
        cp.multiply(Vx, Vx, out=tmp)
        ndimage.gaussian_filter(
            tmp, rho, mode="nearest", output=S[0], truncate=truncate
        )
        cp.multiply(Vy, Vy, out=tmp)
        ndimage.gaussian_filter(
            tmp, rho, mode="nearest", output=S[1], truncate=truncate
        )
        cp.multiply(Vz, Vz, out=tmp)
        ndimage.gaussian_filter(
            tmp, rho, mode="nearest", output=S[2], truncate=truncate
        )
        cp.multiply(Vx, Vy, out=tmp)
        ndimage.gaussian_filter(
            tmp, rho, mode="nearest", output=S[3], truncate=truncate
        )
        cp.multiply(Vx, Vz, out=tmp)
        ndimage.gaussian_filter(
            tmp, rho, mode="nearest", output=S[4], truncate=truncate
        )
        cp.multiply(Vy, Vz, out=tmp)
        ndimage.gaussian_filter(
            tmp, rho, mode="nearest", output=S[5], truncate=truncate
        )

    return S


def eig_special_3d(
    S: cpt.ArrayLike,
    full: bool = False
) -> tuple[cpt.NDArray, cpt.NDArray]:
    """Eigensolution for symmetric real 3-by-3 matrices.

    Args:
        S: A floating point array with shape (6, ...) containing structure tensor. Use `float64` to avoid numerical errors. When using lower precision, ensure that the values of S are not very small/large. Pass `cupy.ndarray` to avoid copying S.
        full: A flag indicating that all three eigenvectors should be returned.
        
    Returns:
        val: An array with shape `(3, ...)` containing eigenvalues.
        vec: An array with shape `(3, ...)` containing the vector corresponding to the smallest eigenvalue if `full` is `False`, otherwise `(3, 3, ...)` containing eigenvectors.

    More:
        An analytic solution of eigenvalue problem for real symmetric matrix,
        using an affine transformation and a trigonometric solution of third
        order polynomial. See https://en.wikipedia.org/wiki/Eigenvalue_algorithm
        which refers to Smith's algorithm https://dl.acm.org/citation.cfm?id=366316.

    Authors: vand@dtu.dk, 2019; niejep@dtu.dk, 2019-2024
    """
    S = cp.asarray(S)

    # Check data type. Must be floating point.
    if not cp.issubdtype(S.dtype, cp.floating):
        raise ValueError("S must be floating point type.")

    # Flatten S.
    input_shape = S.shape
    S = S.reshape(6, -1)

    # Create v vector.
    v = cp.array([[2 * cp.pi / 3], [4 * cp.pi / 3]], dtype=S.dtype)

    # Computing eigenvalues.

    # Allocate vec and val. We will use them for intermediate computations as well.
    if full:
        val = cp.empty((3,) + S.shape[1:], dtype=S.dtype)
        vec = cp.empty((9,) + S.shape[1:], dtype=S.dtype)
        tmp = cp.empty((4,) + S.shape[1:], dtype=S.dtype)
        B03 = val
        B36 = vec[:3]
    else:
        val = cp.empty((3,) + S.shape[1:], dtype=S.dtype)
        vec = cp.empty((3,) + S.shape[1:], dtype=S.dtype)
        tmp = cp.empty((4,) + S.shape[1:], dtype=S.dtype)
        B03 = val
        B36 = vec

    # Views for B.
    B0 = B03[0]
    B1 = B03[1]
    B2 = B03[2]
    B3 = B36[0]
    B4 = B36[1]
    B5 = B36[2]

    # Compute q, mean of diagonal. We need to use q multiple times later.
    # Using cp.mean has precision issues.
    q = cp.add(S[0], S[1], out=tmp[0])
    q += S[2]
    q /= 3

    # Compute S minus q. Insert it directly into B where it'll stay.
    Sq = cp.subtract(S[:3], q, out=B03)

    # Compute s, off-diagonal elements. Store in part of B not yet used.
    s = cp.sum(cp.multiply(S[3:], S[3:], out=B36), axis=0, out=tmp[1])
    s *= 2

    # Compute p.
    p = cp.sum(cp.multiply(Sq, Sq, out=B36), axis=0, out=tmp[2])
    del Sq  # Last use of Sq.
    p += s

    p *= 1 / 6
    cp.sqrt(p, out=p)

    # Compute inverse p, while avoiding 0 division.
    # Reuse s allocation and delete s variable.
    p_inv = s
    del s
    non_zero_p_mask = p == 0
    cp.divide(1, p, out=p_inv)
    p_inv[non_zero_p_mask] = 0

    # Compute B. First part is already filled.
    B03 *= p_inv
    cp.multiply(S[3:], p_inv, out=B36)

    # Compute d, determinant of B.
    d = cp.prod(B03, axis=0, out=tmp[3])

    # Reuse allocation for p_inv and delete variable.
    d_tmp = p_inv
    del p_inv
    # Computation of d.
    cp.multiply(B2, B3, d_tmp)
    d_tmp *= B3
    d -= d_tmp
    cp.multiply(B4, B4, out=d_tmp)
    d_tmp *= B1
    d -= d_tmp
    cp.prod(B36, axis=0, out=d_tmp)
    d_tmp *= 2
    d += d_tmp
    cp.multiply(B5, B5, out=d_tmp)
    d_tmp *= B0
    d -= d_tmp
    d *= 0.5
    # Ensure -1 <= d/2 <= 1.
    cp.clip(d, -1, 1, out=d)

    # Compute phi. Beware that we reuse d variable!
    phi = d
    del d
    phi = cp.arccos(phi, out=phi)
    phi /= 3

    # Compute val, ordered eigenvalues. Resuing B allocation.
    del B03, B36, B0, B1, B2, B3, B4, B5

    cp.add(v, phi[cp.newaxis], out=val[:2])
    val[2] = phi
    cp.cos(val, out=val)
    p *= 2
    val *= p
    val += q

    # Remove all variable using tmp allocation.
    del q
    del p
    del phi
    del d_tmp

    # Computing eigenvectors -- either only one or all three.
    if full:
        l = val
        vec = vec.reshape(3, 3, -1)
        vec_tmp = tmp[:3]
    else:
        l = val[0]
        vec_tmp = tmp[2]

    # Compute vec. The tmp variable can be reused.

    # u = S[4] * S[5] - (S[2] - l) * S[3]
    u = cp.subtract(S[2], l, out=vec[0])
    cp.multiply(u, S[3], out=u)
    u_tmp = cp.multiply(S[4], S[5], out=tmp[3])
    cp.subtract(u_tmp, u, out=u)
    # Put values of u into vector 2 aswell.

    # v = S[3] * S[5] - (S[1] - l) * S[4]
    v = cp.subtract(S[1], l, out=vec_tmp)
    cp.multiply(v, S[4], out=v)
    v_tmp = cp.multiply(S[3], S[5], out=tmp[3])
    cp.subtract(v_tmp, v, out=v)

    # w = S[3] * S[4] - (S[0] - l) * S[5]
    w = cp.subtract(S[0], l, out=vec[2])
    cp.multiply(w, S[5], out=w)
    w_tmp = cp.multiply(S[3], S[4], out=tmp[3])
    cp.subtract(w_tmp, w, out=w)

    vec[1] = u
    cp.multiply(u, v, out=vec[0])
    u = vec[1]
    cp.multiply(u, w, out=vec[1])
    cp.multiply(v, w, out=vec[2])

    # Remove u, v, w and l variables.
    del u
    del v
    del w
    del l

    # Normalizing -- depends on number of vectors.
    if full:
        # vec is [x1 x2 x3, y1 y2 y3, z1 z2 z3]
        l = cp.sum(cp.square(vec), axis=0, out=vec_tmp)[:, cp.newaxis]
        vec = cp.swapaxes(vec, 0, 1)
    else:
        # vec is [x1 y1 z1] = v1
        l = cp.sum(cp.square(vec, out=tmp[:3]), axis=0, out=vec_tmp)

    cpx.rsqrt(l, out=l)
    vec *= l

    val = val.reshape(val.shape[:-1] + input_shape[1:])
    vec = vec.reshape(vec.shape[:-1] + input_shape[1:])

    return val, vec