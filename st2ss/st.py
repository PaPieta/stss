from st2ss import structure_tensor_2d, structure_tensor_3d, eig_special_2d, eig_special_3d, util

import numpy as np

def structure_tensor(image, sigma, ring_filter=True, rho=None, out_S=None, eig_decomp=True, truncate=4.0):
    """Generalized structure tensor for 2D and 3D data.

    Arguments:
        image: array_like
            A 2D or 3D array containing the image.
        sigma: scalar
            Derivative Gaussian filter size, correlated to feature size if ring_filter=True.
        ring_filter: bool
            If True, runs the algorithm version with ring filter instead of the integration filter
        rho: scalar
            Only if ring_filter=False. An integration scale giving the size over the neighborhood in which the
            orientation is to be analysed.
        out_S: ndarray, optional
            A Numpy array in which to place the output elements of structure tensor S.
        eig_decomp: bool
            If True, the eigenvalues and eigenvectors of the structure tensor are computed.
        truncate: float
            Truncate the filter at this many standard deviations. Default is 4.0.

    Returns:
        S: ndarray
            An array containing elements of structure tensor.
            If image is 2D, with shape (3, volume.shape), containing (s_xx, s_yy, s_xy).
            If image is 3D, with shape (6, volume.shape), containing (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz).
        val, vec: ndarray
            Optional eigenvectors and eigenvalues of the structure tensor if eig_decomp=True.
    Authors: papi@dtu.dk, 2023
    """

    if out_S is None:
        # Allocate S.
        if image.ndim == 2:
            S = np.empty((3, ) + image.shape, dtype=image.dtype)
        elif image.ndim == 3:
            S = np.empty((6, ) + image.shape, dtype=image.dtype)
        else:
            raise ValueError('Image must be 2D or 3D.')
    else:
        # S is already allocated. We assume the size is correct.
        S = out_S


    if image.ndim == 2:
        S = structure_tensor_2d(image, sigma, ring_filter, rho, S, truncate)
        if eig_decomp:
            val,vec =  eig_special_2d(S)
            return S,val,vec
        else:
            return S
    elif image.ndim == 3:
        S = structure_tensor_3d(image, sigma, ring_filter, rho, S, truncate)
        if eig_decomp:
            val,vec =  eig_special_3d(S)
            return S,val,vec
        else:
            return S
    else:
        raise ValueError('Image must be 2D or 3D.')
    

def scale_space(image, sigma_list, correctScale=True, ring_filter=True, rho_list=None, truncate=4.0):
    """Structure tensor scale space for 2D and 3D data. Returns a single structure tensor result for each pixel,
    chosen based on the scale that returns the highest trace of the structure tensor matrix.

    Arguments:
        image: array_like
            A 2D or 3D array containing the image.
        sigma: scalar
            Derivative Gaussian filter size, correlated to feature size if ring_filter=True.
        correctScale: bool
            If True, the scale values are corrected to better reflect the feature size. 
            Set to False for easier scale range selection.
        ring_filter: bool
            If True, runs the algorithm version with ring filter instead of the integration filter
        rho: scalar
            Only if ring_filter=False. An integration scale giving the size over the neighborhood in which the
            orientation is to be analysed.
        eig_decomp: bool
            If True, the eigenvalues and eigenvectors of the structure tensor are computed.
        truncate: float
            Truncate the filter at this many standard deviations. Default is 4.0.

    Returns:
        S: ndarray
            An array containing elements of structure tensor.
            If image is 2D, with shape (3, volume.shape), containing (s_xx, s_yy, s_xy).
            If image is 3D, with shape (6, volume.shape), containing (s_xx, s_yy, s_zz, s_xy, s_xz, s_yz).
        val, vec: ndarray
            Eigenvectors and eigenvalues of the structure tensor.
        scale: ndarray
            Scale values chosen for each pixels based on the structure tensor matrix trace

    Authors: papi@dtu.dk, 2023
    """

    if correctScale and not ring_filter:
        raise ValueError('Scale correction returns incorrect values if ring filter is disabled. Set correctScale=False or ring_filter=True.')

    # Allocate S.
    if image.ndim == 2:
        S_opt = np.empty((3, ) + image.shape, dtype=image.dtype)
        S = np.empty((3, ) + image.shape, dtype=image.dtype)
        S_size=3
    elif image.ndim == 3:
        S_opt = np.empty((6, ) + image.shape, dtype=image.dtype)
        S = np.empty((6, ) + image.shape, dtype=image.dtype)
        S_size=6
    else:
        raise ValueError('Image must be 2D or 3D.')
    
    # Repeat rho if None
    if rho_list is None:
        rho_list = np.repeat(None,len(sigma_list))

    # Loop over scales
    for i in range(len(sigma_list)):
        # Compute structure tensor at a single scale
        S = structure_tensor(image, sigma_list[i], ring_filter, rho_list[i], S, eig_decomp=False, truncate=truncate)

        # Compute trace of the structure tensor matrix
        discr = np.sum(S[0:np.ceil(S_size/2).astype(int)],axis=0)

        if i == 0:
            S_opt = np.copy(S)
            scale_opt = np.ones(image.shape, dtype=float)*sigma_list[i]
            discr_opt = np.copy(discr)
        else:
            S_opt = np.where(discr>discr_opt, S, S_opt)
            scale_opt = np.where(discr>discr_opt, sigma_list[i], scale_opt)
            discr_opt = np.where(discr>discr_opt, discr, discr_opt)

    # Compute eigenvalues and eigenvectors of the structure tensor
    if image.ndim == 2:
        val,vec =  eig_special_2d(S_opt)
    elif image.ndim == 3:
        val,vec =  eig_special_3d(S_opt)

    # Apply optional scale correction
    if correctScale:
        scale_opt = util.correct_scale(scale_opt, val)

    return S_opt,val,vec,scale_opt
    