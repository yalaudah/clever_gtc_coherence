
from __future__ import division
import numpy as np
from numpy.linalg import eigvals
from scipy.stats import multivariate_normal

def calc(x, cube_size=3, sigma=15):
    # make sure cube_size is odd:
    assert (cube_size % 2 != 0)

    x = np.array(x)
    dims = x.shape
    cube_size = int(cube_size)
    npad = ((cube_size * 2 - 1 - dims[0], 0), (cube_size - 1, cube_size - 1), (cube_size - 1, cube_size - 1))
    x = np.pad(x, pad_width=npad, mode='symmetric')

    buffer = (cube_size - 1) // 2

    def unfold_tensor(array, mode):
        return np.rollaxis(array, mode, 0).reshape(array.shape[mode], -1)

    def gaussian_kernel(cube_size):
        xx, yy, zz = np.mgrid[-1.0:1.0:cube_size * 1j, -1.0:1.0:cube_size * 1j, -1.0:1.0:cube_size * 1j]
        xyz = np.column_stack([xx.flat, yy.flat, zz.flat])
        G = multivariate_normal.pdf(xyz, mean=[0, 0, 0], cov=sigma)
        G = (2 * np.pi * sigma) ** (3.0 / 2.0) * G
        return G.reshape([cube_size, cube_size, cube_size])

    CC1 = np.zeros((dims[1], dims[2]))
    CC2 = np.zeros((dims[1], dims[2]))
    CC3 = np.zeros((dims[1], dims[2]))

    G = gaussian_kernel(cube_size)

    for i in xrange(buffer, dims[1] - buffer):
        for j in xrange(buffer, dims[2] - buffer):
            aCube = x[:cube_size, i - buffer:i + buffer + 1, j - buffer:j + buffer + 1]
            aCube = aCube * G

            # Mode 1:
            D1 = unfold_tensor(aCube, 0)
            colMeans = np.mean(D1, axis=0)
            D1_tilde = D1 - colMeans
            C1 = np.matmul(np.transpose(D1_tilde), D1_tilde)
            lambda1 = np.max(eigvals(C1))
            CC1[i, j] = np.absolute(lambda1 / np.trace(C1))

            # Mode 2:
            D2 = unfold_tensor(aCube, 1)
            colMeans = np.mean(D2, axis=0)
            D2_tilde = D2 - colMeans
            C2 = np.matmul(np.transpose(D2_tilde), D2_tilde)
            lambda2 = np.max(eigvals(C2))
            CC2[i, j] = np.absolute(lambda2 / np.trace(C2))

            # Mode 3:
            D3 = unfold_tensor(aCube, 2)
            colMeans = np.mean(D3, axis=0)
            D3_tilde = D3 - colMeans
            C3 = np.matmul(np.transpose(D3_tilde), D3_tilde)
            lambda3 = np.max(eigvals(C3))
            CC3[i, j] = np.absolute(lambda3 / np.trace(C3))

    coherence = 0.34*CC1 + 0.33*CC2 + 0.33*CC3

    return coherence

def fetch(sliceID, ds_extents, cube_size):
    num_slices_before = int(cube_size-1)
    num_slices_after = int(cube_size-1)
    section_type, section_value = sliceID
    listing = []
    for i in range(-num_slices_before, 0):
        this_value = section_value - ds_extents[section_type + '_step']
        if this_value < ds_extents[section_type + '_min']:
            continue
        listing.append(i)
    listing.append(sliceID)
    for i in range(1, num_slices_after + 1):
        this_value = section_value + ds_extents[section_type + '_step']
        if this_value > ds_extents[section_type + '_max']:
            continue
        listing.append(i)
    return listing
