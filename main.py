
m __future__ import division

import numpy as np
from scipy.stats import multivariate_normal

def calc(x, cube_size=3, sigma=2.5):
    # make sure cube_size is odd:
    assert(cube_size%2 != 0)
    
    center = np.ceil(cube_size/2)
    buffer = (cube_size-1)//2 # integer_division
    
    def unfold_tensor(array,mode):
      return np.rollaxis(array,mode,0).reshape(array.shape[mode],-1)
      
    def gaussian_kernel(cube_size, buffer):
      x_ = np.linspace(-buffer,buffer,)    
        
      
      multivariate_normal(xyz,mean=0,cov=sigma)
      
      G = np.zeros((cube_size,cube_size,cube_size))



#      for i in xrange(cube_size):
#        for j in xrange(cube_size):
#         for k in xrange(cube_size):
#            G[i,j,k]=np.exp(-0.5*)
      
      
      
      
      return G
    
    CC1 = np.zeros() # should be size of 2D input in x
    CC2 = np.zeros() # should be size of 2D input in x
    CC3 = np.zeros() # should be size of 2D input in x


    center = np.ceil(cube_size/2)









    x = np.array(x)
    dims = x.shape
    cube_size = int(cube_size)
    GoT_value = np.zeros(x.shape[1:])
    npad = ((cube_size*2-1-dims[0],0), (cube_size-1,cube_size-1),(cube_size-1, cube_size-1))
    x = np.pad(x, pad_width=npad, mode='symmetric')

    def dissimilarity(x1, x2):
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)
        temp1 = np.fft.fftn(np.abs(x1 - x2), axes=(0, 1, 2))
        temp2 = np.fft.fftn(np.abs(temp1), axes=(0, 1, 2))
        temp3 = np.abs(temp2)
        return np.mean(temp3)

    for i in range(dims[1]):
        for j in range(dims[2]):

            analysis_cube_11 = x[:cube_size, i+int((cube_size-1)/2):i+int(3*(cube_size-1)/2)+1, j+int((cube_size-1)/2):j+int(3*(cube_size-1)/2)+1]

            analysis_cube_12 = x[cube_size-1:, i+int((cube_size-1)/2):i+int(3*(cube_size-1)/2)+1, j+int((cube_size-1)/2):j+int(3*(cube_size-1)/2)+1]

            analysis_cube_21 = x[int((cube_size-1)/2):int(3*(cube_size-1)/2+1), i:i+cube_size, j+int((cube_size-1)/2):j+int(3*(cube_size-1)/2)+1]

            analysis_cube_22 = x[int((cube_size-1)/2):int(3*(cube_size-1)/2+1), i+cube_size-1:i+2*cube_size-1, j+int((cube_size-1)/2):j+int(3*(cube_size-1)/2)+1]

            analysis_cube_31 = x[int((cube_size-1)/2):int(3*(cube_size-1)/2+1), i+int((cube_size-1)/2):i+int(3*(cube_size-1)/2)+1, j:j+cube_size]

            analysis_cube_32 = x[int((cube_size-1)/2):int(3*(cube_size-1)/2+1), i+int((cube_size-1)/2):i+int(3*(cube_size-1)/2)+1, j+cube_size-1:j+2*cube_size-1]

            dis1 = dissimilarity(analysis_cube_11, analysis_cube_12)
            dis2 = dissimilarity(analysis_cube_21, analysis_cube_22)
            dis3 = dissimilarity(analysis_cube_31, analysis_cube_32)

            GoT_value[i, j] = (dis1 + dis2 + dis3)

    return np.tanh(1 + np.abs(GoT_value)/np.max(np.abs(GoT_value)))

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
