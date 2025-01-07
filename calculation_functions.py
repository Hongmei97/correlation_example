import numpy as np
import pandas as pd
import scipy
from scipy import signal




#________________________________________________________________________________________
def calc_corr_dir(u_dir, v_dir):
    '''
    :param u_dir: 2d array of the director field horizontal vector
    :param v_dir: 2d array of the director field vertical vector 
    :return: 2d array of the correlation function of the direcor field

    step 1: compute the autocorrelation of u^2, v^2 and uv seperately and also two Ones 2d array
    step 2: compute the autocorrelation of two Ones 2d array to get the vector number distribution
    step 3: assemble the cos2theta 2d array =  2 * (autocorr_u_squre + autocorr_v_squre + 2*autocorr_uv) - autocorr_ones
    step 4: average the cos2theta 2d array by the vector number distribution 2d array
    '''

    # compute the autocorrelation of u^2, v^2 and uv, seperately
    u_squre = np.square(u_dir)
    v_squre = np.square(v_dir)
    u_times_v = u_dir * v_dir

    # flip the 2nd 2d array first, calculate the convolution will get the autocorrelation
    autocorr_u_squre = scipy.signal.fftconvolve(u_squre, np.flip(np.flip(u_squre, axis=0), axis=1), mode='same')
    autocorr_v_squre = scipy.signal.fftconvolve(v_squre, np.flip(np.flip(v_squre, axis=0), axis=1), mode='same')
    autocorr_uv = scipy.signal.fftconvolve(u_times_v, np.flip(np.flip(u_times_v, axis=0), axis=1), mode='same')

    # calculate the numbers of vector elements
    Ones = np.ones_like(autocorr_uv)
    number_of_elements = scipy.signal.fftconvolve(Ones, np.flip(np.flip(Ones, axis=0), axis=1), mode='same')
    
    # assemble to the correlation function 2d array
    autocorr = 2 * (autocorr_u_squre + autocorr_v_squre + 2*autocorr_uv) - number_of_elements

    # average the correlation 2d array by vector distribution, namely, number of vector elements
    corr_director = autocorr/number_of_elements

    return corr_director




#________________________________________________________________________________________
def calc_corr_norm(u, v):
    '''
    :param u: 2d array of the vector field horizontal vector
    :param v: 2d array of the vector field vertical vector 
    :return: 2d array of correlation function of the vector field, which is normalized first and then do the average

    step 1: normalized the raw vectors u and v to unit vectors
    step 2: compute the autocorrelation of u and v seperately, and sum them up
    step 3: compute the autocorrelation of two Ones 2d array to get the vector number distribution 
    step 4: average the autocorrelation by the vector distribution
    '''

    # Compute normalized u and v 
    orientation = np.arctan2(v, u)
    u_norm = np.cos(orientation)
    v_norm = np.sin(orientation)

    # compute the autocorrelation of u and v seperately
    autocorr_u = scipy.signal.fftconvolve(u_norm, np.flip(np.flip(u_norm, axis=0), axis=1), mode='same')
    autocorr_v = scipy.signal.fftconvolve(v_norm, np.flip(np.flip(v_norm, axis=0), axis=1), mode='same')
    autocorr = autocorr_u + autocorr_v

    # calculate the numbers of vector elements
    Ones = np.ones_like(autocorr)
    number_of_elements = scipy.signal.fftconvolve(Ones, np.flip(np.flip(Ones, axis=0), axis=1), mode='same')

    # average the autocorrelation by the numbers of vector elements
    corr_unit_vector = autocorr/number_of_elements

    return corr_unit_vector





#________________________________________________________________________________________
# normalization by GLOBAL variance
def calc_corr_glo(u, v):
    '''
    :param u: 2d array of the vector field horizontal vector
    :param v: 2d array of the vector field vertical vector 
    :return: 2d array of correlation function of the vector field, which is averaged first and then normalized by global variance
    
    step 1: compute the autocorrelation of u and v seperatly and sum up, which derived the autocorrelation of the vector field
    step 2: compute the autocorrelation of two Ones 2d array to get the vector number distribution 
    step 2: average the autocorrelation by the vector number distribution, to get the averaged autocorrelation
    step 3: normalize the averaged autocorrelation by the variance of the whole frame
    '''


    # calculate the autocorrelation of u and v
    autocorr_u = scipy.signal.fftconvolve(u, np.flip(np.flip(u, axis=0), axis=1), mode='same')
    autocorr_v = scipy.signal.fftconvolve(v, np.flip(np.flip(v, axis=0), axis=1), mode='same')
    autocorr = (autocorr_u + autocorr_v)

    # calculate the numbers of vector elements
    Ones = np.ones_like(autocorr)
    number_of_elements = scipy.signal.fftconvolve(Ones, np.flip(np.flip(Ones, axis=0), axis=1), mode='same')

    # average the correlation 2d array by the numbers of vector elements
    autocorr_averaged = autocorr/number_of_elements

    # normalize the averaged correlation by its global variance
    global_variance = autocorr_averaged[len(autocorr_averaged)//2][len(autocorr_averaged[0])//2]
    corr_normalized_global = autocorr_averaged/global_variance

    return corr_normalized_global


#________________________________________________________________________________________
# Average over the local varience of kenel
def calc_corr_loc(u, v):
    '''
    :param u: 2d array of the vector field horizontal vector
    :param v: 2d array of the vector field vertical vector 
    :return: 2d array of correlation function of the vector field, which is averaged first and then normalized by the local variance

    step 1: compute the autocorrelation of u and v seperatly, and sum them up
    step 2: compute the covolution of square_u_v and ones
    step 3: divide autocorrelation by the convolution of square_u_v
    '''

    # calculate the autocorrelation of u and v
    autocorr_u = scipy.signal.fftconvolve(u, np.flip(np.flip(u, axis=0), axis=1), mode='same')
    autocorr_v = scipy.signal.fftconvolve(v, np.flip(np.flip(v, axis=0), axis=1), mode='same')
    autocorr = (autocorr_u + autocorr_v)

    # calculate the sum of local variance
    Ones = np.ones_like(autocorr)
    squared_velocity = np.square(u) + np.square(v)
    local_variance_sum = scipy.signal.fftconvolve(np.flip(np.flip(squared_velocity, axis=0), axis=1), Ones, mode='same', axes=None)

    # normalize the autocorrelation by the sum of local variance
    corr_normalized_local = autocorr/local_variance_sum

    return corr_normalized_local



#________________________________________________________________________________________
# Normalize through the geomatric mean of variance
def calc_corr_gm(u, v):
    '''
    :param u: 2d array of the vector field horizontal vector
    :param v: 2d array of the vector field vertical vector 
    :return: 2d array of correlation function of the vector field, 
             which is averaged first and then normalized by the geomatric mean of the local variance and its flipped one
    '''

    # calculate the autocorrelation of u and v
    autocorr_u = scipy.signal.fftconvolve(u, np.flip(np.flip(u, axis=0), axis=1), mode='same')
    autocorr_v = scipy.signal.fftconvolve(v, np.flip(np.flip(v, axis=0), axis=1), mode='same')
    autocorr = (autocorr_u + autocorr_v)

    # calculate sum of local variance and flipped local variance
    Ones = np.ones_like(autocorr)
    squared_velocity = np.square(u) + np.square(v)
    local_variance_sum = scipy.signal.fftconvolve(squared_velocity, Ones, mode='same', axes=None)
    local_variance_flipped_sum = np.flip(np.flip(local_variance_sum, axis=0), axis=1)
    GM = np.sqrt(local_variance_flipped_sum * local_variance_sum)

    # normalize the autocorrelation by the geomatric mean
    corr_normalized_geomatric = autocorr/GM

    return corr_normalized_geomatric


#________________________________________________________________________________________
# Normalize through the Arithmetic mean of variance
def calc_corr_am(u, v):
    '''
    :param u: 2d array of the vector field horizontal vector
    :param v: 2d array of the vector field vertical vector 
    :return: 2d array of correlation function of the vector field, 
             which is averaged first and then normalized by the arithmetic mean of the local variance and its flipped one
    '''

    # calculate the autocorrelation of u and v
    autocorr_u = scipy.signal.fftconvolve(u, np.flip(np.flip(u, axis=0), axis=1), mode='same')
    autocorr_v = scipy.signal.fftconvolve(v, np.flip(np.flip(v, axis=0), axis=1), mode='same')
    autocorr = (autocorr_u + autocorr_v)

    # Calculate the sum of local variance and its flipped local variance
    Ones = np.ones_like(autocorr)
    squared_velocity = np.square(u) + np.square(v)
    local_variance = scipy.signal.fftconvolve(squared_velocity, Ones, mode='same', axes=None)
    local_variance_flipped = np.flip(np.flip(local_variance, axis=0), axis=1)
    AM = (local_variance + local_variance_flipped ) / 2

    # normalized the autocorrelation by the arithmetic mean
    corr_normalized_arithmetic = autocorr/AM

    return corr_normalized_arithmetic




#________________________________________________________________________________________
# calculate the azimuthally averaged correlation function and its corresponding distance 1d array
def corr_to_1d(x, y, autocorr_matrix):
    '''
    :param x: 2d array of the horizontal location, the original point is located in the center of the 2d array
    :param y: 2d array of the vertical location, the original point is located in the center of the 2d array
    :param autocorr_matrix: 2d array of the correlation function of the field, the original point represents the perfect correlation
    :return: 1d array of correlation function that is averaged azimuthally, and its corresponding distance
    '''


    dist_matrix = np.sqrt((x)**2 + (y)**2)

    df = pd.DataFrame(
        {
        'dist_matrix':dist_matrix.reshape(-1),
        'correlation':autocorr_matrix.reshape(-1),
        }
    )

    df = df.sort_values(by=['dist_matrix'])
    df_grouped = df.groupby('dist_matrix').mean().reset_index()

    corr_mean = df_grouped['correlation'].to_numpy()
    dist_mean = df_grouped['dist_matrix'].to_numpy()

    return  dist_mean, corr_mean






