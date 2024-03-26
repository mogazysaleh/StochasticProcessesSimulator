# imports
import numpy as np
import scipy
import scipy.io as sio

def get_mean(arr : np.array):
    '''
        returns the mean of all elements in a given 1D numpy array
    '''
    return np.sum(arr) / arr.size

def get_variance(arr : np.array):
    '''
        returns the variance of all elements in a given 1D numpy array
    '''
    return get_mean(np.square(arr)) - np.square(get_mean(arr))


def get_3rd_moment(arr: np.array):
    '''
        returns the 3rd moment of all elements in a given 1D numpy array
    '''
    return get_mean(np.power(arr, 3))


def get_mgf(arr : np.array, order=0, range=(0, 2), step=0.01):
    '''
        Calculates the moment generating function (MGF) of a given random variable for every value on the range.

        Inputs:
            - arr: 1D vector of values of the sample space.
            - order: specifies the order of the derivative of MGF, with order=0 being defined as no derivative at all.
            - range: the range of the variable t.
            - step: resolution of the sampling.

        Outputs:
            - MGF of the input random variable.
            - t values at which the MGF was generates.
    '''

    # generate values of t at which the moment generating function will be evaluated
    T = np.arange(range[0], range[1] + step, step)

    # calcuate MGF for each value of t
    MGF = []
    for t in T:
        
        # get x^n corresponding to taking the n-th derivative of the mgf
        x_order = np.power(arr, order)

        # get e^(xt)
        exp = np.exp(np.multiply(arr, t))

        # calculate the mgf for the current t
        mgf = get_mean(np.multiply(x_order, exp))

        # append value to the list of MGF values
        MGF.append(mgf)

    # cast the MGF list into a numpy array
    MGF = np.array(MGF)

    return MGF, T

def get_ensemble_mean(ensemble : np.array):
    '''
        Calculates the ensemble mean of a given ensemble of random processes.
        Assumes the i-th sample function is accessible using ensemble[i].

        Outputs:
            - A random sequence representing the ensemble mean.
    '''

    ensemble_mean = []

    # loop over i-th entries in all processes
    for i in range(ensemble.shape[1]):

        # append the ensemble mean for the i-th entry
        ensemble_mean.append(get_mean(ensemble[:, i]))

    # cast the list into a numpy array
    ensemble_mean = np.array([ensemble_mean])

    return ensemble_mean

def get_acf_matrix(ensemble: np.array):
    '''
        Calculates the Auto-Correlation Function (ACF) between every two random variables i,j in the ensemble.
        It is assumed that each vector on axis 1 of the ensemble is a RV.

        Outputs:
            - A 2D matrix where the entry in the i-th and j-th column is the ACF of the i-th and j-th variables.
    '''

    # start with the empty acf matrix
    acf_matrix = np.zeros((ensemble.shape[1], ensemble.shape[1]))

    # loop over RVs to populate the matrix
    for i in range(ensemble.shape[1]):
        for j in range(ensemble.shape[1]):

            # calculate acf between i-th and j-th variable
            acf = get_mean(np.multiply(ensemble[:, i], ensemble[:, j]))

            # set value of the calculated acf in the acf matrix
            acf_matrix[i, j] = acf

    return acf_matrix


def get_PSD(ensemble, time):
    '''
        Calculates the PSD of the random process.
    '''

    # calculate the sampling frequency
    fs = 1 / (time[0, 1] - time[0, 0]);

    
    # calculate the PSD of every sample function
    ensemble_psd = []
    for i in range(ensemble.shape[0]):

        # get psd
        (f, psd) = scipy.signal.welch(ensemble[i, :], fs, nperseg=ensemble.shape[1])

        # append to generate an ensemple of PSDs
        ensemble_psd.append(psd.tolist())

    # convert into 2D numpy array
    ensemble_psd = np.array(ensemble_psd)

    # return the ensemble mean of the PSD ensemble
    return f, get_ensemble_mean(ensemble_psd)

def get_time_mean(sample_function, T):
    '''
        Calculates the time mean of a given sample function

        Inputs:
            - sample_function: a 1D numpy array representing the sample function.
            - T: time interval of the sample function.

        Outputs:
            - a single value that is the time mean.
    '''

    return np.sum(sample_function) / T

def get_total_power(ensemble, time):
    
    '''
        Returns the total average power of the random process.
    '''

    # get total time interval
    t = time[0, -1] - time[0, 0]

    # calculate the average power for each sample
    powers = []
    for i in range(ensemble.shape[0]):

        # calculate value of power for the current sample
        power = np.sum(np.square(ensemble[i, :])) / np.size(ensemble[i, :])

        # append power to list of powers
        powers.append(power)

    # return the average of all powers
    return get_mean(np.array(powers))

def autocorrelation_function(data):
    '''
    Computes the autocorrelation function of a given time series data.
    '''

    n = len(data)

    acf = np.zeros(n)

    for lag in range(n):
        cross_product = sum(data[lag:] * data[:n - lag])
        acf[lag] = cross_product

    lags = np.arange(0, len(acf))
    return lags, acf / n


def generate_uniformRV(a, b):

    '''
        Generates a uniformly distributed RV and sames it to a .mat file
    '''

    # Generate 101 samples from a uniform distribution between -3 and 5
    theta_samples = np.random.uniform(a, b, 100000)

    # save to a .mat file
    sio.savemat('Uniform_Variable.mat', {'s': theta_samples})

    return theta_samples


def generate_normalRV(m, v):

    '''
        Generates a uniformly distributed RV and sames it to a .mat file
    '''

    
    theta_samples = np.random.normal(m, np.sqrt(v), 100000)

    # save to a .mat file
    sio.savemat('Normal_Variable.mat', {'s': theta_samples})

    return theta_samples


def generate_uniform_process(a=0, b=np.pi, t0=0, t1=2, step=0.02):

    # Generate time values from 0 to 2 with a step 
    t = np.arange(t0, t1, step)

    # Generate 101 random values for θ from U(a, b)
    theta_values = np.random.uniform(a, b, 100)

    # Initialize an array to store Z(t) for each θ
    Z_t_values = np.zeros((100, len(t)))

    # Generate the random process Z(t) for each θ
    for i, theta in enumerate(theta_values):
        Z_t_values[i, :] = np.cos(4 * np.pi * t + theta)


    t = np.array([t.tolist()])
    # save to a .mat file
    sio.savemat('Uniform_Process.mat', {'X': Z_t_values, 't': t})

    return Z_t_values


def generate_normal_process(m=-5, v=5, t0=0, t1=2, step=0.02):

    # Generate time values from 0 to 2 with a step 
    t = np.arange(t0, t1, step)

    # Generate Normal values for A with N(-5, 5)
    A_values = np.random.normal(m, np.sqrt(v), 100)

    # Initialize an array to store W(t) for each A
    W_t_values = np.zeros((100, len(t)))

    
    # Generate the random process W(t) for each A
    for i, A in enumerate(A_values):
        W_t_values[i, :] = A * np.cos(4 * np.pi * t)

    t = np.array([t.tolist()])
    # save to a .mat file
    sio.savemat('Normal_Process.mat', {'X': W_t_values, 't': t})

    