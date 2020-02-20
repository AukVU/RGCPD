# C imports first
cimport numpy as np

# other imports
import numpy as np
import itertools as it
import scipy.stats as st


# Type declarations
DTYPE = np.float
ctypedef np.float_t DTYPE_t
ctypedef np.int_t DTYPE_int_t

# Savar model
def create_graph(dict links_coeffs, bint return_lag = True):  # bint is bool in cython

    """
    :param links_coeffs:
    :param return_lag: if True, return max lag, otherwise returns only np.ndarray
    From the shape of [j, i, tau]
    :return:
    """

    # Define N
    cdef int N = len(links_coeffs)

    # We find the max_lag
    # TODO: No pot ser cpdef perque hi ha el for despres del =
    cdef int lag
    cdef int max_lag = max(abs(lag)
                  for (_, lag), _ in it.chain.from_iterable(links_coeffs.values()))

    # We create an empty graph
    cdef DTYPE_t[:, :, ::1] graph = np.zeros((N, N, max_lag + 1), dtype=DTYPE)

    # Compute the graph values
    cdef Py_ssize_t i, j, tau  # Py_ssize_t is the proper C type for Python array indices.
    cdef DTYPE_t coeff
    for j, values in links_coeffs.items():
        for (i, tau), coeff in values:
            graph[j, i, abs(tau) - 1] = coeff if tau != 0 else 0

    if return_lag:
        return np.asarray(graph), max_lag
    else:
        return np.asarray(graph)

def compute_linear_savar(np.ndarray data_field_noise, np.ndarray weights,
                         np.ndarray graph, bint w_time_dependant = False):

    """
    Computes the time series of a savar model
    :param data_field_noise: contains the noise at y level
    :param weights noise
    :param time_steps numer of time-steps to compute
    :param w_time_dependant If Ture, then the dimensions of W have time in the first axis
    """

    # D = data_field \in R^(T, l)
    # W \in R^(l,n)
    # U = W^(-1)
    # N = D*W
    # N \in R^(T, n)
    # N*U = D*W*U -> N*U = D*1

    print("We are inside compute linear savar")

    #TODO: check all the dimensions requiered and more asertions

    # definitions
    cdef unsigned int n_var = graph.shape[0]
    cdef unsigned int max_lag = graph.shape[2]
    cdef unsigned int locations = data_field_noise.shape[1]

    # Network data
    cdef DTYPE_t[:, ::1] network_data
    cdef DTYPE_t[::1] network_data_next

    # Moore-Rose pseudoinverse
    cdef DTYPE_t[:, ::1] U
    if not w_time_dependant:
        U = np.linalg.pinv(weights)

    # In case of W_t
    cdef unsigned int time_steps = data_field_noise.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=3] U_t = np.zeros((time_steps, n_var, locations)) # Time inverse
    # U (t, n, l)

    cdef Py_ssize_t t
    if w_time_dependant:
        for t in range(time_steps):
            U_t[t, :, :] = np.linalg.pinv(weights[t, :, :])

    # Iterate in time
    if not w_time_dependant:  # Normal case, W invariant to time
        for t in range(max_lag, time_steps):
            network_data = data_field_noise[t-max_lag:t, :] @ weights
            # Network_data_next (i)
            network_data_next = compute_next_time_step(network_data = np.asarray(network_data).transpose(),
                                                       graph = graph)
            data_field_noise[t, :] += np.dot(network_data_next, U)

    else:  # Now w is variant to time

        for t in range(max_lag, time_steps):
            # y (t, l, 1) x W_t(t, l, n) = (t, l, n) SUM l = (t, i)
            network_data = np.sum(data_field_noise[t-max_lag:t, :,
                                                   np.newaxis]*weights[t-max_lag:t, :, :], axis=1)

            # Network_data_next (i)
            network_data_next = compute_next_time_step(network_data = np.asarray(network_data).transpose(),
                                                       graph = graph)

            data_field_noise[t, :] += np.dot(network_data_next, U_t[t, :, :])

    # Recover the network data
    if not w_time_dependant:
        network_data = data_field_noise @ weights
    else:
        print("this has been reached")
        network_data = np.sum(data_field_noise[:, :, np.newaxis]*weights, axis=1)

    return data_field_noise, np.asarray(network_data)


cdef compute_next_time_step(np.ndarray network_data, np.ndarray graph):
    """
    Computes the next time step for a network in savar model
    Network_data must be (n_var, time-max_lag:time)
    """

    # some definitions
    cdef unsigned int n_var = graph.shape[0]
    cdef unsigned int max_lag = graph.shape[2]

    # Check if network data is (n_var, max_lag)
    assert network_data.shape[0] == n_var, "The number of variables of network and graph do not match"
    assert network_data.shape[1] == max_lag,  "The max_lag of network and graph do not match"

    # Store the past values
    cdef DTYPE_t[:, :, ::1] data_past  # Define data_past as a memory view
    data_past = np.repeat(
        network_data[:, :][:, ::-1].reshape(1, n_var, max_lag),
        n_var, axis=0)

    # Return the result of the operation (transpose it so is time X n_var)
    return (data_past * graph).sum(axis=(2,1)).transpose()


def generate_cov_matrix(np.ndarray noise_weights, float spatial_covariance,
                        float variance= 1, str method = "geometric_mean"):

    """
    :param noise_weights:
    :param spatial_covariance:
    :return:
    """

    # Basic definitions
    cdef Py_ssize_t N = noise_weights.shape[0]
    cdef Py_ssize_t L = noise_weights.shape[1]
    cdef Py_ssize_t K = noise_weights.shape[2]

    if method not in ("geometric_mean", "equal_noise"):
        raise Exception("method must be geometric_mean or equal_noise")

    # Define matrices used
    cdef np.ndarray[DTYPE_t, ndim=2] cov
    cov = np.zeros((L*K, K*L), dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] cov_flat
    cov_flat = np.zeros((L*K*K*L), dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=1] flat_weight

    # Fill de related areas according to noise weights
    cdef int i
    cdef np.ndarray n

    if method == "geometric_mean":
        for n in noise_weights:
            flat_weight = n.reshape(L*K)
            print(flat_weight.max())
            cov_flat += np.sqrt(np.kron(flat_weight,flat_weight))/N
        cov = cov_flat.reshape((L*K, K*L))*spatial_covariance

    if method == "equal_noise":
        for n in noise_weights:
            nonzero = n.reshape(L * K).nonzero()[0]
            # print(np.c_[nonzero].shape)
            for i in nonzero:
                # same factor for all cells that influence other cells within same weight-block
                cov[i, nonzero] = spatial_covariance

    # Set diagonal to 1
    np.fill_diagonal(cov, variance)

    return cov


# Other functions
cpdef create_load(tuple dimensions, float var = 1., float mean = 0.):
    """
    creates a random matrix.
    :param dimensions: 
    :param var: 
    :param mean: 
    :return:  np.ndarray of shape dimensions filled with random numbers
    """
    cdef unsigned int x, y
    x, y = dimensions
    return np.random.randn(x, y)*var+mean

## LINEAR REGRESSION
cpdef estimate_coef_linear_regression(np.ndarray x, np.ndarray y, bint return_intercept = False):

    """
    Computes a smiple linear regression between two vecotrs
    y_i = \beta_0 + \beta_1 x_i + \epsilon_i
    \beta_1 = \frac{SS_{xy}}{SS_{xx}}
    \beta_0 = \overbar{y}-\beta_1 \overbar{x}
    :param x_a:
    :param y_a:
    :param return_intercept:
    :return:
    """

    # number of observations/points
    cdef Py_ssize_t n = np.size(x)

    # mean of x and y vector
    cdef float m_x = np.mean(x)
    cdef float m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    # SS_{xy} = \sum_{i=1}^{n}y_ix_i-n\overbar{x}\overbar{y}
    cdef float SS_xy = np.sum(y*x) - n*m_y*m_x
    cdef float SS_xx = np.sum(x*x) - n*m_x*m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1*m_x

    if return_intercept:
        return b_0, b_1
    else:
        return b_1

def remove_previous_time_step_effect(np.ndarray data, bint normalized = True):
    """
    For each grid_point, remove the effect of the previous one. T is reduced in one.
    :param data: matrix of shape (TxN)
    :param if true we don't use the interception
    :return: matrix size (t-1xN)
    """

    # Number of grid porints time-steps and results
    cdef unsigned int n = data.shape[1]
    cdef unsigned int T = data.shape[0]
    cdef DTYPE_t[:, ::1] results = np.zeros(shape=(T-1, n), dtype=DTYPE)

    # Need it because there the assigment cannot be done from from np.array to memoryview
    cdef DTYPE_t[::1] substraction

    #The elements of the linear regression
    cdef float coeff
    cdef float intercept

    cdef np.ndarray x
    cdef np.ndarray y

    # To iterate
    cdef Py_ssize_t i

    # For each grid point perform the linear regression
    for i in range(n):
        x = data[:-1, i]
        y = data[1:, i]
        coeff, intercept = estimate_coef_linear_regression(x, y, return_intercept=True)

        if normalized:
            substraction = np.subtract(y, np.multiply(coeff,x))
            results[:, i] = substraction
        else:
            substraction = np.subtract(np.substract(y, np.multiply(coeff,x)), intercept)
            results[:, i] = substraction
    return np.asarray(results)

def deseason_data(np.ndarray data, unsigned int period):
    """
    remove seasonality for each n. Data is (TxN)
    """
    cdef unsigned int n = data.shape[1]
    cdef unsigned int t = data.shape[0]

    data = data.reshape(-1, period, n)
    data = data - data.mean(axis=0)

    return data.reshape(t, n)

def standardize_data(np.ndarray data, Py_ssize_t axis=0):
    """
    standardize data
    """
    data -= data.mean(axis=axis)
    data /= data.std(axis=axis)
    return data

def remove_previous_time_step_effect_all(np.ndarray[DTYPE_t, ndim=2] data,
                                        unsigned int reduce_time_size=0):
    """
    Remove the effect of all the others grid points in the previous time-step
    :param reduce_time_size:
    :param data: matrix of shape (TxN)
    :return: matrix size (t-1xN)
    """
    # Number of grid porints time-steps and results
    cdef int n = data.shape[1]
    cdef unsigned int T = data.shape[0]

    cdef np.ndarray[DTYPE_t, ndim=2] results = np.zeros((T-1, n))

    cdef Py_ssize_t i

    for i in range(n):
        print("Starting loop {} of {}".format(i, n))

        results[:, i] = np.asarray(multivariate_linear_time_inf(data[:, i], np.delete(data, i, axis=1),
                                                     reduce_time_size=reduce_time_size))
    print("loop ended")

    return results

cdef np.ndarray[DTYPE_t, ndim=1] multivariate_linear_time_inf(np.ndarray[DTYPE_t, ndim=1] y,
                                                              np.ndarray[DTYPE_t, ndim=2] X,
                                                              unsigned int reduce_time_size = 0):

    """
    Computes b_tilde (of other past variables) and returns residuals
    :param reduce_time_size: 
    :param y: Actual y shape(t)
    :param X: Other variables shape (t, v)
    :reduce_time_size Use only n time-steps to approximate \Beta tilde. use 0 to use all
    b_tilde = (X^TX)^{−1}X^Ty
    y_tilde = Xb_tilde
    residuals = y-y_tilde this is returned
    """
    # In case of reduce_time_size
    cdef unsigned int t_r = reduce_time_size

    # Some definitions
    cdef np.ndarray[DTYPE_t, ndim=1] b_tilde
    cdef np.ndarray[DTYPE_t, ndim=1] y_tilde

    # process X to remove the current time-step
    if t_r is 0:
        X = X[1:, :]
        y = y[:-1]
    else:
        X = X[1:t_r, :]
        y = y[:-1]

    b_tilde = np.linalg.pinv(X.transpose()@X)@X.transpose()@y

    return y-X@b_tilde

def compare_modes(np.ndarray[DTYPE_t, ndim = 2] true_weights, np.ndarray[DTYPE_t, ndim = 2] estimated_weights,
                  np.ndarray[DTYPE_t, ndim = 2] data = None, bint W_t = False,
                  str compar_method = 'both', unsigned int n_modes = 3,
                  bint verbose = False):
    """
    Returns the correlation between the predicted modes and the estimated ones, always returning the
    comparision of the real_mode with the most similar.
    :param true_weights for the modes, in 2 dimensions (N, nx*ny)
    :param estimated_weights for the modes, in 2 dimensions (N, nx*ny)
    :param data shape (t, nx, ny)
    :param compar_method could be 'modes', 'signal' or 'both'. If modes, we look at the pearson correlation
                          between the grid points, if signal between the resulting signal.
    :param W_t if True then w is a function of time and must be in shape (T, N*L)
    :param n_modes the real number of modes
    :param verbose the verbose level
    :return: array like (pearson, and position) for each option (modes, signal)
    """


    print("Start the function")

    # Some assertions
    if compar_method not in ['modes', 'signal', 'both']:
        raise ValueError("compar_method must be, modes, signal or both")
    if W_t and compar_method not in ['signal']:
        raise ValueError("if w_t then modes cannot be compared...")

    # def used variables
    cdef unsigned int n_components = estimated_weights.shape[0]
    cdef np.ndarray[DTYPE_t, ndim = 2] results_w = np.zeros((n_modes, 2))  # Results for grid comparision
    cdef np.ndarray[DTYPE_t, ndim = 2] results_s = np.zeros((n_modes, 2))  # Results for grid comparision
    cdef np.ndarray[DTYPE_t, ndim = 2] com_matrix_w = np.zeros((n_modes, n_components))
    cdef np.ndarray[DTYPE_t, ndim = 2] com_matrix_s = np.zeros((n_modes, n_components))

    # For w_t
    cdef np.ndarray[DTYPE_t, ndim = 3] true_weights_3
    cdef unsigned int time_steps = data.shape[0]
    cdef unsigned int locations = data.shape[1]

    if W_t:
        true_weights_3 = true_weights.reshape(time_steps, n_modes, locations)

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef float r
    cdef float prob

    # Grid comparision
    if compar_method in ['modes', 'both']:
        for i in range(n_modes):
            for j in range(n_components):
                r, prob = st.pearsonr(true_weights[i, :], estimated_weights[j, :])
                com_matrix_w[i, j] = r

                if verbose:
                    print("The component {} and {} have relation {}".format(i, j, r))

        for i in range(n_modes):
            results_w[i, 0] = np.max(abs(com_matrix_w[i, :]))
            results_w[i, 1] = np.argmax(abs(com_matrix_w[i, :]))

    # Signal comparision
    cdef np.ndarray[DTYPE_t, ndim = 2] true_signal
    cdef np.ndarray[DTYPE_t, ndim = 2] estimated_signal

    print("This has been reached")


    if compar_method in ['signal', 'both']:

        if not W_t:
            true_signal = data @ true_weights.transpose()  # Shape (t, N)
            estimated_signal = data @ estimated_weights.transpose()  # Shape(t, N)
        else:
            # True_weights (t, n, l) data(t, l) -> W(t, l, n) data(t, l, 1) = (t, n)
            true_signal = np.sum(data[:, :, np.newaxis]*np.transpose(true_weights_3, axes=(0, 2, 1)),
                                 axis=1)
            estimated_signal = data @ estimated_weights.transpose()  # Shape(t, N)

        for i in range(n_modes):
            for j in range(n_components):
                r, prob = st.pearsonr(true_signal[:, i], estimated_signal[:, j])
                com_matrix_s[i, j] = r

                if verbose:
                    print("The component {} and {} have relation {}".format(i, j, r))

        for i in range(n_modes):
            results_s[i, 0] = np.max(abs(com_matrix_s[i, :]))
            results_s[i, 1] = np.argmax(abs(com_matrix_s[i, :]))

    if compar_method == 'modes':
        return results_w

    if compar_method == 'signal':
        return results_s

    if compar_method == 'both':
        return results_w, results_s

## Non linear
cdef float aux_func(float xtmp, str which):
    funcDict = {
            "linear": xtmp,
            "quadratic": xtmp ** 2,
            "cubic": xtmp ** 3,
            "inverse": 1. / xtmp,
            "log": np.log(np.abs(xtmp)),
            # "f1"        :   2. * xtmp**2 / (1. + 0.5 * xtmp**4),
            "f2": 1*(xtmp + 5. * xtmp ** 2 * np.exp(-xtmp ** 2 / 20.)),  # THE BEST Changing the 1 changes the non-linearty
            "f3": .05 * (xtmp + 20. * 1. / (np.exp(-2. * (xtmp)) + 1.) * np.exp(-xtmp ** 2 / 100.)),
            "f4": (1. - 4. * xtmp ** 3 * np.exp(-xtmp ** 2 / 2.)) * xtmp,
            # "f4"        :   (1. - 4. * np.sin(xtmp)* xtmp**3 * np.exp(-xtmp**2 / 2.) ) * xtmp,
            # "f5"        :   (1. - 2 * xtmp**2 * np.exp(-xtmp**2 / 2.)) * xtmp,
    }

    return funcDict[which]

cdef float func(float coeff, float x, str coupling):
    return coeff * aux_func(x, coupling)

cpdef unsigned int find_max_lag(links_coeffs):
    """
    Given the links_coeffs returns max lag must be a non-linear one
    """

    cdef unsigned int max_lag = 0
    cdef unsigned int var_j

    for var_j, links in links_coeffs.items():
        for (_, lag_var), _, _ in links:
            if abs(lag_var) > max_lag:
                max_lag = abs(lag_var)

    return max_lag

cdef compute_nonlinear_next_time_step(np.ndarray network_data, dict links_coeffs):
    """
    Computes the next time step for a network in savar model
    Network_data must be (n_var, time-max_lag:time)
    returns a np.array of shape 1 and length = N_var
    """

    # definitions
    cdef unsigned int n_var = len(links_coeffs)
    cdef unsigned int max_lag = find_max_lag(links_coeffs)

    # Check if network data is (n_var, max_lag)
    assert network_data.shape[0] == n_var, "The number of variables of network and graph do not match"
    assert network_data.shape[1] == max_lag,  "The max_lag of network and graph do not match"

    # Return the result of the operation (transpose it so is time X n_var)
    cdef DTYPE_t[::1] x_values = np.zeros(n_var)
    cdef Py_ssize_t var_j
    cdef Py_ssize_t var_i
    cdef Py_ssize_t lag_var
    cdef float coeff
    cdef str function

    for var_j, links in links_coeffs.items():
        for (var_i, lag_var), coeff, function in links:
            x_values[var_j] += func(coeff, network_data[var_i, lag_var-1], function)
    return x_values



    #return (data_past * graph).sum(axis=(2,1)).transpose()  # Return the next-time step in X domain

def compute_nonlinear_linear_savar(np.ndarray data_field_noise, np.ndarray weights,
                         dict links_coeffs):

    """
    Computes the time series of a savar model
    :param data_field_noise: contains the noise at y level
    :param weights noise
    :params time_steps numer of time-steps to compute
    """

    # For linear
    # D = data_field \in R^(T, l)
    # W \in R^(l,n)
    # U = W^(-1)
    # N = D*W
    # N \in R^(T, n)
    # N*U = D*W*U -> N*U = D*1
    # For non-linear use a for loop

    print("We are inside compute nonlinear savar")

    #TODO: check all the dimensions requiered

    # Assert
    assert weights.ndim <= 2, "Ndim is {} and should be two".format(weights.ndim)

    # definitions
    cdef unsigned int n_var = len(links_coeffs)
    cdef unsigned int max_lag = find_max_lag(links_coeffs)

    # Network data
    cdef DTYPE_t[:, ::1] network_data
    cdef DTYPE_t[::1] network_data_next

    # Moore-Rose pseudoinverse
    cdef DTYPE_t[:, ::1] U = np.linalg.pinv(weights)

    # Iterate in time
    cdef Py_ssize_t t
    cdef Py_ssize_t time_steps = data_field_noise.shape[0]


    for t in range(max_lag, time_steps):
        network_data = data_field_noise[t-max_lag:t, :] @ weights
        network_data_next = compute_nonlinear_next_time_step(network_data = np.asarray(network_data).transpose(),
                                                   links_coeffs = links_coeffs)

        data_field_noise[t, :] += np.dot(network_data_next, U)
    # Recover the network data
    network_data = data_field_noise @ weights

    return data_field_noise, np.asarray(network_data)