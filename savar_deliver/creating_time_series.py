import spatial_models as models
import matplotlib.pyplot as plt
from functions import create_random_mode, check_stability
from c_functions import deseason_data, standardize_data, compare_modes

import numpy as np


# Some definitions

nx = 30  
ny = 90 # Each component is 30x30
T = 5000 # Time 


# Setup spatial weights of underlying processes

N = 3 # Three components

noise_weights = np.zeros((N, nx, ny))
modes_weights = np.zeros((N, nx, ny))

# load = 1
spatial_covariance = 0.3    # decrease spatial_covariance if covariance mat not pos. semidef


# There is a function to create random  modes 
_ = create_random_mode((30, 30), plot=True, random = True)

# If no random X is independent of y
_ = create_random_mode((30, 30), plot=True, random = False)


# We can create the modes with it.
noise_weights = np.zeros((N, nx, ny))
noise_weights[0, :, :30] = create_random_mode((30, 30), random = False)
noise_weights[1, :, 30:60] = create_random_mode((30, 30), random = False)
noise_weights[2, :, 60:] = create_random_mode((30, 30), random = False)


# How the modes look like
# plt.imshow(noise_weights.sum(axis=0))
# plt.colorbar()
# plt.show()


# We can use the same
modes_weights = noise_weights

# Or just assume that the mean value of the whole regions is used.
#noise_weights = np.zeros((N, nx, ny))
#noise_weights[0, :, :30] = 0.01
#noise_weights[1, :, 30:60] = 0.01
#noise_weights[2, :, 60:] = 0.01

# And the causal model
links_coeffs = {
    0: [((0, 1), 0.5), ((2, 2), -0.4)],
    1: [((1, 1), 0.5), ((0, 1), 0.4)],
    2: [((2, 1), 0.5), ((1, 1), 0.4)]
}

# One good thing of SAVAR is that if the underlying process is stable and stationary, then SAVAR is also both. 
# Independently of W. This is, we only need to check for stationarity of \PHI and not of W^+\PHI W
check_stability(links_coeffs)



# This warning tell us that \Sigma is not positive semidefinite We have to change the spatial covariance.
# See: https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.random.multivariate_normal.html
# A good value using create_random_mode is 0.2
spatial_covariance = 0.2
savar = models.savarModel(
    links_coeffs=links_coeffs,
    modes_weights=modes_weights, 
    ny=ny,
    nx=nx,
    T=5000,
    spatial_covariance=spatial_covariance,
    spatial_factor=0.1,
    noise_weights=noise_weights,
    transient = 200, 
    n_variables = N,
    verbose = True
)
savar.create_linear_savar_data()