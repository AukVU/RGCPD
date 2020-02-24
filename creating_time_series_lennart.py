import sys
sys.path.insert(0,'savar_deliver')

import spatial_models as models
import matplotlib.pyplot as plt
from functions import create_random_mode, check_stability
from c_functions import deseason_data, standardize_data, compare_modes

import numpy as np
from scipy.signal import savgol_filter

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp

import seaborn as sns
sns.set()

def get_links_coeffs(links_coeffs):
    links_coeffs = links_coeffs.capitalize()
    if links_coeffs == 'Start':
        return {
            0: [((0, 1), 0), ((2,10), 0.9), ((1,10), 0.9)],
            1: [((1, 1), 0)],
            2: [((2, 1), 0)]
        }
    elif links_coeffs == 'Xavier':
        return {
            0: [((0, 1), 0.5), ((2, 2), -0.4)],
            1: [((1, 1), 0.5), ((0, 1), 0.4)],
            2: [((2, 1), 0.5), ((1, 1), 0.4)]
        }

def plot_timeseries_func(savar, plot_points):
    fig, axes = plt.subplots(3, 2)
    plt.setp(axes, xlim=(1,plot_points), ylim=(-10,10))


    axes[0,0].plot(savar.network_data[:plot_points, 0])
    axes[1,0].plot(savar.network_data[:plot_points, 1])
    axes[2,0].plot(savar.network_data[:plot_points, 2])


    yhat = savgol_filter(savar.network_data[:plot_points, 0], 51, 3) # window size 51, polynomial order 3
    axes[0,1].plot(yhat)
    yhat = savgol_filter(savar.network_data[:plot_points, 1], 51, 3) # window size 51, polynomial order 3
    axes[1,1].plot(yhat)
    yhat = savgol_filter(savar.network_data[:plot_points, 2], 51, 3) # window size 51, polynomial order 3
    axes[2,1].plot(yhat)
    plt.show()


def create_time_series(settings, links_coeffs, verbose=False, plot_modes=False, plot_timeseries=False):
    nx, ny, T, N = settings['nx'], settings['ny'], settings['T'], settings['N']
    spatial_covariance = settings['spatial_covariance']

    if verbose:
        print('\n---------------------------' +
                f"\nThe settings are:\n\nNumber of modes = {N}\nnx = {nx}, ny = {ny}, T = {T}\nspatial_covariance = {spatial_covariance}\n" +
                f"Random modes = {settings['random_modes']}\nSpatial factor = {settings['spatial_factor']}" +
                f"\nTransient = {settings['transient']}\n" + '---------------------------\n')

    noise_weights = np.zeros((N, nx, N * nx))
    modes_weights = np.zeros((N, nx, N * nx))

    for mode in range(N):
        modes_weights[mode, :, mode * nx:(mode + 1) * nx] = create_random_mode((30, 30), random = settings['random_modes'])
    
    if plot_modes:
        plt.imshow(modes_weights.sum(axis=0))
        plt.colorbar()
        plt.show()
    
    if settings['noise_use_mean']:
        for mode in range(N):
            noise_weights[mode, :, mode * nx:(mode + 1) * nx] = 0.01
    else:
        noise_weights = modes_weights

    if links_coeffs == None:
        links_coeffs = get_links_coeffs('Xavier')
    elif type(links_coeffs) == str:
        links_coeffs = get_links_coeffs(links_coeffs)
    check_stability(links_coeffs)

    savar = models.savarModel(
        links_coeffs=links_coeffs,
        modes_weights=modes_weights, 
        ny=ny,
        nx=nx,
        T=T,
        spatial_covariance=spatial_covariance,
        spatial_factor=settings['spatial_factor'],
        noise_weights=noise_weights,
        transient = settings['transient'], 
        n_variables = N,
        verbose = verbose
    )
    savar.create_linear_savar_data()

    if plot_timeseries:
        plot_points = settings['plot_points']
        plot_timeseries_func(savar, plot_points)
    
    data = savar.network_data
    # data, _ = pp.var_process(links_coeffs, T=1000)
    dataframe = pp.DataFrame(data)

    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=11, pc_alpha=None)
    pcmci.print_significant_links(p_matrix=results['p_matrix'],
                                        val_matrix=results['val_matrix'],
                                        alpha_level=0.05)




# Setting up the settings and running above code

settings = {}
settings['N'] = 3
settings['nx'], settings['ny'], settings['T'] = 30, settings['N'] * 30, 5000
settings['spatial_covariance'] = 0.1
settings['random_modes'] = False
settings['noise_use_mean'] = False
settings['transient'] = 200
settings['spatial_factor'] = 0.1
settings['plot_points'] = 500


links_coeffs = 'start'

create_time_series(settings, links_coeffs, verbose=True, plot_modes=False, plot_timeseries=True)