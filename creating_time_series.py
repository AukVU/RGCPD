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

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def get_links_coeffs(links_coeffs):
    links_coeffs = links_coeffs.capitalize()
    if links_coeffs == 'Model1':
        return {
            0: [((0, 1), 0), ((2,10), 0.9), ((1,10), 0.4)],
            1: [((1, 1), 0)],
            2: [((2, 1), 0)]
        }
    elif links_coeffs == 'Model2':
        return {
            0: [((0, 1), 0), ((2,10), 0.9), ((1,10), 0.4)],
            1: [((1, 1), 0), ((3,10), 0.9)],
            2: [((2, 1), 0), ((4,10), 0.4)],
            3: [((3, 1), 0)],
            4: [((4, 1), 0)],
        }
    elif links_coeffs == 'Xavier':
        return {
            0: [((0, 1), 0.5), ((2, 2), -0.4)],
            1: [((1, 1), 0.5), ((0, 1), 0.4)],
            2: [((2, 1), 0.5), ((1, 1), 0.4)]
        }

def plot_timeseries_func(savar, plot_points, settings):
    fig, axes = plt.subplots(settings['N'], 2)
    plt.setp(axes, xlim=(1,plot_points), ylim=(-10,10))

    for plot in range(settings['N']):
        axes[plot,0].plot(savar.network_data[:plot_points, plot])
        yhat = savgol_filter(savar.network_data[:plot_points, plot], 51, 3) # window size 51, polynomial order 3
        axes[plot,1].plot(yhat)
    plt.show()

def draw_network_func(links_coeffs, settings, results=None):
    N = nx.DiGraph()
    N.add_nodes_from(range(settings['N']))
    plt.subplot(111)
    if results != None:
        plt.subplot(121)

    for mode in links_coeffs:
        for link in links_coeffs[mode]:
            start_node = mode
            end_node = link[0][0]
            N.add_edges_from([(start_node, end_node, {'color': 'red', 'weight': link[1]*5, 'length': 10})])

    pos = nx.spectral_layout(N)
    # pos = nx.spring_layout(N)
    pos = graphviz_layout(N, prog='neato', root=0, args='')
    edges = N.edges()
    colors = [N[u][v]['color'] for u,v in edges]
    weights = [N[u][v]['weight'] for u,v in edges]
    nx.draw(N, pos, edges=edges, edge_color=colors, width=weights, with_labels=True)
    plt.title('Real causal map')

    # plt.subplot(122)
    if results != None:
        plt.subplot(122)
        for mode in results:
            for link in results[mode]:
                start_node = mode
                end_node = link[0][0]
                N.add_edges_from([(start_node, end_node, {'color': 'red', 'weight': link[1]*5, 'length': 10})])

        pos = nx.spectral_layout(N)
        # pos = nx.spring_layout(N)
        pos = graphviz_layout(N, prog='neato', root=0, args='')
        edges = N.edges()
        colors = [N[u][v]['color'] for u,v in edges]
        weights = [N[u][v]['weight'] for u,v in edges]
        nx.draw(N, pos, edges=edges, edge_color=colors, width=weights, with_labels=True)
        plt.title('Found causal map')
        plt.plot([0.5, 0.5], [0, 1], color='black', lw=1,transform=plt.gcf().transFigure, clip_on=False)
    plt.show()


def create_time_series(settings, links_coeffs, verbose=False, plot_modes=False, plot_timeseries=False, draw_network=False):
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

    if draw_network:
        draw_network_func(links_coeffs, settings)

    savar = models.savarModel(
        links_coeffs=links_coeffs,
        modes_weights=modes_weights, 
        ny=ny,
        nx=nx,
        T=T,
        spatial_covariance=spatial_covariance,
        noise_weights=noise_weights,
        transient = settings['transient'], 
        n_variables = N,
        verbose = verbose
    )
    savar.create_linear_savar_data()

    if plot_timeseries:
        plot_points = settings['plot_points']
        plot_timeseries_func(savar, plot_points, settings)
    
    data = savar.network_data
    # data, _ = pp.var_process(links_coeffs, T=1000)
    dataframe = pp.DataFrame(data)

    cond_ind_test = ParCorr()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
    results = pcmci.run_pcmci(tau_max=11, pc_alpha=0.01)
    pcmci.print_significant_links(p_matrix=results['p_matrix'],
                                        val_matrix=results['val_matrix'],
                                        alpha_level=0.05)
    parents = pcmci.all_parents
    val_min = pcmci.val_min

    results = {}
    for mode in parents:
        links_results = []
        for link in parents[mode]:
            value = val_min[mode][link]
            links_results.append(((link[0], -1 * link[1]), value))
        results[mode] = links_results
    draw_network_func(links_coeffs, settings, results=results)




# Setting up the settings and running above code

settings = {}
settings['N'] = 5
settings['nx'], settings['ny'], settings['T'] = 30, settings['N'] * 30, 5000
settings['spatial_covariance'] = 0.3
settings['random_modes'] = False
settings['noise_use_mean'] = False
settings['transient'] = 200
settings['spatial_factor'] = 0.1
settings['plot_points'] = 500


links_coeffs = 'model2'

create_time_series(settings, links_coeffs,  verbose=True,
                                            plot_modes=False,
                                            plot_timeseries=True,
                                            draw_network=True)