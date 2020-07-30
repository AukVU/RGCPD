import sys
sys.path.insert(0,'savar_deliver')

import spatial_models as models
import matplotlib.pyplot as plt
from functions import create_random_mode, check_stability
from c_functions import deseason_data, standardize_data#, compare_modes

import numpy as np

np.set_printoptions(threshold=sys.maxsize)
from scipy.signal import savgol_filter

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp
from tigramite.plotting import plot_graph

import seaborn as sns
sns.set()

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from netCDF4 import Dataset

import xarray as xr

import os
import pandas as pd

import shutil

from c_dim_methods import get_varimax_loadings_standard as varimax


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
    elif links_coeffs == 'Model3':
        return {
            0: [((0, 1), 0), ((2,10), 0.2), ((1,10), 0.1)],
            1: [((1, 1), 0), ((3,10), 0.9)],
            2: [((2, 1), 0), ((5,10), 0.2)],
            3: [((3, 1), 0), ((4,10), 0.5), ((5,10), 0.9)],
            4: [((4, 1), 0), ((6,10), 0.4)],
            5: [((5, 1), 0), ((6,10), 0.45)],
            6: [((6, 1), 0), ((7,10), 0.8)],
            7: [((7, 1), 0), ((4,10), 0.9)],
        }
    elif links_coeffs == 'Xavier':
        return {
            0: [((0, 1), 0.5), ((2, 2), -0.4)],
            1: [((1, 1), 0.5), ((0, 1), 0.4)],
            2: [((2, 1), 0.5), ((1, 1), 0.4)]
        }

def plot_timeseries_func(savar, plot_points, settings, output='test'):
    fig, axes = plt.subplots(settings['N'], 2)
    plt.setp(axes, xlim=(1,plot_points), ylim=(-10,10))

    for plot in range(min(settings['N'], 7)):
        axes[plot,0].plot(savar.network_data[:plot_points, plot])
        yhat = savgol_filter(savar.network_data[:plot_points, plot], 51, 3) # window size 51, polynomial order 3
        axes[plot,1].plot(yhat)
    user_dir = settings['user_dir']
    filepath = user_dir + f'/Code_Lennart/results/{output}/plots'
    if os.path.isdir(filepath) != True : os.makedirs(filepath)
    filename = filepath  + '/timeseries.pdf'
    if os.path.isfile(filename):
        os.remove(filename)
    plt.savefig(filename, format='pdf')
    # plt.show()

def pcmci_plot(links_coeffs, settings, results=None, output='test'):
    print('Plotting')
    N = settings['N']
    var_names = range(N)
    var_names = [f'Precur {i}' for i in var_names]
    var_names[0] = 'Target'
    if results == None:
        link_matrix = np.zeros((settings['N'], settings['N'], 11), dtype=bool)
        val_matrix = np.zeros((settings['N'], settings['N'], 11))
        for i, link in enumerate(links_coeffs):
            for sublink in links_coeffs[link]:
                link_matrix[sublink[0][0]][i][sublink[0][1]] = True
                val_matrix[sublink[0][0]][i][sublink[0][1]] = sublink[1]
        user_dir = settings['user_dir']
        filepath = user_dir + f'/Code_Lennart/results/{output}/plots'
        if os.path.isdir(filepath) != True : os.makedirs(filepath)
        filename = filepath  + '/network_tigramite.pdf'
        if os.path.isfile(filename):
            os.remove(filename)
        plot_graph(val_matrix, link_matrix=link_matrix, save_name=filename, var_names=var_names, node_label_size=1)
    elif results != None:
        link_matrix = np.zeros((settings['N'], settings['N'], 11), dtype=bool)
        val_matrix = np.zeros((settings['N'], settings['N'], 11))
        for i, link in enumerate(results):
            for sublink in results[link]:
                link_matrix[sublink[0][0]][i][sublink[0][1]] = True
                val_matrix[sublink[0][0]][i][sublink[0][1]] = sublink[1]
        user_dir = settings['user_dir']
        filepath = user_dir + f'/Code_Lennart/results/{output}/plots'
        if os.path.isdir(filepath) != True : os.makedirs(filepath)
        filename = filepath  + '/network_tigramite_pcmci.pdf'
        if os.path.isfile(filename):
            os.remove(filename)
        plot_graph(val_matrix, link_matrix=link_matrix, save_name=filename, var_names=var_names, node_label_size=1)

def draw_network_func(links_coeffs, settings, results=None, output='test'):
    M = nx.DiGraph()
    M.add_nodes_from(range(settings['N']))
    N = nx.DiGraph()
    N.add_nodes_from(range(settings['N']))
    plt.subplot(111)
    if results != None:
        plt.subplot(121)

    for mode in links_coeffs:
        for link in links_coeffs[mode]:
            start_node = link[0][0]
            end_node = mode
            color = 'red'
            link_strength = link[1]
            if link[1] < 0:
                link_strength = -1 * link[1]
                color = 'blue'
            M.add_edges_from([(start_node, end_node, {'color': color, 'weight': link_strength * 5, 'length': 10})])

    pos = nx.spectral_layout(M)
    pos = graphviz_layout(M, prog='neato', root=0, args='')
    edges = M.edges()
    colors = [M[u][v]['color'] for u,v in edges]
    weights = [M[u][v]['weight'] for u,v in edges]
    nx.draw(M, pos, edges=edges, edge_color=colors, width=weights, with_labels=True)
    plt.title('Real causal map')

    # plt.subplot(122)
    if results != None:
        plt.subplot(122)
        for mode in results:
            for link in results[mode]:
                start_node = link[0][0]
                end_node = mode
                N.add_edges_from([(start_node, end_node, {'color': 'red', 'weight': link[1]*5, 'length': 10})])

        pos = nx.spectral_layout(N)
        pos = graphviz_layout(N, prog='neato', root=0, args='')
        edges = N.edges()
        colors = [N[u][v]['color'] for u,v in edges]
        weights = [N[u][v]['weight'] for u,v in edges]
        nx.draw(N, pos, edges=edges, edge_color=colors, width=weights, with_labels=True)
        plt.title('Found causal map')
        plt.plot([0.5, 0.5], [0, 1], color='black', lw=1,transform=plt.gcf().transFigure, clip_on=False)
    user_dir = settings['user_dir']
    filepath = user_dir + f'/Code_Lennart/results/{output}/plots'
    if os.path.isdir(filepath) != True : os.makedirs(filepath)
    filename = filepath  + '/network.pdf'
    if os.path.isfile(filename):
        os.remove(filename)
    plt.savefig(filename, format='pdf')
    # plt.show()

def create_causal_map(N, settings, verbose=False):
    result = None
    while result == None:
        links_coeffs = {}
        autocorrelation = 0.1 * np.random.random() + 0.8
        for mode in range(0, N):
            possible_links = list(range(N))
            del possible_links[mode]
            links_coeffs[mode] = [((mode, 10), autocorrelation)]
            start = 1
            if mode == 0:
                for link in range(np.random.randint(1,3)):
                    # choice = np.random.choice(possible_links)
                    # possible_links.remove(choice)
                    # strength = max(0, np.random.uniform(low=0,high=0.1) + settings['signal'])
                    strength = max(0, 0.1 * np.random.uniform(low=0,high=3) + settings['signal']) * np.random.choice([-1,1])
                    if verbose:
                        print(f"Link {link + 1} with strength {strength}")
                    links_coeffs[mode] += [((link + 1, 10), strength)]
            elif mode == (N - 1):
                pass
            else:
                for link in range(np.random.randint(start,4)):
                    choice = np.random.choice(possible_links)
                    possible_links.remove(choice)
                    strength = (0.1 * np.random.uniform(low=-2,high=0) + settings['signal']) * np.random.choice([-1,1])
                    links_coeffs[mode] += [((choice, 10), strength)]
            autocorrelation = 0.9 + np.random.uniform(low=-0.095, high=0.095)
        try:
            check_stability(links_coeffs)
            result = "Generated"
        except:
            # print("New try making causal map")
            pass
    return links_coeffs

# def create_causal_map_inv(N, settings, verbose=False):
#     result = None
#     while result == None:
#         links_coeffs = {}
#         autocorrelation = 0.1 * np.random.random() + 0.8
#         for mode in range(0, N):
#             links_coeffs[mode] = [((mode, 10), autocorrelation)]
#         for mode in range(0, N):
#             possible_links = list(range(N))
#             del possible_links[mode]
#             start = 1
#             if mode == 0:
#                 for link in range(np.random.randint(1,3)):
#                     # choice = np.random.choice(possible_links)
#                     # possible_links.remove(choice)
#                     strength = max(0, np.random.uniform(low=0,high=0.1) + settings['signal'])
#                     if verbose:
#                         print(f"Link {link + 1} with strength {strength}")
#                     links_coeffs[link + 1] += [((mode, 10), strength)]
#             elif mode == (N - 1):
#                 pass
#             else:
#                 for link in range(np.random.randint(start,3)):
#                     choice = np.random.choice(possible_links)
#                     possible_links.remove(choice)
#                     links_coeffs[choice] += [((mode, 10), 0.1 * np.random.uniform(low=-1,high=1) + settings['signal'])]
#             autocorrelation = 0.9 + np.random.uniform(low=-0.095, high=0.095)
#         try:
#             check_stability(links_coeffs)
#             result = "Generated"
#         except:
#             # print("New try making causal map")
#             pass
#     return links_coeffs

def write_nc(savar, data_field, settings, output='test'):
    user_dir = settings['user_dir']
    filename = user_dir + f'/Code_Lennart/results/{output}/NC/{output}.nc'
    nc = Dataset(filename, 'w', format='NETCDF4')
    data = savar.network_data
    # clusters = data.shape[1] - 1
    # columns = clusters / 2
    # columns = int(columns * 2 - 1)
    lon = np.arange(0, (settings['N'] - 1) * settings['nx'],1)
    lat = np.arange(10,10 + settings['nx'],1)
    if settings['area_size'] == 'small':
        lon = np.arange(-140,-120,1)
        lat = np.arange(10,20,1)
    elif settings['area_size'] == 'very_small':
        lon = np.arange(-140,-135,1)
        lat = np.arange(10,15,1)

    # temp_data = np.zeros((data.shape[0], len(lat), len(lon)))
    # # temp_data = np.random.normal(0, 0.01, (data.shape[0], len(lat), len(lon)))

    # step_x = int(len(lon) / columns)
    # step_y = int(len(lat) / 3)
    # for time in range(temp_data.shape[0]):
    #     start = 0
    #     for i in range(int(clusters/2)):
    #         temp_data[time][0:step_y, start:start + step_x, ] += data[time][1 + i]# + np.random.normal(0,1)
    #         start = start + 2 * step_x
    #     start = 0
    #     for i in range(int(clusters/2)):
    #         temp_data[time][2 * step_y:3 * step_y, start:start + step_x] += data[time][1 + int(clusters / 2) + i]# + np.random.normal(0,1)
    #         start = start + 2 * step_x

    print(f'DATA FIELD heeft size: {data_field.shape}')
    temp_data = data_field[:, :, settings['nx']:]

    nc.createDimension('longitude', len(lon))
    nc.createDimension('latitude', len(lat))
    nc.createDimension('time', None)

    longitude = nc.createVariable('longitude', 'i4', 'longitude')
    latitude = nc.createVariable('latitude', 'i4', 'latitude')
    data_nc = nc.createVariable('test_SAVAR', 'f4', ('time', 'latitude', 'longitude'))
    time = nc.createVariable('time', 'i4', 'time')

    longitude[:] = lon #The "[:]" at the end of the variable instance is necessary
    latitude[:] = lat
    data_nc[:,:,:] = temp_data
    time[:] = range(temp_data.shape[0])


    time.units = 'days since 1979-01-01 00:00:00'
    time.calendar = 'gregorian'
    latitude.units = 'degrees_north'
    longitude.units = 'degrees_east'

    nc.close()

    data_dict = {f'{output}_target':data[:,0]}
    filename = user_dir + f'/Code_Lennart/results/{output}/NC/{output}_target.nc'
    nct = Dataset(filename, 'w', format='NETCDF4')
    lon = np.arange(0, (settings['N'] - 1) * settings['nx'],1)
    lat = np.arange(10,10 + settings['nx'],1)
    if settings['area_size'] == 'small':
        lon = np.arange(-140,-120,1)
        lat = np.arange(10,20,1)
    elif settings['area_size'] == 'very_small':
        lon = np.arange(-140,-135,1)
        lat = np.arange(10,15,1)

    tstarget = np.zeros((5, data.shape[0]))
    for i in range(5):
        tstarget[i,:] = data[:,0]
    
    # tstarget = data_field[:, :, :30]

    nct.createDimension('latitude', len(lat))
    nct.createDimension('longitude', len(lon))
    nct.createDimension('cluster', 5)
    nct.createDimension('time', None)

    longitude = nct.createVariable('longitude', 'i4', 'longitude')
    latitude = nct.createVariable('latitude', 'i4', 'latitude')
    ts = nct.createVariable('ts', 'f4', ('cluster', 'time'))
    time = nct.createVariable('time', 'i4', 'time')
    cluster = nct.createVariable('cluster', 'i4', 'cluster')

    longitude[:] = lon #The "[:]" at the end of the variable instance is necessary
    latitude[:] = lat
    ts[:,:] = tstarget
    cluster[:] = range(1,6)
    time[:] = range(tstarget.shape[1])

    time.units = 'days since 1979-01-01 00:00:00'
    time.calendar = 'proleptic_gregorian'



    nct.close()

def save_time_series(savar, settings, output='test'):
    user_dir = settings['user_dir']
    data = savar.data_field @ savar.modes_weights.reshape(settings['N'], -1).transpose()
    periods = data.shape[0]
    # print('save time series')
    filename = user_dir + f'/Code_Lennart/results/{output}/time_series'
    if os.path.isdir(filename) != True : os.makedirs(filename)
    filename = user_dir + f'/Code_Lennart/results/{output}/time_series/timeseries.csv'

    dates = pd.date_range(start='1/1/1979', periods=periods)
    df = pd.DataFrame(dates, columns=['time'])
    df['time'] = pd.to_datetime(df['time'])   
    df['year'] = df.time.dt.year
    df['month'] = df.time.dt.month
    df['day'] = df.time.dt.day
    df.drop('time', axis=1, inplace=True)
    df['target'] = data[:,0]
    for i in range(1, data.shape[1]):
        df[f'ts{i}'] = data[:,i]
    df.to_csv(filename, index=False)

def save_matrices(settings, path, pmatrix, val_matrix=None, iteratelist=None):
    if os.path.isdir(path) != True : os.makedirs(path)
    if iteratelist == None:
        iteratelist = range(settings['N'])
    for i, p in enumerate(iteratelist):
        pmatrixnp = np.matrix(pmatrix[i])
        np.save(path + f"/p_{p}.npy", pmatrixnp)
        if val_matrix is not None:
            valmatrixnp = np.matrix(val_matrix[i])
            np.save(path + f"/val_{p}.npy", valmatrixnp)

def create_real_matrices(settings, general_path, links_coeffs):
    N = settings['N']
    pmatrix = [[[1 for lag in range(3)] for x in range(N)] for y in range(N)]
    val_matrix = [[[0 for lag in range(3)] for x in range(N)] for y in range(N)]
    number_of_links = 0
    for key in range(settings['N']):
        for link in links_coeffs[key]:
            if key == 0:
                number_of_links += 1
            van = link[0][0]
            lag = 1
            val = link[1]
            naar = key
            # print(f'van {van} naar {naar}')
            pmatrix[van][naar][lag] = 0
            val_matrix[van][naar][lag] = val
    real_links = np.ones(N)
    real_links[:number_of_links] = 0
    path = general_path + '/matrices/AAA_real'
    if os.path.isdir(path) != True : os.makedirs(path)
    np.save(path + '/ZZZ_correlated', list(range(N)))
    np.save(path + '/ZZZ_real_links', real_links)
    print(f'\n\n The real links are {real_links}\n\n')
    save_matrices(settings, path, pmatrix, val_matrix)

def create_real_matrices_old(settings, general_path, links_coeffs):
    N = settings['N']
    pmatrix = [[[1 for lag in range(3)] for x in range(N)] for y in range(N)]
    val_matrix = [[[0 for lag in range(3)] for x in range(N)] for y in range(N)]
    number_of_links = 0
    for key in range(settings['N']):
        for link in links_coeffs[key]:
            if link[0][0] == 0:
                number_of_links += 1
            van = key
            lag = 1
            val = link[1]
            naar = link[0][0]
            # print(f'van {van} naar {naar}')
            pmatrix[van][naar][lag] = 0
            val_matrix[van][naar][lag] = val
    real_links = np.ones(N)
    real_links[:number_of_links] = 0
    path = general_path + '/matrices/AAA_real'
    if os.path.isdir(path) != True : os.makedirs(path)
    np.save(path + '/ZZZ_correlated', list(range(N)))
    np.save(path + '/ZZZ_real_links', real_links)
    print(f'\n\n The real links are {real_links}\n\n')
    save_matrices(settings, path, pmatrix, val_matrix)


def create_time_series(settings, links_coeffs, verbose=False, plot_modes=False, plot_timeseries=False, draw_network=False, cluster=False):
    nx, ny, T, N = settings['nx'], settings['ny'], settings['T'], settings['N']
    spatial_covariance = settings['spatial_covariance']
    general_path = settings['user_dir'] + '/' + settings['extra_dir'] + '/results/' + settings['filename']

    if (settings['netcdf4'] or settings['save_time_series'] or settings['do_pcmci'] or
                              settings['save_matrices']    or (settings['plot_points'] != None)):
        remove_path = general_path
        shutil.rmtree(remove_path, ignore_errors=True)
        os.makedirs(remove_path + '/NC')

    if settings['random_causal_map']:
        links_coeffs = create_causal_map(N, settings, verbose)
    elif links_coeffs == None:
        print('Using default causal map of Xavier')
        links_coeffs = get_links_coeffs('Xavier')
    elif type(links_coeffs) == str:
        links_coeffs = get_links_coeffs(links_coeffs)
    else:
        print('Using default causal map of Xavier')
        links_coeffs = get_links_coeffs('Xavier')
    check_stability(links_coeffs)
    N = settings['N'] = len(links_coeffs)

    create_real_matrices(settings, general_path, links_coeffs)
    if verbose and not cluster:
        print('\n---------------------------' +
                f"\nThe settings are:\n\nNumber of modes = {N}\nnx = {nx}, ny = {ny}, T = {T}\nspatial_covariance = {spatial_covariance}\n" +
                f"Random modes = {settings['random_modes']}\nSpatial factor = {settings['spatial_factor']}" +
                f"\nTransient = {settings['transient']}\n" + 
                f"\nPCMCI = {settings['do_pcmci']}" +
                f"\nRandom causal map = {settings['random_causal_map']}" + 
                f"\nWrite to .nc file = {settings['netcdf4']}\n" +'---------------------------\n')


    noise_weights = np.zeros((N, nx, N * nx))
    modes_weights = np.zeros((N, nx, N * nx))

    for mode in range(N):
        modes_weights[mode, :, mode * nx:(mode + 1) * nx] = create_random_mode((nx, nx), random = settings['random_modes'], plot=False)
    
    if plot_modes:
        if not cluster:
            plt.imshow(modes_weights.sum(axis=0))
            plt.colorbar()
            plt.show()
    
    if settings['noise_use_mean']:
        for mode in range(N):
            noise_weights[mode, :, mode * nx:(mode + 1) * nx] = settings['noise_level']
            # print(f"hoi, the noise is {settings['noise_level']}")
    else:
        noise_weights = modes_weights
    

    if draw_network and not cluster:
        draw_network_func(links_coeffs, settings, output=settings['filename'])
        pcmci_plot(links_coeffs, settings, output=settings['filename'])
    # sys.exit()


    savar = models.savarModel(
        links_coeffs=links_coeffs,
        modes_weights=modes_weights, 
        ny=ny,
        nx=nx,
        T=T,
        spatial_covariance=spatial_covariance,
        # covariance_noise_method='geometric_mean',
        # covariance_noise_method='equal_noise',
        variance_noise=settings['noise_level'],
        noise_weights=noise_weights,
        # random_noise=(0,settings['noise_level']),
        transient = settings['transient'], 
        n_variables = N,
        verbose = verbose
    )
    savar.create_linear_savar_data()

    if plot_modes and not cluster:
        modes = varimax(savar.data_field)
        for i in range(5):
            plt.imshow(modes['weights'][:, i].reshape(settings['nx'], settings['ny']))
            plt.colorbar()
            plt.show()

    datafield = savar.data_field.reshape(-1, settings['nx'], settings['ny'])

    if plot_timeseries and not cluster:
        plot_points = settings['plot_points']
        plot_timeseries_func(savar, plot_points, settings, output=settings['filename'])
    
    if settings['netcdf4']:
        write_nc(savar, datafield, settings, output=settings['filename'])

    if settings['save_time_series']:
        save_time_series(savar, settings, output=settings['filename'])

    if settings['do_pcmci']:
        data = savar.data_field @ savar.modes_weights.reshape(N, -1).transpose()
        # data, _ = pp.var_process(links_coeffs, T=1000)
        dataframe = pp.DataFrame(data)

        cond_ind_test = ParCorr()
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        pcmci_output = pcmci.run_pcmci(tau_max=11, pc_alpha=0.01)
        if verbose:
            pcmci.print_significant_links(p_matrix=pcmci_output['p_matrix'],
                                                val_matrix=pcmci_output['val_matrix'],
                                                alpha_level=0.01)


        pcmci_matrix_path = general_path + '/matrices/pcmci_test'
        if os.path.isdir(pcmci_matrix_path) != True : os.makedirs(pcmci_matrix_path)
        np.save(pcmci_matrix_path + '/ZZZ_correlated', list(range(settings['N'])))
        p_matrix = pcmci_output['p_matrix'][:, :, [0,10,11]]
        val_matrix = pcmci_output['val_matrix'][:, :, [0,10,11]]
        save_matrices(settings, pcmci_matrix_path, p_matrix, val_matrix)

        parents = pcmci.all_parents
        val_min = pcmci.val_min

        results = {}
        for mode in parents:
            links_results = []
            for link in parents[mode]:
                value = val_min[mode][link]
                links_results.append(((link[0], -1 * link[1]), value))
            results[mode] = links_results
        if not cluster:
            # draw_network_func(links_coeffs, settings, results=results, output=settings['filename'])
            pcmci_plot(links_coeffs, settings, results=results, output=settings['filename'])

        




# Setting up the settings and running above code

# settings = {}
# settings['N'] = 5
# settings['nx'], settings['ny'], settings['T'] = 30, settings['N'] * 30, 14975 #5114
# settings['spatial_covariance'] = 0.3
# settings['random_modes'] = False
# settings['noise_use_mean'] = False
# settings['transient'] = 200
# settings['spatial_factor'] = 0.1

# settings['user_dir'] = user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'
# settings['extra_dir'] = 'Code_Lennart'
# settings['filename'] = 'test_met_savar2'


# settings['random_causal_map'] = True
# settings['area_size'] = None


# ## If any of the following settings is set to True, the results folder with {filename} will be removed!
# ## Also when 'plot_points' is not None
# settings['netcdf4'] = True
# settings['save_time_series'] = True
# settings['do_pcmci'] = True
# settings['save_matrices'] = True
# settings['plot_points'] = 500


# links_coeffs = 'model3'

# create_time_series(settings, links_coeffs,  verbose=True,
#                                             plot_modes=False,
#                                             plot_timeseries=True,
#                                             draw_network=True)