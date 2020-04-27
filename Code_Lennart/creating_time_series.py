import sys
sys.path.insert(0,'savar_deliver')

import spatial_models as models
import matplotlib.pyplot as plt
from functions import create_random_mode, check_stability
from c_functions import deseason_data, standardize_data, compare_modes

import numpy as np

np.set_printoptions(threshold=sys.maxsize)
from scipy.signal import savgol_filter

from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr
import tigramite.data_processing as pp

import seaborn as sns
sns.set()

import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

from netCDF4 import Dataset

import xarray as xr

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

def plot_timeseries_func(savar, plot_points, settings):
    fig, axes = plt.subplots(settings['N'], 2)
    plt.setp(axes, xlim=(1,plot_points), ylim=(-10,10))

    for plot in range(settings['N']):
        print(type(savar.network_data))
        print(savar.network_data.shape)
        axes[plot,0].plot(savar.network_data[:plot_points, plot])
        yhat = savgol_filter(savar.network_data[:plot_points, plot], 51, 3) # window size 51, polynomial order 3
        axes[plot,1].plot(yhat)
    plt.show()

def draw_network_func(links_coeffs, settings, results=None):
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
            M.add_edges_from([(start_node, end_node, {'color': 'red', 'weight': link[1]*5, 'length': 10})])

    pos = nx.spectral_layout(M)
    # pos = nx.spring_layout(N)
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
        # pos = nx.spring_layout(N)
        pos = graphviz_layout(N, prog='neato', root=0, args='')
        edges = N.edges()
        colors = [N[u][v]['color'] for u,v in edges]
        weights = [N[u][v]['weight'] for u,v in edges]
        nx.draw(N, pos, edges=edges, edge_color=colors, width=weights, with_labels=True)
        plt.title('Found causal map')
        plt.plot([0.5, 0.5], [0, 1], color='black', lw=1,transform=plt.gcf().transFigure, clip_on=False)
    plt.show()

def create_causal_map(N):
    links_coeffs = {}
    autocorrelation = 0.5 * np.random.random() + 0.25
    for mode in range(0, N):
        possible_links = list(range(N))
        del possible_links[mode]
        links_coeffs[mode] = [((mode, 10), autocorrelation)]
        start = 0
        if mode == 0:
            start = 1
        for link in range(np.random.randint(start,3)):
            choice = np.random.choice(possible_links)
            possible_links.remove(choice)
            links_coeffs[mode] += [((choice, 10), 0.5 * np.random.random() + 0.5)]
    print(links_coeffs)
    return links_coeffs

def write_nc(savar, output='test'):
    user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'
    filename = user_dir + f'/Code_Lennart/NC/{output}.nc'
    nc = Dataset(filename, 'w', format='NETCDF4')
    data = savar.network_data
    clusters = data.shape[1] - 1
    columns = clusters / 2
    columns = int(columns * 2 - 1)
    lon = np.arange(-140,-59,1)
    lat = np.arange(10,76,1)

    temp_data = np.zeros((data.shape[0], len(lat), len(lon)))

    step_x = int(81 / columns)
    step_y = int(66 / 3)
    for time in range(temp_data.shape[0]):
        start = 0
        for i in range(int(clusters/2)):
            temp_data[time][start:start + step_x, 0:step_y] = data[time][1 + i]
            start = start + 2 * step_x
        start = 0
        for i in range(int(clusters/2)):
            temp_data[time][start:start + step_x, 2 * step_y:3 * step_y] = data[time][1 + int(clusters / 2) + i]
            start = start + 2 * step_x

    nc.createDimension('lon', len(lon))
    nc.createDimension('lat', len(lat))
    nc.createDimension('time', None)

    longitude = nc.createVariable('longitude', 'f4', 'lon')
    latitude = nc.createVariable('latitude', 'f4', 'lat')
    data_nc = nc.createVariable('test_SAVAR', 'f4', ('time', 'lat', 'lon'))
    time = nc.createVariable('time', 'i4', 'time')

    longitude[:] = lon #The "[:]" at the end of the variable instance is necessary
    latitude[:] = lat
    data_nc[:,:,:] = temp_data
    time[:] = range(temp_data.shape[0])


    time.units = 'days since 1979-01-01 00:00:00'
    time.calendar = 'gregorian'

    nc.close()

    # print(data[:,0])
    data_dict = {f'{output}_target':data[:,0]}
    # print(data_dict)
    # np.savez(user_dir + f'/Code_Lennart/NC/{output}.npz', data[:,0], allow_pickle=False)
    # np.save(user_dir + f'/Code_Lennart/NC/{output}.npy', data_dict, allow_pickle=True)
    filename = user_dir + f'/Code_Lennart/NC/{output}_target.nc'
    nct = Dataset(filename, 'w', format='NETCDF4')
    lon = np.arange(225,301,1)
    lat = np.arange(70,29,-1)
    print('length latitude')
    print(len(lat))

    tstarget = np.zeros((5, data.shape[0]))
    for i in range(5):
        tstarget[i,:] = data[:,0]

    nct.createDimension('latitude', len(lat))
    nct.createDimension('longitude', len(lon))
    nct.createDimension('cluster', 5)
    nct.createDimension('time', None)

    longitude = nct.createVariable('longitude', 'f4', 'longitude')
    latitude = nct.createVariable('latitude', 'f4', 'latitude')
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
    # ts.name = 'ts'

    # nct.set_coords(['lon', 'lat'])



    nct.close()

    ds = xr.open_dataset(filename, decode_cf=True, decode_coords=True, decode_times=False)
    ds.set_coords(['longitude', 'latitude'])
    filename = user_dir + f'/Code_Lennart/NC/{output}_target2.nc'
    ds.to_netcdf(path=filename, mode='w')


    print('hoi')

def create_time_series(settings, links_coeffs, verbose=False, plot_modes=False, plot_timeseries=False, draw_network=False):
    nx, ny, T, N = settings['nx'], settings['ny'], settings['T'], settings['N']
    spatial_covariance = settings['spatial_covariance']

    if settings['random_causal_map']:
        links_coeffs = create_causal_map(N)
    elif links_coeffs == None:
        print('Using default causal map of Xavier')
        links_coeffs = get_links_coeffs('Xavier')
    elif type(links_coeffs) == str:
        links_coeffs = get_links_coeffs(links_coeffs)
    else:
        print('Using default causal map of Xavier')
        links_coeffs = get_links_coeffs('Xavier')
    check_stability(links_coeffs)
    print(links_coeffs)
    N = len(links_coeffs)

    if verbose:
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
    
    if settings['netcdf4']:
        write_nc(savar)
    
    if settings['do_pcmci']:
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
settings['nx'], settings['ny'], settings['T'] = 30, settings['N'] * 30, 5114
settings['spatial_covariance'] = 0.3
settings['random_modes'] = False
settings['noise_use_mean'] = False
settings['transient'] = 200
settings['spatial_factor'] = 0.1

settings['plot_points'] = 500
settings['do_pcmci'] = False
settings['random_causal_map'] = True
settings['netcdf4'] = True


links_coeffs = 'model3'

create_time_series(settings, links_coeffs,  verbose=True,
                                            plot_modes=False,
                                            plot_timeseries=True,
                                            draw_network=True)