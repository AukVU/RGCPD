import os, inspect, sys

if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
user_dir = os.path.expanduser('~')
user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/') 
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)

import creating_time_series as cts
import causal_score

import shutil

from RGCPD import RGCPD
from find_precursors import relabel
from class_BivariateMI_PCMCI import BivariateMI_PCMCI

from class_BivariateMI_PCMCI import corr_map
from class_BivariateMI_PCMCI import entropy_map
from class_BivariateMI_PCMCI import parcorr_map
from class_BivariateMI_PCMCI import granger_map
from class_BivariateMI_PCMCI import gpdc_map
from class_BivariateMI_PCMCI import rcot_map
from class_BivariateMI_PCMCI import cmiknn_map

import numpy as np
import pandas as pd

import itertools
flatten = itertools.chain.from_iterable




def rename_labels(rg):
    all_locs = []
    for precur in rg.list_for_MI:
        prec_labels = precur.prec_labels.copy()
        # prec_labels = prec_labels.median(dim='split')
        if all(np.isnan(prec_labels.values.flatten()))==False:
            split_locs = []
            for split in range(len(prec_labels.values)): 
                labels = np.nan_to_num(prec_labels.values[split])[0]
                shape = labels.shape
                rows, columns = shape[0], shape[1]
                middle, offset = int(rows/2), int(rows/6)
                N_areas = int(columns / rows)
                locs = []
                reassign = {}
                i = 0
                for loc in range(N_areas):
                    area = labels[middle - offset: middle + offset, rows * loc + middle - offset: rows * loc + middle + offset]
                    area_nonzero = np.nonzero(area)
                    if len(area_nonzero[0]) > 0:
                        locs.append(loc+1)
                        value = area[area_nonzero[0][0]][area_nonzero[1][0]]
                        reassign[value] = loc+1
                relabeld = relabel(precur.prec_labels.values[split], reassign).astype('float')
                relabeld[relabeld == 0] = np.nan
                precur.prec_labels.values[split] = relabeld
                split_locs.append(locs)
            all_locs.append(split_locs)
            # all_locs.append(list(set(flatten(split_locs))))
        else:
            pass
        
    return all_locs




def run_multiple(settings):
    # bivariate_list = [corr_map, entropy_map, granger_map, gpdc_map, cmiknn_map, granger_map]
    bivariate_list = [parcorr_map]
    # noise_list = np.arange(0, 0.05, 0.005)
    noise_list = np.arange(0.01, 0.05, 0.05)
    modes_list = np.arange(4,13,1)

    iterations = 10


    # pcmci_df = pd.DataFrame(columns=modes_list)
    # bivariate_df = pd.DataFrame(columns=modes_list)
    # pcmci_df = np.zeros((iterations, len(modes_list)))
    # bivariate_df = np.zeros((iterations, len(modes_list)))

    for bivariate in bivariate_list:
        pcmci_df = np.zeros((iterations, len(modes_list)))
        bivariate_df = np.zeros((iterations, len(modes_list)))
        for noise in noise_list:
            for mode_i, N in enumerate(modes_list):
                for iteration in range(iterations):
                    settings['N'] = N
                    settings['ny'] = N * 30
                    print('')
                    print(N)
                    print(bivariate..__name__)
                    print(f'Iteration {iteration + 1}/{iterations}')
                    settings['noise_level'] = 0.5
                    cts.create_time_series(settings, links_coeffs, verbose=False,
                                                            plot_modes=False,
                                                            plot_timeseries=False,
                                                            draw_network=False)

                    local_base_path = user_dir
                    local_script_dir = os.path.join(local_base_path, "ERA5" )
                    
                    output = settings['filename']

                    # bivariate = corr_map

                    list_of_name_path = [#('test_target', local_base_path + '/Code_Lennart/NC/test.npy'),
                                        (1, local_base_path + f'/Code_Lennart/results/{output}/NC/{output}_target.nc'),
                                        ('test_precur', local_base_path + f'/Code_Lennart/results/{output}/NC/{output}.nc')
                    ]

                    list_for_MI   = [BivariateMI_PCMCI(name='test_precur', func=bivariate, kwrgs_func={'alpha':.05, 'FDR_control':True}, distance_eps=250)]

                    start_end_TVdate = None
                    start_end_date = None

                    RGCPD_path = local_base_path + f'/Code_Lennart/results/{output}/output_RGCPD/{bivariate.__name__}'
                    shutil.rmtree(RGCPD_path, ignore_errors=True)
                    os.makedirs(RGCPD_path)
                    rg = RGCPD(list_of_name_path=list_of_name_path, 
                            #    list_for_EOFS=list_for_EOFS,
                            list_for_MI=list_for_MI,
                            start_end_TVdate=start_end_TVdate,
                            start_end_date=start_end_date,
                            tfreq=10, lags_i=np.array([1]),
                            verbosity=0,
                            path_outmain=RGCPD_path)

                    selbox = None

                    anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False, 'st2':False}]

                    rg.pp_precursors(selbox=selbox, anomaly=anomaly)

                    rg.pp_TV()

                    #kwrgs_events={'event_percentile':66}
                    kwrgs_events=None
                    rg.traintest(method='random10', kwrgs_events=kwrgs_events)

                    rg.calc_corr_maps()

                    rg.cluster_list_MI()

                    # rg.quick_view_labels()

                    try:
                        locs = rename_labels(rg)

                        rg.get_ts_prec(precur_aggr=None)
                    except:
                        iteration -= 1
                        print("PASSED BECAUSE NO AREAS FOUND")
                        continue


                    rg.PCMCI_df_data(pc_alpha=None, 
                                    tau_max=2,
                                    max_combinations=2)

                    # rg.PCMCI_get_links(alpha_level=0.1)

                    # rg.PCMCI_plot_graph(s=1)

                    # rg.quick_view_labels()

                    # rg.plot_maps_sum()

                    p_matrices = np.array([rg.pcmci_results_dict[i]['p_matrix'] for i in rg.pcmci_results_dict])
                    area_lengths = [len(i) for i in p_matrices]
                    common_length = max(set(area_lengths), key = area_lengths.count)
                    most_common_p_matrix = np.where(np.array(area_lengths) == common_length)
                    p_matrices = p_matrices[most_common_p_matrix]
                    p_matrix = np.mean(p_matrices, axis=0)
                    # p_matrix = p_matrix.mean(0)
                    val_matrix = np.array([rg.pcmci_results_dict[i]['val_matrix'] for i in rg.pcmci_results_dict])[most_common_p_matrix]
                    val_matrix = np.mean(val_matrix, axis=0)
                    pcmci_matrix_path = local_base_path + f'/Code_Lennart/results/{output}' + f'/matrices/{bivariate.__name__}'
                    # settings = {'N': len(rg.pcmci_results_dict[0])}
                    locs = list(np.array(locs)[0][most_common_p_matrix])
                    locs = list(set(flatten(locs)))
                    locs = [0] + locs
                    print(locs)
                    print(common_length)
                    if len(locs) > common_length:
                        iteration -= 1
                        print("PASSED BECAUSE OF UNEQUAL FOUND AREAS")
                        continue
                    cts.save_matrices(settings, pcmci_matrix_path, p_matrix, val_matrix, iteratelist=locs)
                    np.save(pcmci_matrix_path + '/ZZZ_correlated', locs)

                    score = causal_score.calculate_causal_score(settings, val=False, verbose=False, locs=locs)
                    # print(score)

                    path = local_base_path + f'/Code_Lennart/results/{output}/scores'
                    if os.path.isdir(path) != True : os.makedirs(path)
                    for i, key in enumerate(score.columns):
                        key = key.split(' ', 1)[0]
                        if key == 'pcmci_test':
                            # pcmci_df = pcmci_df.append({N: score.values[0][i]}, ignore_index=True)
                            pcmci_df[iteration][mode_i] = score.values[0][i]
                        elif key != 'real':
                            # bivariate_df = bivariate_df.append({N: score.values[0][i]}, ignore_index=True)
                            bivariate_df[iteration][mode_i] = score.values[0][i]
                            print('\n\nScore:')
                            print(score.values[0][i])
        path = local_base_path + f'/Code_Lennart/results/scores/{output}'
        if os.path.isdir(path) != True : os.makedirs(path)
        bivariate_df = pd.DataFrame(bivariate_df, columns=modes_list)
        bivariate_df.to_csv(path + f'/{bivariate.__name__}.csv', index=False)
    pcmci_df = pd.DataFrame(pcmci_df, columns=modes_list)
    pcmci_df.to_csv(path + '/pcmci.csv', index=False)
    

    return





settings = {}
settings['N'] = 5
settings['nx'], settings['ny'], settings['T'] = 30, settings['N'] * 30, 5114
settings['spatial_covariance'] = 0.3
settings['random_modes'] = False
settings['noise_use_mean'] = True
settings['noise_level'] = 0
settings['transient'] = 200
settings['spatial_factor'] = 0.1

settings['user_dir'] = user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'
settings['extra_dir'] = 'Code_Lennart'
settings['filename'] = 'multiple'


settings['random_causal_map'] = True
settings['area_size'] = None


## If any of the following settings is set to True, the results folder with {filename} will be removed!
## Also when 'plot_points' is not None
settings['netcdf4'] = True
settings['save_time_series'] = True
settings['do_pcmci'] = True
settings['save_matrices'] = True
settings['plot_points'] = 500
links_coeffs = 'model3'

settings['alpha'] = 0.01
settings['measure'] = 'average'
settings['val_measure'] = 'average'

run_multiple(settings)