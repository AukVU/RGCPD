import os, inspect, sys

if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/') 
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)

df_ana_dir2 = os.path.join(curr_dir, 'df_analysis/') # add df_ana path
sys.path.append(df_ana_dir2)

import creating_time_series as cts
import causal_score

import shutil

from RGCPD import RGCPD
from find_precursors import relabel
from class_BivariateMI_PCMCI import BivariateMI_PCMCI

from class_BivariateMI_PCMCI import corr_map
from class_BivariateMI_PCMCI import entropy_map
from class_BivariateMI_PCMCI import parcorr_map_spatial
from class_BivariateMI_PCMCI import parcorr_map_time
from class_BivariateMI_PCMCI import granger_map
from class_BivariateMI_PCMCI import gpdc_map
from class_BivariateMI_PCMCI import rcot_map
from class_BivariateMI_PCMCI import cmiknn_map

import numpy as np
import pandas as pd
from datetime import date

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
                for loc in range(N_areas):
                    area = labels[middle - offset: middle + offset, rows * loc + middle - offset: rows * loc + middle + offset]
                    area_nonzero = np.nonzero(area)
                    if len(area_nonzero[0]) > 0:
                        locs.append(loc+1)
                        value = area[area_nonzero[0][0]][area_nonzero[1][0]]
                        reassign[value] = loc+1
                locs = list(reassign.values())
                relabeld = relabel(precur.prec_labels.values[split], reassign).astype('float')
                relabeld[relabeld == 0] = np.nan
                precur.prec_labels.values[split] = relabeld
                split_locs.append(locs)
            all_locs.append(split_locs)
            # all_locs.append(list(set(flatten(split_locs))))
        else:
            pass
        
    return all_locs


def filter_matrices(matrices, locs, locs_intersect=None):
    if locs_intersect == None:
        locs_intersect = list(set.intersection(*map(set, locs)))
    else:
        locs_intersect = locs_intersect[1:]
    filtered_matrices = np.zeros((len(matrices), len(locs_intersect) + 1, len(locs_intersect) + 1, len(matrices[0][0][0])))
    for i, loc in enumerate(locs):
        indices = list(np.where(np.isin(loc, locs_intersect))[0])
        indices = [0] + [i+1 for i in indices]
        filtered_matrices[i] = matrices[i][indices][:, indices]
    return filtered_matrices, ([0] + locs_intersect)

     





def run_DBSCAN_test(settings, cluster=None, iterations=10):
    N = settings['N']
    # bivariate_list = [corr_map, entropy_map, granger_map, gpdc_map, cmiknn_map, granger_map]
    bivariate_list = [corr_map]
    bivariate = corr_map
    bivariate_kwrgs_list = [(0,0,0)]
    table_list = None
    test = None

    test_splits = 5

    DBSCAN_list = list(np.array(cluster).flat)
    if cluster == None:
        DBSCAN_list = np.arange(100, 410, 50)
        table_list = DBSCAN_list
        test = 'DBSCAN epsilon' 
    print(f"\nTested cluster epsilons: {DBSCAN_list}")


    


    # pcmci_df = pd.DataFrame(columns=modes_list)
    # bivariate_df = pd.DataFrame(columns=modes_list)
    # pcmci_df = np.zeros((iterations, len(modes_list)))
    # bivariate_df = np.zeros((iterations, len(modes_list)))

    # for biv_i, bivariate in enumerate(bivariate_list):
    pcmci_df = np.zeros((iterations, len(table_list)))
    print(table_list)
    bivariate_df = np.zeros((len(bivariate_list), iterations, len(table_list)))
    print(bivariate_df)
    for cluster_i, cluster_eps in enumerate(DBSCAN_list):
        for iteration in range(iterations):
            settings['N'] = N
            settings['ny'] = N * 30
            settings['model'] = 'multiple'
            settings['autocor'] = 0.9
            settings['timefreq'] = 1
            print(f'Iteration {iteration + 1}/{iterations}')
            print('')
            print(f"cluster_eps: {cluster_eps}")
            settings['noise_level'] = 10 #0.5
            # settings['signal'] = 0.6
            settings['T'] = 1826
            cts.create_time_series(settings, links_coeffs, verbose=False,
                                                    plot_modes=False,
                                                    plot_timeseries=False,
                                                    draw_network=False,
                                                    cluster=True)

            print("Finished generating!")
            local_base_path = user_dir
            local_script_dir = os.path.join(local_base_path, "ERA5" )
            
            output = settings['filename']

            # bivariate = corr_map
            print(bivariate.__name__)
            list_of_name_path = [#('test_target', local_base_path + '/Code_Lennart/NC/test.npy'),
                                (1, local_base_path + f'/{output}/NC/{output}_target.nc'),
                                ('test_precur', local_base_path + f'/{output}/NC/{output}.nc')
            ]

            kwrgs_bivariate = {}
            if bivariate == parcorr_map_time:
                lag = bivariate_kwrgs_list[biv_i][0]
                target = bivariate_kwrgs_list[biv_i][1]
                precur = bivariate_kwrgs_list[biv_i][2]
                kwrgs_bivariate = {'lag':lag, 'target':target, 'precur':precur}

            print(kwrgs_bivariate)

            list_for_MI   = [BivariateMI_PCMCI(name='test_precur', func=bivariate, kwrgs_func={'alpha':.1, 'FDR_control':False}, distance_eps=cluster_eps, min_area_in_degrees2=3, kwrgs_bivariate=kwrgs_bivariate)]

            start_end_TVdate = ('7-1', '12-31')
            start_end_date = ('1-1', '12-31')

            RGCPD_path = local_base_path + f'/{output}/output_RGCPD/{bivariate.__name__}'
            shutil.rmtree(RGCPD_path, ignore_errors=True)
            os.makedirs(RGCPD_path)
            print(RGCPD_path)
            rg = RGCPD(list_of_name_path=list_of_name_path, 
                    #    list_for_EOFS=list_for_EOFS,
                    list_for_MI=list_for_MI,
                    start_end_TVdate=start_end_TVdate,
                    start_end_date=start_end_date,
                    tfreq=settings['timefreq'], lags_i=np.array([1]),
                    verbosity=0,
                    path_outmain=RGCPD_path)

            selbox = None

            anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False, 'st2':False}]

            rg.pp_precursors(selbox=selbox, anomaly=False, detrend=False)

            rg.pp_TV()

            #kwrgs_events={'event_percentile':66}
            kwrgs_events=None
            rg.traintest(method=f'random{test_splits}', kwrgs_events=kwrgs_events)

            rg.calc_corr_maps()

            # rg.plot_maps_corr(save=True)

            rg.cluster_list_MI()

            # rg.quick_view_labels()

            try:
                locs = rename_labels(rg)

                rg.get_ts_prec(precur_aggr=None)
            except:
                # iteration -= 1
                print("PASSED BECAUSE NO AREAS FOUND")
                continue


            rg.PCMCI_df_data(pc_alpha=None, 
                            tau_max=2,
                            max_combinations=2)

            # rg.PCMCI_get_links(alpha_level=0.1)

            # rg.PCMCI_plot_graph(s=1)

            # rg.quick_view_labels()

            # rg.plot_maps_sum()

            # p_matrices = np.array([rg.pcmci_results_dict[i]['p_matrix'] for i in rg.pcmci_results_dict])
            # area_lengths = [len(i) for i in p_matrices]
            # common_length = max(set(area_lengths), key = area_lengths.count)
            # most_common_p_matrix = np.where(np.array(area_lengths) == common_length)
            # p_matrices = p_matrices[most_common_p_matrix]
            # p_matrix = np.mean(p_matrices, axis=0)
            # # p_matrix = p_matrix.mean(0)
            # val_matrix = np.array([rg.pcmci_results_dict[i]['val_matrix'] for i in rg.pcmci_results_dict])[most_common_p_matrix]
            # val_matrix = np.mean(val_matrix, axis=0)

            # parents = rg.parents_dict[0][0]
            # parents = [i[0] for i in parents if i[1] == -1]




            pcmci_matrix_path = local_base_path + f'/{output}' + f'/matrices/{bivariate.__name__}'
            if bivariate.__name__ == 'parcorr_map_time':
                pcmci_matrix_path = pcmci_matrix_path + f'-{lag}-{target}-{precur}'
            # settings = {'N': len(rg.pcmci_results_dict[0])}
            if len(locs) == 0:
                locs = [[] for _ in range(test_splits)]
            else:
                locs = list(np.array(locs)[0])#[most_common_p_matrix])  #[0]
            p_matrices = np.array([rg.pcmci_results_dict[i]['p_matrix'] for i in rg.pcmci_results_dict])
            area_lengths = [len(i) for i in p_matrices]
            common_length = max(set(area_lengths), key = area_lengths.count)
            p_matrices, locs_filtered = filter_matrices(p_matrices, locs)
            val_matrices = np.array([rg.pcmci_results_dict[i]['val_matrix'] for i in rg.pcmci_results_dict])
            val_matrices, locs = filter_matrices(val_matrices, locs, locs_intersect=locs_filtered)

            p_matrix = np.mean(p_matrices, axis=0)
            val_matrix = np.mean(val_matrices, axis=0)


            # locs = list(set(flatten(locs)))
            # locs = [0] + locs_filtered
            print(f'\n\nFound regions {locs}')
            # print(f'Found parents for split 0: {list(np.array(locs)[parents])}\n')
            # print(common_length)
            if len(locs) > common_length:
                # iteration -= 1
                print("PASSED BECAUSE OF UNEQUAL FOUND AREAS")
                continue
            cts.save_matrices(settings, pcmci_matrix_path, p_matrix, val_matrix, iteratelist=locs)
            np.save(pcmci_matrix_path + '/ZZZ_correlated', locs)

            score = causal_score.calculate_causal_score(settings, val=False, verbose=False, locs=locs, target_only=True, no_ac=False)
            # print(score)

            if cluster == None:
                table_i = cluster_i
            elif modes == None:
                table_i = mode_i
            elif years == None:
                table_i = year_i
            elif noises == None:
                table_i = noise_i
            print(f'Table_i: {table_i}')
            # path = local_base_path + f'/{output}/scores'
            # if os.path.isdir(path) != True : os.makedirs(path)
            test_list = [bivariate.__name__ for bivariate in bivariate_list]
            for i, key in enumerate(score.columns):
                key = key.split(' ', 1)[0]
                if key[:7] == 'parcorr':
                    key2, lag, target, precur = key.split('-')
                    target = (target == 'True')
                    precur = (precur == 'True')
                    key = key2
                if key == 'pcmci_test':
                    # pcmci_df = pcmci_df.append({N: score.values[0][i]}, ignore_index=True)
                    pcmci_df[iteration][table_i] = score.values[0][i]
                    print("PCMCI score:")
                    print(score.values[0][i])
                elif key in test_list:
                    test_list_index = test_list.index(key)
                    if key[:7] == 'parcorr':
                        test_list_index = bivariate_kwrgs_list.index((int(lag),target,precur))
                    # bivariate_df = bivariate_df.append({N: score.values[0][i]}, ignore_index=True)
                    bivariate_df[test_list_index][iteration][table_i] = score.values[0][i]
                    if key[:7] == 'parcorr':
                        print(f'Score {key}, lag {lag}, target {target}, precur {precur}:')
                    else:
                        print(f'Score {key}:')
                    print(score.values[0][i])
    path = local_base_path + f'/scores/{output}'
    if os.path.isdir(path) != True : os.makedirs(path)
    for biv_i, df in enumerate(bivariate_df):
        print(df)
        bivariate_df = pd.DataFrame(df, columns=table_list)
        path2 = path + f'/{test}_{bivariate_list[biv_i].__name__}'
        if bivariate_list[biv_i].__name__ == 'parcorr_map_time':
            kwrgs = bivariate_kwrgs_list[biv_i]
            path2 = path2 + f'-{kwrgs[0]}-{kwrgs[1]}-{kwrgs[2]}'
        bivariate_df.to_csv(path2 + '.csv', index=False)
    print("------------------------------------")
    print(pcmci_df)
    pcmci_df = pd.DataFrame(pcmci_df, columns=table_list)
    pcmci_df.to_csv(path + f'/{test}_pcmci.csv', index=False)
    

    return





settings = {}
settings['N'] = 7
settings['nx'], settings['ny'], settings['T'] = 30, settings['N'] * 30, 3287
settings['spatial_covariance'] = 500
settings['random_modes'] = False
settings['noise_use_mean'] = False
settings['noise_level'] = 10
settings['transient'] = 200
settings['spatial_factor'] = 0.1

settings['user_dir'] = user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'
settings['extra_dir'] = 'Code_Lennart'
settings['filename'] = 'multiple_test'

if len(sys.argv) > 1:
    settings['user_dir']  = sys.argv[1]
    user_dir = settings['user_dir']  + '/' + settings['extra_dir'] + '/results'
else:
    settings['user_dir'] = "/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD"
    user_dir = settings['user_dir']  + '/' + settings['extra_dir'] + '/results'
print(f"DIR is: {user_dir}")



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

settings['signal'] = 0.25

run_DBSCAN_test(settings, cluster=None, iterations=30)