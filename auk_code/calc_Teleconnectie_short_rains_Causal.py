# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:23:07 2021

@author: Auk
"""

import os, inspect, sys
# main_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
main_dir ='C:/Users/Auk/Documents/GitHub/RGCPD'
print(main_dir)
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering')
if RGCPD_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    
from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI, functions_pp
from IPython.display import Image
import numpy as np
import find_precursors, plot_maps

# define input: 
path_test = os.path.join(main_dir, 'data') # path of test data
# format list_of_name_path = [('TVname', 'TVpath'), ('prec_name', 'prec_path')]
list_of_name_path = [(18, os.path.join(path_test, 'q85_dendo_1bae1.nc')),
                    ('sst', os.path.join(path_test,'sst_1950-2020_1_12_monthly_1.0deg.nc'))]

list_for_MI = [BivariateMI(name='sst', func=class_BivariateMI.corr_map, 
                           alpha=.01, FDR_control=True, 
                           lags=np.array([1,2,3]),
                           # lags=np.array([['1950-06-01', '1950-08-31'],
                           #                  ['1950-09-01', '1950-11-30'],
                           #                  ['1950-12-01', '1951-02-28']]), # <- selecting time periods to aggregate
                           distance_eps=700, min_area_in_degrees2=5)]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           # tfreq=None, # <- seasonal forecasting mode, set tfreq to None! 
           tfreq=3,
           start_end_TVdate=('10-01', '12-31'),# <- defining DJF target period 
           start_end_date=None,
           start_end_year=None,
           path_outmain=os.path.join(main_dir,'data'))

rg.pp_precursors(detrend=True, anomaly=True, selbox=None)
# start_end_year=(1950,2020)
# TV_start_end_year = (start_end_year[0]+1,2020)

# kwrgs_core_pp_time = {'start_end_year' : TV_start_end_year }

rg.pp_TV(TVdates_aggr=False,ext_annual_to_mon=True)
         # kwrgs_core_pp_time=kwrgs_core_pp_time)

rg.plot_df_clust()

rg.fulltso

rg.traintest(method='random_20')

rg.calc_corr_maps() 
#%%

rg.cluster_list_MI()


sst = rg.list_for_MI[0] 

periodnames = ['JJA','SON','DJF']
sst.prec_labels['lag'] = ('lag', periodnames)
sst.corr_xr['lag'] = ('lag', periodnames)

rg.plot_maps_corr()
rg.quick_view_labels(mean=True)

rg.get_ts_prec()
rg._df_count 

# df_prec_regions = find_precursors.labels_to_df(rg.list_for_MI[0].prec_labels)
# df_prec_regions # center lat,lon coordinates and size (in number of gridcells)

# split = find_precursors.split_region_by_lonlat
# new_labels, label = split(rg.list_for_MI[0].prec_labels, label=1,
#              kwrgs_mask_latlon={'latmax':30}) # <- split region 1 by 30 degree latitude
# rg.list_for_MI[0].prec_labels = new_labels

# rg.get_ts_prec()
# rg.df_data

rg.PCMCI_df_data(tigr_function_call='run_pcmci',
                 kwrgs_tigr={'tau_min': 0,
                             'tau_max': 1,
                             'pc_alpha': 0.05,
                             'max_conds_dim': 2,
                             'max_combinations': 2,
                             'max_conds_py': 2,
                             'max_conds_px': 2})


rg.PCMCI_get_links(var=rg.TV.name, alpha_level=.05)
rg.df_links
rg.plot_maps_sum()

q =.85 
import func_models as fc_utils
from stat_models_cont import ScikitModel
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegressionCV


# choose type prediciton (continuous or probabilistic) by making comment #
prediction = 'continuous'   
# prediction = 'events' ; q = .66 # quantile threshold for event definition

if prediction == 'continuous':
    model = ScikitModel(Ridge, verbosity=0)
    # You can also tune parameters by passing a list of values. Then GridSearchCV from sklearn will 
    # find the set of parameters that give the best mean score on all kfold test sets. 
    # below we pass a list of alpha's to tune the regularization.
    alphas = list(np.concatenate([[1E-20],np.logspace(-5,0, 6), np.logspace(.01, 2.5, num=25)]))                       
    kwrgs_model = {'scoringCV':'neg_mean_absolute_error',
                   'kfold':10,
                   'alpha':alphas} # large a, strong regul.
elif prediction == 'events':
    model = ScikitModel(LogisticRegressionCV, verbosity=0)
    kwrgs_model = {'kfold':10,
                   'scoring':'neg_brier_score'}

    

target_ts = rg.TV.RV_ts ; 
target_ts = (target_ts - target_ts.mean()) / target_ts.std()
if prediction == 'events':
    if q >= 0.5:
        target_ts = (target_ts > target_ts.quantile(q)).astype(int)
    elif q < .5:
        target_ts = (target_ts < target_ts.quantile(q)).astype(int)
    BSS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).BSS
    score_func_list = [BSS, fc_utils.metrics.roc_auc_score]
    
elif prediction == 'continuous':
    RMSE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).RMSE
    MAE_SS = fc_utils.ErrorSkillScore(constant_bench=float(target_ts.mean())).MAE
    score_func_list = [RMSE_SS, fc_utils.corrcoef, MAE_SS]
        
    
out = rg.fit_df_data_ridge(target=target_ts,
                            keys=rg.df_data.columns[1:-2],
                            fcmodel=model,
                            kwrgs_model=kwrgs_model,
                            transformer=None,
                            tau_min=0, tau_max=0) # <- lag should be zero
predict, weights, model_lags = out

df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(predict,
                                                                 rg.df_data.iloc[:,-2:],
                                                                 score_func_list,
                                                                 n_boot = 100,
                                                                 score_per_test=False,
                                                                 blocksize=1,
                                                                 rng_seed=1)


lag = 0
if prediction == 'events':
    print(model.scikitmodel.__name__, '\n', f'Test score at lag {lag}\n',
          'BSS {:.2f}\n'.format(df_test_m.loc[0].loc[lag].loc['BSS']),
          'AUC {:.2f}'.format(df_test_m.loc[0].loc[lag].loc['roc_auc_score']),
          '\nTrain score\n',
          'BSS {:.2f}\n'.format(df_train_m.mean(0).loc[lag]['BSS']),
          'AUC {:.2f}'.format(df_train_m.mean(0).loc[lag]['roc_auc_score']))
elif prediction == 'continuous':
    print(model.scikitmodel.__name__, '\n', 'Test score\n',
              'RMSE {:.2f}\n'.format(df_test_m.loc[0][lag]['RMSE']),
              'MAE {:.2f}\n'.format(df_test_m.loc[0][lag]['MAE']),
              'corrcoef {:.2f}'.format(df_test_m.loc[0][lag]['corrcoef']),
              '\nTrain score\n',
              'RMSE {:.2f}\n'.format(df_train_m.mean(0).loc[lag]['RMSE']),
              'MAE {:.2f}\n'.format(df_train_m.mean(0).loc[lag]['MAE']),
              'corrcoef {:.2f}'.format(df_train_m.mean(0).loc[lag]['corrcoef']))
predict
m = model_lags['lag_0']['split_0']
m # if prediction == 'continuous', this show the GridSearchCV output, else it shows the fitted logistic model.

#if prediction == 'continuous':
#    print(m.cv_results_['mean_test_score'])
    
# df_test_m.loc[0].plot.bar(rot=0, color=['blue', 'green', 'purple'], figsize=(10,4))

from stat_models import plot_importances
df_weights, fig = plot_importances(models_splits_lags=model_lags, lag=0)

