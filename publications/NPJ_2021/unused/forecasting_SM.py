#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 09:59:06 2020

@author: semvijverberg
"""


import os, inspect, sys
if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
import numpy as np
from time import time
import cartopy.crs as ccrs ; import matplotlib.pyplot as plt
import matplotlib
from sklearn import metrics
import pandas as pd
import xarray as xr
# import sklearn.linear_model as scikitlinear
import argparse

user_dir = os.path.expanduser('~')
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')
data_dir = os.path.join(main_dir,'publications/paper2/data')
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

path_raw = user_dir + '/surfdrive/ERA5/input_raw'


from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI
import func_models as fc_utils
import functions_pp; import df_ana
import plot_maps; import core_pp

# Optionally set font to Computer Modern to avoid common missing font errors
# matplotlib.rc('font', family='serif', serif='cm10')

# matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble'] = [r'\boldmath']

region = 'eastern'
precursors = ['sst', 'z500']


expers = np.array(['fixed_corr', 'adapt_corr'])
seeds = np.array([1,2,3])
combinations = np.array(np.meshgrid(precursors, expers, seeds)).T.reshape(-1,3)

i_default = 2



def parseArguments():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Optional arguments
    parser.add_argument("-i", "--intexper", help="intexper", type=int,
                        default=i_default)
    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parseArguments()
    out = combinations[args.intexper]
    precurname = out[0]
    experiment = out[1]
    seed = int(out[2])
    if precurname =='sst':
        tfreq = 60
    elif precurname =='z500':
        tfreq = 15
    print(f'arg {args.intexper} - Target {precurname}, Experiment {experiment}, tfreq {tfreq}')
else:
    precurname = precursors[0]


    tfreq = 15
    experiment = 'fixed_corr'
    experiment = 'adapt_corr'
    seed = 1


calc_ts='region mean' # pattern cov



alpha_corr=.05
cluster_label = 'SM'
name_ds = 'SM'
TVpath = os.path.join(data_dir, 'SM2_09-11-20_14hr.h5')


precur_aggr = tfreq
method     = 'ran_strat10' ;
n_boot = 5000

append_main = ''

if precurname == 'sst':
    precursor = ('sst', os.path.join(path_raw, 'sst_1979-2018_1_12_daily_1.0deg.nc'))
elif precurname == 'z500':
    precursor = ('z500', os.path.join(path_raw, 'z500hpa_1979-2018_1_12_daily_2.5deg.nc'))

#%% run RGPD
start_end_TVdate = ('06-01', '08-31')
start_end_date = ('1-1', '12-31')
list_of_name_path = [(cluster_label, TVpath),
                     precursor]

list_for_MI   = [BivariateMI(name=precurname, func=class_BivariateMI.parcorr_map_time,
                            alpha=alpha_corr, FDR_control=True,
                            kwrgs_func={'precursor':True},
                            distance_eps=1200, min_area_in_degrees2=10,
                            calc_ts=calc_ts, selbox=(0,360,-10,90),
                            lags=np.array([0]))]
if calc_ts == 'region mean':
    s = ''
else:
    s = '_' + calc_ts.replace(' ', '')

path_out_main = os.path.join(main_dir, f'publications/paper2/output/{precurname}{s}{append_main}/')

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           list_import_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           start_end_year=None,
           tfreq=tfreq,
           path_outmain=path_out_main,
           append_pathsub='_' + experiment)



if precurname == 'sst':
    title = r'$parcorr(SST_t, SM_t\ |\ SST_{t-1},SM_{t-1})$'
elif precurname == 'z500':
    title = r'$parcorr(Z500_t, SM_t\ |\ Z500_{t-1},SM_{t-1})$'

subtitles = np.array([['']]) #, f'lag 2 (15 day lead)']] )
kwrgs_plotcorr = {'row_dim':'split', 'col_dim':'lag',
              'units':'Corr. Coeff. [-]', #zoomregion':(130,320,-10,60),
              'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
              'subtitles':subtitles, 'title':title,
              'title_fontdict':{'fontsize':16, 'fontweight':'bold'}}

if precurname == 'sst':
    kwrgs_plotcorr.update({'y_ticks':np.arange(-10,61,20),
                           'x_ticks':np.arange(130, 280, 25),
                           'aspect':2, 'hspace':-.47,
                           'wspace':-.15, 'size':3, 'cbar_vert':-.1,})
elif precurname == 'z500':
    kwrgs_plotcorr.update({'aspect':3.8, 'size':2.5,
                           'hspace':0.0, 'cbar_vert':-.08,
                           'n_yticks':6})



#%%
if experiment == 'fixed_corr':
    rg.pp_TV(name_ds=name_ds, detrend=False)

    subfoldername = '_'.join([precurname,rg.hash, experiment.split('_')[0],
                          str(precur_aggr), str(alpha_corr), method,
                          str(seed)])

    rg.pp_precursors()
    rg.traintest(method=method, seed=seed, subfoldername=subfoldername)
    rg.calc_corr_maps()
    rg.cluster_list_MI()
    rg.quick_view_labels(save=True, append_str=experiment)
    # plotting corr_map
    rg.plot_maps_corr(save=True,
                      kwrgs_plot=kwrgs_plotcorr,
                      min_detect_gc=1.0,
                      append_str=experiment)


# rg.get_ts_prec()
#%% (Adaptive) forecasting
months = {'April-May'    : ('04-01', '05-31'),
          'May-June'    : ('05-01', '06-30'),
          'June-July'   : ('06-01', '07-30'),
           'July-Aug'    : ('07-01', '08-31'),
           'Aug-Sept'    : ('08-01', '09-30'),
           'Sept-Okt'    : ('09-01', '10-31')}

if precur_aggr == 60:
    months = {'Feb-May'  : ('02-01', '05-31'),
              'March-June'  : ('03-01', '06-30'),
              'April-July'  : ('04-01', '07-30'),
              'May-Aug'    : ('05-01', '08-31'),
              'June-Sept'    : ('06-01', '09-30'),
              'July-Okt'    : ('07-01', '10-31')}

    # months = {'March-June'  : ('03-01', '06-30'),
    #           'April-July'  : ('04-01', '07-30'),
    #           'May-Aug'    : ('05-01', '08-31'),
    #           'June-Sept'    : ('06-01', '09-30'),
    #           'July-Okt'    : ('07-01', '10-31')}

monthkeys= list(months.keys()) ; oneyrsize = 0


if precur_aggr == 15:
    blocksize=2
    lag = 1
elif precur_aggr==60:
    blocksize=1
    lag = 1

score_func_list = [metrics.mean_squared_error, fc_utils.corrcoef]
list_test = []
list_test_b = []
no_info_fc = []
dm = {} # dictionairy months
for month, start_end_TVdate in months.items():
    # month, start_end_TVdate = list(months.items())[0]
    if experiment == 'fixed_corr':
        # overwrite RV_mask
        rg.get_ts_prec(precur_aggr=precur_aggr,
                       start_end_TVdate=start_end_TVdate)
    elif experiment == 'adapt_corr':
        rg.start_end_TVdate = start_end_TVdate # adapt target period
        rg.pp_TV(name_ds=name_ds, detrend=False)
        subfoldername = '_'.join([precurname,rg.hash, experiment.split('_')[0],
                          str(precur_aggr), str(alpha_corr), method,
                          str(seed)])
        rg.pp_precursors()
        rg.traintest(method=method, seed=seed, subfoldername=subfoldername)
        finalfolder = rg.path_outsub1.split('/')[-1]

        rg.calc_corr_maps()
        rg.cluster_list_MI()
        rg.quick_view_labels(save=True, append_str=experiment+'+'+month)
        rg.get_ts_prec(precur_aggr=precur_aggr)

        # plotting corr_map
        rg.plot_maps_corr(save=True,
                          kwrgs_plot=kwrgs_plotcorr,
                          min_detect_gc=1.0,
                          append_str=experiment+'_'+month)
        dm[month] = rg.list_for_MI[0].corr_xr.copy()

    alphas = np.append(np.logspace(.1, 1.5, num=25), [250])
    kwrgs_model = {'scoring':'neg_mean_squared_error',
                   'alphas':alphas, # large a, strong regul.
                   'normalize':False}

    keys = [k for k in rg.df_data.columns[:-2] if k != rg.TV.name]
    if len(keys) != 0:
        oneyr = functions_pp.get_oneyr(rg.df_data['RV_mask'].loc[0][rg.df_data['RV_mask'].loc[0]])
        oneyrsize = oneyr.size
        if monthkeys.index(month) >= 1:
            nextyr = functions_pp.get_oneyr(rg.df_data['RV_mask'].loc[0][rg.df_data['RV_mask'].loc[0]])
            if nextyr.size != oneyrsize:
                raise ValueError

        fc_mask = rg.df_data.iloc[:,-1].loc[0]#.shift(lag, fill_value=False)
        # rg.df_data = rg._replace_RV_mask(rg.df_data, replace_RV_mask=(fc_mask))
        target_ts = rg.df_data.iloc[:,[0]].loc[0][fc_mask]
        target_ts = (target_ts - target_ts.mean()) / target_ts.std()

        # ScikitModel = scikitlinear.LassoCV

        out = rg.fit_df_data_ridge(target=target_ts,
                                   keys=keys,
                                   tau_min=lag, tau_max=lag,
                                   kwrgs_model=kwrgs_model,
                                   transformer=fc_utils.standardize_on_train)

        predict, weights, models_lags = out
        prediction = predict.rename({predict.columns[0]:'target',lag:'Prediction'},
                                    axis=1)

        if monthkeys.index(month)==0:
            weights_norm = weights.mean(axis=0, level=1)
            weights_norm.div(weights_norm.max(axis=0)).T.plot(kind='box')

        df_train_m, df_test_s_m, df_test_m, df_boot = fc_utils.get_scores(prediction,
                                                                 rg.df_data.iloc[:,-2:],
                                                                 score_func_list,
                                                                 n_boot = n_boot,
                                                                 blocksize=blocksize,
                                                                 rng_seed=1)
        # Benchmark prediction
        n_splits = rg.df_data.index.levels[0].size
        observed = pd.concat(n_splits*[target_ts], keys=range(n_splits))
        benchpred = observed.copy()
        benchpred[:] = np.zeros_like(observed) # fake pred
        benchpred = pd.concat([observed, benchpred], axis=1)

        bench_MSE = fc_utils.get_scores(benchpred,
                                       rg.df_data.iloc[:,-2:],
                                       [metrics.mean_squared_error],
                                       n_boot = 0,
                                       blocksize=blocksize,
                                       rng_seed=1)[2]
        bench_MSE = float(bench_MSE.values)


        print(df_test_m)
        df_boot['mean_squared_error'] = (bench_MSE-df_boot['mean_squared_error'])/ \
                                                bench_MSE

        df_test_m['mean_squared_error'] = (bench_MSE-df_test_m['mean_squared_error'])/ \
                                                bench_MSE

        cvfitalpha = [models_lags[f'lag_{lag}'][f'split_{s}'].alpha_ for s in range(n_splits)]
        print('mean alpha {:.0f}'.format(np.mean(cvfitalpha)))
        maxalpha_c = list(cvfitalpha).count(alphas[-1])
        if maxalpha_c > n_splits/3:
            print(f'\n{month} alpha {int(np.mean(cvfitalpha))}')
            print(f'{maxalpha_c} splits are max alpha\n')
            # maximum regularization selected. No information in timeseries
            df_test_m['corrcoef'][:] = 0
            df_boot['corrcoef'][:] = 0
            no_info_fc.append(month)
        df_test = functions_pp.get_df_test(prediction.merge(rg.df_data.iloc[:,-2:],
                                                        left_index=True,
                                                        right_index=True)).iloc[:,:2]

    else:
        print('no precursor timeseries found, scores all 0')
        df_boot = pd.DataFrame(data=np.zeros((n_boot, len(score_func_list))),
                            columns=['mean_squared_error', 'corrcoef'])
        df_test_m = pd.DataFrame(np.zeros((1,len(score_func_list))),
                                 columns=['mean_squared_error', 'corrcoef'])


    list_test_b.append(df_boot)
    list_test.append(df_test_m)
    # df_ana.loop_df(df=rg.df_data[keys], colwrap=1, sharex=False,
    #                       function=df_ana.plot_timeseries,
    #                       kwrgs={'timesteps':rg.fullts.size,
    #                                   'nth_xyear':5})


import matplotlib.patches as mpatches
corrvals = [test.values[0,1] for test in list_test]
MSE_SS_vals = [test.values[0,0] for test in list_test]
df_scores = pd.DataFrame({'RMSE-SS':MSE_SS_vals, 'Corr. Coef.':corrvals},
                         index=monthkeys)
df_test_b = pd.concat(list_test_b, keys = monthkeys,axis=1)

yerr = [] ; quan = [] ; alpha = .10
for i in range(df_test_b.columns.size):
    Eh = 1 - alpha/2 ; El = alpha/2
    tup = [df_test_b.iloc[:,i].quantile(El), df_test_b.iloc[:,i].quantile(Eh)]
    quan.append(tup)
    mean = df_scores.values.flatten()[i]
    tup = abs(mean-tup)
    yerr.append(tup)
_yerr = np.array(yerr).T.reshape(len(monthkeys)*2,2, order='A')
_yerr = np.array(yerr).reshape(2,len(monthkeys)*2,
                               order='F').reshape(2,2,len(monthkeys))
ax = df_scores.plot.bar(rot=0, yerr=_yerr,
                        capsize=8, error_kw=dict(capthick=1),
                        color=['blue', 'green'],
                        legend=False)
for noinfo in no_info_fc:
    # first two children are not barplots
    idx = monthkeys.index(noinfo) + 2
    ax.get_children()[idx].set_color('r') # RMSE bar

ax.set_ylabel('Skill Score', fontsize=16)
# ax.set_xlabel('Months', fontsize=16)

title = f'Seasonal dependence of SM-{precurname} [{precur_aggr}-day mean]'
ax.set_title(title,
             fontsize=16)
ax.tick_params(axis='both', labelsize=13)

if tfreq==15 and experiment=='adapt_corr':
    patch1 = mpatches.Patch(color='blue', label='RMSE-SS')
    patch2 = mpatches.Patch(color='green', label='Corr. Coef.')
    handles = [patch1, patch2]
    # manually define a new patch
    if len(no_info_fc) != 0:
        patch = mpatches.Patch(color='red', label=f'alpha={int(alphas[-1])}')
        handles.append(patch)
    ax.legend(handles=handles,
              fontsize=16, frameon=True, facecolor='grey',
              framealpha=.5)
ax.set_ylim(-0.5, 1)
plt.savefig(os.path.join(rg.path_outsub1,
             f'skill_score_vs_months_{precur_aggr}tf_lag{lag}_nb{n_boot}_blsz{blocksize}_{alpha_corr}.pdf'))
#%%
if experiment == 'adapt_corr':




    corr = dm[monthkeys[0]].mean(dim='split').drop('time')
    list_xr = [corr.expand_dims('months', axis=0) for i in range(len(monthkeys))]
    corr = xr.concat(list_xr, dim = 'months')
    corr['months'] = ('months', monthkeys)

    np_data = np.zeros_like(corr.values)
    np_mask = np.zeros_like(corr.values)
    for i, f in enumerate(monthkeys):
        corr_xr = dm[f]
        vals = corr_xr.mean(dim='split').values
        np_data[i] = vals
        mask = corr_xr.mask.mean(dim='split')
        np_mask[i] = mask

    corr.values = np_data
    mask = (('months', 'lag', 'latitude', 'longitude'), np_mask )
    corr.coords['mask'] = mask

    precur = rg.list_for_MI[0]
    f_name = 'corr_{}_a{}'.format(precur.name,
                                precur.alpha) + '_' + \
                                f'{experiment}_lag{lag}_' + \
                                f'tf{precur_aggr}_{method}'

    corr.to_netcdf(os.path.join(rg.path_outsub1, f_name+'.nc'), mode='w')
    import_ds = core_pp.import_ds_lazy
    corr = import_ds(os.path.join(rg.path_outsub1, f_name+'.nc'))[precur.name]
    subtitles = np.array([monthkeys])
    kwrgs_plot = {'aspect':2, 'hspace':.3,
                  'wspace':-.4, 'size':1.25, 'cbar_vert':-0.1,
                  'units':'Corr. Coeff. [-]',
                  'clim':(-.60,.60), 'map_proj':ccrs.PlateCarree(central_longitude=220),
                  'y_ticks':False,
                  'x_ticks':False,
                  'subtitles':subtitles}

    fig = plot_maps.plot_corr_maps(corr, mask_xr=corr.mask, col_dim='months',
                                   row_dim=corr.dims[1],
                                   **kwrgs_plot)

    fig_path = os.path.join(rg.path_outsub1, f_name)+rg.figext
#%%
    plt.savefig(fig_path, bbox_inches='tight')


#%%

# df = df_test_b.stack().reset_index(level=1)
# dfx = df.groupby(['level_1'])
# axes = dfx.boxplot()
# axes[0].set_ylim(-0.5, 1)
#%%
# import seaborn as sns
# df_ap = pd.concat(list_test_b, axis=0, ignore_index=True)
# df_ap['months'] = np.repeat(monthkeys, list_test_b[0].index.size)
# # df_ap.boxplot(by='months')
# ax = sns.boxplot(x=df_ap['months'], y=df_ap['mean_squared_error'])
# ax.set_ylim(-0.5, 1)
# plt.figure()
# ax = sns.boxplot(x=df_ap['months'], y=df_ap['corrcoef'])
# ax.set_ylim(-0.5, 1)

# #%%
# columns_my_order = monthkeys
# fig, ax = plt.subplots()
# for position, column in enumerate(columns_my_order):
#     ax.boxplot(df_test_b.loc[column], positions=[position,position+.25])

# ax.set_xticks(range(position+1))
# ax.set_xticklabels(columns_my_order)
# ax.set_xlim(xmin=-0.5)
# plt.show()


#%%

