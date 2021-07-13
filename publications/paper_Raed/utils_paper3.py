#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 11:14:02 2021

@author: semvijverberg
"""



import os, sys
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

# import sklearn.linear_model as scikitlinear
from matplotlib import gridspec
from matplotlib.offsetbox import TextArea, VPacker, AnnotationBbox

user_dir = os.path.expanduser('~')
os.chdir(os.path.join(user_dir,
                      'surfdrive/Scripts/RGCPD/publications/paper_Raed/'))
curr_dir = os.path.join(user_dir, 'surfdrive/Scripts/RGCPD/RGCPD/')
main_dir = '/'.join(curr_dir.split('/')[:-2])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
assert main_dir.split('/')[-1] == 'RGCPD', 'main dir is not RGCPD dir'
cluster_func = os.path.join(main_dir, 'clustering/')
fc_dir = os.path.join(main_dir, 'forecasting')

if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(fc_dir)

import func_models as fc_utils
import functions_pp, find_precursors




def get_df_mean_SST(rg, mean_vars=['sst'], alpha_CI=.05,
                    n_strongest='all',
                    weights=True, labels=None,
                    fcmodel=None, kwrgs_model=None,
                    target_ts=None):


    periodnames = list(rg.list_for_MI[0].corr_xr.lag.values)
    df_pvals = rg.df_pvals.copy()
    df_corr  = rg.df_corr.copy()
    unique_keys = np.unique(['..'.join(k.split('..')[1:]) for k in rg.df_data.columns[1:-2]])
    if labels is not None:
        unique_keys = [k for k in unique_keys if k in labels]

    # dict with strongest mean parcorr over growing season
    mean_SST_list = []
    # keys_dict = {s:[] for s in range(rg.n_spl)} ;
    keys_dict_meansst = {s:[] for s in range(rg.n_spl)} ;
    for s in range(rg.n_spl):
        mean_SST_list_s = []
        sign_s = df_pvals[s][df_pvals[s] <= alpha_CI].dropna(axis=0, how='all')
        for uniqk in unique_keys:
            # uniqk = '1..smi'
            # region label (R) for each month in split (s)
            keys_mon = [mon+ '..'+uniqk for mon in periodnames]
            # significant region label (R) for each month in split (s)
            keys_mon_sig = [k for k in keys_mon if k in sign_s.index] # check if sig.
            if uniqk.split('..')[-1] in mean_vars and len(keys_mon_sig)!=0:
                # mean over region if they have same correlation sign across months
                for sign in [1,-1]:
                    mask = np.sign(df_corr.loc[keys_mon_sig][[s]]) == sign
                    k_sign = np.array(keys_mon_sig)[mask.values.flatten()]
                    if len(k_sign)==0:
                        continue
                    # calculate mean over n strongest SST timeseries
                    if len(k_sign) > 1:
                        meanparcorr = df_corr.loc[k_sign][[s]].squeeze().sort_values()
                        if n_strongest == 'all':
                            keys_str = meanparcorr.index
                        else:
                            keys_str = meanparcorr.index[-n_strongest:]
                    else:
                        keys_str  = k_sign
                    if weights:
                        fit_masks = rg.df_data.loc[s].iloc[:,-2:]
                        df_d = rg.df_data.loc[s][keys_str].copy()
                        df_d = df_d.apply(fc_utils.standardize_on_train_and_RV,
                                          args=[fit_masks, 0])
                        df_d = df_d.merge(fit_masks, left_index=True,right_index=True)
                        # df_train = df_d[fit_masks['TrainIsTrue']]
                        df_mean, model = fcmodel.fit_wrapper({'ts':target_ts},
                                                          df_d, keys_str,
                                                          kwrgs_model)


                    else:
                        df_mean = rg.df_data.loc[s][keys_str].copy().mean(1)
                    month_strings = [k.split('..')[0] for k in sorted(keys_str)]
                    df_mean = df_mean.rename({0:''.join(month_strings) + '..'+uniqk},
                                             axis=1)
                    keys_dict_meansst[s].append( df_mean.columns[0] )
                    mean_SST_list_s.append(df_mean)
            elif uniqk.split('..')[-1] not in mean_vars and len(keys_mon_sig)!=0:
                # use all timeseries (for each month)
                mean_SST_list_s.append(rg.df_data.loc[s][keys_mon_sig].copy())
                keys_dict_meansst[s] = keys_dict_meansst[s] + keys_mon_sig
            # elif len(keys_mon_sig) == 0:
            #     data = np.zeros(rg.df_RV_ts.size) ; data[:] = np.nan
            #     pd.DataFrame(data, index=rg.df_RV_ts.index)
        # if len(keys_mon_sig) != 0:
        df_s = pd.concat(mean_SST_list_s, axis=1)
        mean_SST_list.append(df_s)
    df_mean_SST = pd.concat(mean_SST_list, keys=range(rg.n_spl))
    df_mean_SST = df_mean_SST.merge(rg.df_splits.copy(),
                                    left_index=True, right_index=True)
    return df_mean_SST, keys_dict_meansst



#%% Functions for plotting continuous forecast
def df_scores_for_plot(rg_list, name_object):
    df_scores = [] ; df_boot = [] ; df_tests = []
    for i, rg in enumerate(rg_list):
        verification_tuple = rg.__dict__[name_object]
        df_scores.append(verification_tuple[2])
        df_boot.append(verification_tuple[3])
        df_tests.append(verification_tuple[1])
    df_scores = pd.concat(df_scores, axis=1)
    df_boot = pd.concat(df_boot, axis=1)
    df_tests = pd.concat(df_tests, axis=1)
    return df_scores, df_boot, df_tests

def df_predictions_for_plot(rg_list):
    df_preds = []
    for i, rg in enumerate(rg_list):
        rg.df_fulltso.index.name = None
        if i == 0:
            prediction = rg.prediction_tuple[0]
            prediction = rg.merge_df_on_df_data(rg.df_fulltso, prediction)
        else:
            prediction = rg.prediction_tuple[0].iloc[:,[1]]
        df_preds.append(prediction)
        if i+1 == len(rg_list):
            df_preds.append(rg.df_splits)
    df_preds  = pd.concat(df_preds, axis=1)
    return df_preds

def plot_scores_wrapper(df_scores, df_boot, df_scores_cf=None, df_boot_cf=None):
    orientation = 'horizontal'
    alpha = .1
    if 'BSS' in df_scores.columns.levels[1]:
        metrics_cols = ['BSS', 'roc_auc_score']
        rename_m = {'BSS': 'BSS', 'roc_auc_score':'ROC-AUC'}
    else:
        metrics_cols = ['corrcoef', 'MAE', 'RMSE', 'r2_score']
        rename_m = {'corrcoef': 'Corr. coeff.', 'RMSE':'RMSE-SS',
                    'MAE':'MAE-SS', 'CRPSS':'CRPSS', 'r2_score':'$R^2$',
                    'mean_absolute_percentage_error':'MAPE'}

    if orientation=='vertical':
        f, ax = plt.subplots(len(metrics_cols),1, figsize=(6, 5*len(metrics_cols)),
                         sharex=True) ;
    else:
        f, ax = plt.subplots(1,len(metrics_cols), figsize=(6.5*len(metrics_cols), 5),
                         sharey=False) ;

    c1, c2 = '#3388BB', '#EE6666'
    for i, m in enumerate(metrics_cols):
        # normal SST
        # columns.levels auto-sorts order of labels, to avoid:
        steps = df_scores.columns.levels[1].size
        labels = [t[0] for t in df_scores.columns][::steps]
        ax[i].plot(labels, df_scores.reorder_levels((1,0), axis=1).loc[0][m].T,
                label='Verification on all years',
                color=c2,
                linestyle='solid')
        ax[i].fill_between(labels,
                            df_boot.reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
                            df_boot.reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
                            edgecolor=c2, facecolor=c2, alpha=0.3,
                            linestyle='solid', linewidth=2)
        # Conditional SST
        if df_scores_cf is not None:
            labels = df_scores_cf.columns.levels[0]
            ax[i].plot(labels, df_scores_cf.reorder_levels((1,0), axis=1).loc[0][m].T,
                    label='Pronounced Pacific state years',
                    color=c1,
                    linestyle='solid')
            ax[i].fill_between(labels,
                                df_boot_cf.reorder_levels((1,0), axis=1)[m].quantile(1-alpha/2.),
                                df_boot_cf.reorder_levels((1,0), axis=1)[m].quantile(alpha/2.),
                                edgecolor=c1, facecolor=c1, alpha=0.3,
                                linestyle='solid', linewidth=2)

        if m == 'corrcoef':
            ax[i].set_ylim(-.2,1)
        elif m == 'roc_auc_score':
            ax[i].set_ylim(0,1)
        else:
            ax[i].set_ylim(-.2,.6)
        ax[i].axhline(y=0, color='black', linewidth=1)
        ax[i].tick_params(labelsize=16, pad=6)
        if i == len(metrics_cols)-1 and orientation=='vertical':
            ax[i].set_xlabel('Forecast month', fontsize=18)
        elif orientation=='horizontal':
            ax[i].set_xlabel('Forecast month', fontsize=18)
        if i == 0:
            ax[i].legend(loc='lower right', fontsize=14)
        ax[i].set_ylabel(rename_m[m], fontsize=18, labelpad=-4)

    f.subplots_adjust(hspace=.1)
    f.subplots_adjust(wspace=.25)
    title = 'Verification high Yield forecast'
    if orientation == 'vertical':
        f.suptitle(title, y=.92, fontsize=18)
    else:
        f.suptitle(title, y=.95, fontsize=18)
    return f

def plot_forecast_ts(df_test_m, df_test):
    fontsize = 16

    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(1, 1, height_ratios=None)
    facecolor='white'
    ax0 = plt.subplot(gs[0], facecolor=facecolor)
    # df_test.plot(ax=ax0)
    ax0.plot_date(df_test.index, df_test.iloc[:,0], ls='-',
                  label='Observed', c='black')

    ax0.plot_date(df_test.index, df_test.iloc[:,1], ls='-', c='red',
                  label=r'Ridge Regression forecast')
    # ax0.set_xticks()
    # ax0.set_xticklabels(df_test.index.year,
    ax0.set_ylabel('Standardized Soy Yield', fontsize=fontsize)
    ax0.tick_params(labelsize=fontsize)
    ax0.axhline(y=0, color='black', lw=1)
    ax0.legend(fontsize=fontsize)

    df_scores = df_test_m.loc[0][df_test_m.columns[0][0]]
    Texts1 = [] ; Texts2 = [] ;
    textprops = dict(color='black', fontsize=fontsize+4, family='serif')
    rename_met = {'RMSE':'RMSE-SS', 'corrcoef':'Corr. Coeff.', 'MAE':'MAE-SS',
                  'BSS':'BSS', 'roc_auc_score':'ROC-AUC', 'r2_score':'$r^2$',
                  'mean_absolute_percentage_error':'MAPE'}
    for k in df_scores.index:
        label = rename_met[k]
        val = round(df_scores[k], 2)
        Texts1.append(TextArea(f'{label}',textprops=textprops))
        Texts2.append(TextArea(f'{val}',textprops=textprops))
    texts_vbox1 = VPacker(children=Texts1,pad=0,sep=4)
    texts_vbox2 = VPacker(children=Texts2,pad=0,sep=4)

    ann1 = AnnotationBbox(texts_vbox1,(.02,.15),xycoords=ax0.transAxes,
                                box_alignment=(0,.5),
                                bboxprops = dict(facecolor='white',
                                                 boxstyle='round',edgecolor='white'))
    ann2 = AnnotationBbox(texts_vbox2,(.21,.15),xycoords=ax0.transAxes,
                                box_alignment=(0,.5),
                                bboxprops = dict(facecolor='white',
                                                 boxstyle='round',edgecolor='white'))
    ann1.set_figure(fig) ; ann2.set_figure(fig)
    fig.artists.append(ann1) ; fig.artists.append(ann2)
    return


#%% Conditional continuous forecast

def get_df_forcing_cond_fc(rg_list, target_ts, fcmodel, kwrgs_model,
                           mean_vars=['sst', 'smi']):
    for j, rg in enumerate(rg_list):

        PacAtl = []
        # find west-sub-tropical Atlantic region
        df_labels = find_precursors.labels_to_df(rg.list_for_MI[0].prec_labels)
        dlat = df_labels['latitude'] - 29
        dlon = df_labels['longitude'] - 290
        zz = pd.concat([dlat.abs(),dlon.abs()], axis=1)
        Atlan = zz.query('latitude < 10 & longitude < 10')
        if Atlan.size > 0:
            PacAtl.append(int(Atlan.index[0]))
        PacAtl.append(int(df_labels['n_gridcells'].idxmax())) # Pacific SST
        PacAtl = [int(df_labels['n_gridcells'].idxmax())] # only Pacific

        weights_norm = rg.prediction_tuple[1]# .mean(axis=0, level=1)
        # weights_norm = weights_norm.sort_values(ascending=False, by=0)

        keys = [k for k in weights_norm.index.levels[1] if int(k.split('..')[1]) in PacAtl]
        keys = [k for k in keys if 'sst' in k] # only SST
        labels = ['..'.join(k.split('..')[1:]) for k in keys] + ['0..smi_sp'] # add smi just because it almost always in there

        df_mean, keys_dict = get_df_mean_SST(rg, mean_vars=mean_vars,
                                             n_strongest='all',
                                             weights=True,
                                             fcmodel=fcmodel,
                                             kwrgs_model=kwrgs_model,
                                             target_ts=target_ts,
                                             labels=labels)


        # apply weighted mean based on coefficients of precursor regions
        weights_norm = weights_norm.loc[pd.IndexSlice[:,keys],:]
        # weights_norm = weights_norm.div(weights_norm.max(axis=0))
        weights_norm = weights_norm.div(weights_norm.max(axis=0, level=0), level=0)
        weights_norm = weights_norm.reset_index().pivot(index='level_0', columns='level_1')[0]
        weights_norm.index.name = 'fold' ; df_mean.index.name = ('fold', 'time')
        PacAtl_ts = weights_norm.multiply(df_mean[keys], axis=1, level=0)
        PacAtl_ts = functions_pp.get_df_test(PacAtl_ts.mean(axis=1),
                                             df_splits=rg.df_splits)

        rg.df_forcing = PacAtl_ts

def cond_forecast_table(rg_list, score_func_list, n_boot=0):
    df_test_m = rg_list[0].verification_tuple[2]
    quantiles = [.15, .25]
    metrics = df_test_m.columns.levels[1]
    if n_boot > 0:
        cond_df = np.zeros((metrics.size, len(rg_list), len(quantiles)*2, n_boot))
    else:
        cond_df = np.zeros((metrics.size, len(rg_list), len(quantiles)*2))
    for i, met in enumerate(metrics):
        for j, rg in enumerate(rg_list):

            PacAtl_ts = rg.df_forcing

            prediction = rg.prediction_tuple[0]
            df_test = functions_pp.get_df_test(prediction,
                                               df_splits=rg.df_splits)

            # df_test_m = rg.verification_tuple[2]
            # cond_df[i, j, 0] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
            for k, l in enumerate(range(0,4,2)):
                q = quantiles[k]
                low = PacAtl_ts < PacAtl_ts.quantile(q)
                high = PacAtl_ts > PacAtl_ts.quantile(1-q)
                mask_anomalous = np.logical_or(low, high)
                # anomalous Boundary forcing
                condfc = df_test[mask_anomalous.values]
                # condfc = condfc.rename({'causal':periodnames[i]}, axis=1)
                cond_verif_tuple = fc_utils.get_scores(condfc,
                                                       score_func_list=score_func_list,
                                                       n_boot=n_boot,
                                                       score_per_test=False,
                                                       blocksize=1,
                                                       rng_seed=1)
                df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple
                rg.cond_verif_tuple  = cond_verif_tuple
                if n_boot == 0:
                    cond_df[i, j, l] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
                else:
                    cond_df[i, j, l, :] = df_boot[df_boot.columns[0][0]][met]
                # mild boundary forcing
                higher_low = PacAtl_ts > PacAtl_ts.quantile(.5-q)
                lower_high = PacAtl_ts < PacAtl_ts.quantile(.5+q)
                mask_anomalous = np.logical_and(higher_low, lower_high) # changed 11-5-21

                condfc = df_test[mask_anomalous.values]
                # condfc = condfc.rename({'causal':periodnames[i]}, axis=1)
                cond_verif_tuple = fc_utils.get_scores(condfc,
                                                       score_func_list=score_func_list,
                                                       n_boot=n_boot,
                                                       score_per_test=False,
                                                       blocksize=1,
                                                       rng_seed=1)
                df_train_m, df_test_s_m, df_test_m, df_boot = cond_verif_tuple
                if n_boot == 0:
                    cond_df[i, j, l+1] = df_test_m[df_test_m.columns[0][0]].loc[0][met]
                else:
                    cond_df[i, j, l+1, :] = df_boot[df_boot.columns[0][0]][met]

    columns = [[f'strong {int(q*200)}%', f'weak {int(q*200)}%'] for q in quantiles]
    columns = functions_pp.flatten(columns)
    if n_boot > 0:
        columns = pd.MultiIndex.from_product([columns, list(range(n_boot))])

    df_cond_fc = pd.DataFrame(cond_df.reshape((len(metrics)*len(rg_list), -1)),
                              index=pd.MultiIndex.from_product([list(metrics), [rg.fc_month for rg in rg_list]]),
                              columns=columns)


    return df_cond_fc


def boxplot_cond_fc(df_cond, metrics: list=None, forcing_name: str='', composite = 30):
    '''


    Parameters
    ----------
    df_cond : pd.DataFrame
        should have pd.MultiIndex of (metric, lead_time) and pd.MultiIndex column
        of (composite, n_boot).
    metrics : list, optional
        DESCRIPTION. The default is None.
    forcing_name : str, optional
        DESCRIPTION. The default is ''.
    composite : TYPE, optional
        DESCRIPTION. The default is 30.

    Returns
    -------
    f : TYPE
        DESCRIPTION.

    '''
    #%%
    import matplotlib as mpl
    mpl.rcParams.update(mpl.rcParamsDefault)

    rename_m = {'corrcoef': 'Corr. coeff.', 'RMSE':'RMSE-SS',
                'MAE':'MAE-SS', 'mean_absolute_error':'Mean Absolute Error',
                'r2_score':'$r^2$ score', 'BSS':'BSS', 'roc_auc_score':'AUC-ROC'}

    n_boot = df_cond.columns.levels[1].size
    columns = [c[0] for c in df_cond.columns[::n_boot]]
    plot_cols = [f'strong {composite}%', f'weak {composite}%']
    if metrics is None:
        indices = np.unique([t[0] for t in df_cond.index],return_index=True)[1]
        metrics = [n[0] for n in df_cond.index[sorted(indices)]] # preserves order

    df_cond = df_cond.loc[metrics]

    metric = metrics[0]
    indices = np.unique([t[1] for t in df_cond.index],return_index=True)[1]
    lead_times = [n[1] for n in df_cond.index[sorted(indices)]] # preserves order
    f, axes = plt.subplots(len(metrics),len(lead_times),
                           figsize=(len(lead_times)*3, 6*len(metrics)**0.5),
                           sharex=True)
    axes = axes.reshape(len(metrics), -1)


    for iax, index in enumerate(df_cond.index):
        metric = index[0]
        lead_time = index[1]
        row = metrics.index(metric) ; col = list(lead_times).index(lead_time)
        ax = axes[row, col]

        data = df_cond.loc[metric, lead_time].values.reshape(len(columns), -1)
        data = pd.DataFrame(data.T, columns=columns)[plot_cols]

        perc_incr = (data[plot_cols[0]].mean() - data[plot_cols[1]].mean()) / abs(data[plot_cols[1]].mean())

        nlabels = plot_cols.copy() ; widths=(.5,.5)
        nlabels = [l.split(' ')[0] for l in nlabels]
        nlabels = [l.capitalize() for l in nlabels]


        boxprops = dict(linewidth=2.0, color='black')
        whiskerprops = dict(linestyle='-',linewidth=2.0, color='black')
        medianprops = dict(linestyle='-', linewidth=2, color='red')
        ax.boxplot(data, labels=['', ''],
                   widths=widths, whis=[2.5, 97.5], boxprops=boxprops, whiskerprops=whiskerprops,
                   medianprops=medianprops, showmeans=True)

        text = f'{int(100*perc_incr)}%'
        if perc_incr > 0: text = '+'+text
        ax.text(0.98, 0.98,text,
                horizontalalignment='right',
                verticalalignment='top',
                transform = ax.transAxes,
                fontsize=15)

        if metric == 'corrcoef' or metric=='roc_auc_score':
            ax.set_ylim(0,1) ; steps = 1
            yticks = np.round(np.arange(0,1.01,.2), 2)
            ax.set_yticks(yticks[::steps])
            ax.set_yticks(yticks, minor=True)
            ax.tick_params(which='minor', length=0)
            ax.set_yticklabels(yticks[::steps])
            if metric=='roc_auc_score':
                ax.axhline(y=0.5, color='black', linewidth=1)
        elif metric == 'mean_absolute_error':
            yticks = np.round(np.arange(0,1.61,.4), 1)
            ax.set_ylim(0,1.6) ; steps = 1
            ax.set_yticks(yticks[::steps])
            ax.set_yticks(yticks, minor=True)
            ax.tick_params(which='minor', length=0)
            ax.set_yticklabels(yticks[::steps])
        else:
            yticks = np.round(np.arange(-.2,1.1,.2), 1)
            ax.set_ylim(-.3,1) ; steps = 2
            ax.set_yticks(yticks[::steps])
            ax.set_yticks(yticks, minor=True)
            ax.tick_params(which='minor', length=0)
            ax.set_yticklabels(yticks[::steps])
            ax.axhline(y=0, color='black', linewidth=1)

        ax.tick_params(which='both', grid_ls='-', grid_lw=1,width=1,
                       labelsize=16, pad=6, color='black')
        ax.grid(which='both', ls='--')
        if col == 0:
            ax.set_ylabel(rename_m[metric], fontsize=18, labelpad=2)
        if row == 0:
            ax.set_title(lead_time, fontsize=18)
        if row+1 == len(metrics):
            ax.set_xticks([1,2])
            ax.set_xticklabels(nlabels, fontsize=14)
            ax.set_xlabel(forcing_name, fontsize=15)
    f.subplots_adjust(wspace=.4)
    #%%
    return f
