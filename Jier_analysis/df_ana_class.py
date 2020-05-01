# -*- coding: utf-8 -*-
import inspect
import os
import sys
import warnings
from typing import List, Tuple, Union, Dict

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
subdates_dir = os.path.join(main_dir, 'RGCPD/')
fc_dir = os.path.join(main_dir, 'forecasting/')

if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(subdates_dir)
    sys.path.append(fc_dir)

import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import mtspec
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import tables
import xarray
from matplotlib import cycler
from scipy import stats
from statsmodels.tsa import stattools

from RGCPD.core_pp import get_subdates
from forecasting import *

pd.options.plotting.matplotlib.register_converters = True # apply custom format as well to matplotlib
plt.style.use('seaborn')

# TODO MAKE ANY FUNCTIONS WORK WITH REGULAR TS AND MULT LEVEL TS FROM PARENT AND CHILD CLASSES
# TODO MAKE BASE CLASSES FOR DFANALYSIS AND VISUALANALYSIS SUCH THAT CHILD CLASS CAN INHERIT DEFAULT VALUES
# TODO FIND WAY TO VISUALISE PARAMS NEEDED IN CHILD CLASS THAT OTHERWISE ONLY BE ACCESSIBLE IN PARENT CLASS
# TODO FIND BETTER WAY TO FIND THE "20" AUTOMATICALLY THAN JUST HARD CODED IN SUBSET SERIES FUNC
class DataFrameAnalysisBase:
    pass 

class DataFrameAnalysis:
    def __init__(self):
        self.keys = None
        self.time_steps = None 
        self.methods = dict() 
        self.period = None
        self.window_size = 0 
        self.threshold_bin = 0
        self.target_variable = None
        self.file_path = ""
        self.selection_year = 0
        self.alpha = 0.01
        self.max_lag = None
        self.flatten = lambda l: [item for sublist in l for item in sublist]

    def __str__(self):
        return f'{self.__class__.__name__}{self.__dict__}'

    def __repr__(self):
        return f'{self.__class__.__name__}{self.__dict__!r}'

    @staticmethod
    def autocorrelation_stats_meth(time_serie, max_lag=None, alpha=0.01):
        " Autocorrelation for 1D-arrays"
        if max_lag == None :
            max_lag = time_serie.size
        return stattools.acf(time_serie.values, nlags=max_lag - 1, unbiased=False, alpha=alpha, fft=True)
    
    @staticmethod
    def load_hdf(file_path):
        hdf = h5py.File(file_path, 'r+')
        dict_of_dfs = {k:pd.read_hdf(file_path, k) for k in hdf.keys()}
        hdf.close()
        return dict_of_dfs
    
    @staticmethod
    def save_hdf( dict_of_dfs, file_path):
        with pd.HDFStore(file_path, 'w') as hdf :
            for k, items in dict_of_dfs.items():
                hdf.put(k, items,  format='table', data_columns=True)
        return 
    
    def __get_keys(self, data_frame, keys):
        if keys == None:
            # retrieve only float series
            keys = self.keys
            type_check = np.logical_or(data_frame.dtypes == 'float',data_frame.dtypes == 'float32')
            keys = type_check[type_check].index
        return keys

    def loop_df_ana(self, df, function, keys=None, to_np=False, kwargs=None):
        # Should be analysis from any function which return non-tuple like results
        keys = self.__get_keys(df, keys)
        if to_np:
            output = df.apply(function, raw=True, **kwargs)
        output = df.apply(function, **kwargs)
        return pd.DataFrame(output, columns=keys)
    
    def loop_df(self, df, functions, args=None, kwargs=None, keys=None):
        # TODO Test this functionality
        # Should be analysis from any function with methods from functions that might return tuples
        # method should be dict with labels as function name and values as function calls
        # keys = self.__get_keys(df, keys)
        # df = df.loc[:,keys]
        # print(type(functions), functions)
        # # if not isinstance(functions, list) or not isinstance(functions, dict) or isinstance(functions, property):
        # return self.apply_concat_series(df, functions, arg=args)
        raise NotImplementedError
    
    def subset_series(self, df_serie, time_steps=None, select_years: Union[int, list] = None):
        if time_steps == None:
            if hasattr(df_serie.index, 'levels'):
                df_serie = df_serie.loc[0]
            _, conf_intval = self.apply_concat_series(df_serie[df_serie.columns[df_serie.dtypes != bool]], self.autocorrelation_stats_meth)
            conf_low = [np.where(conf_intval[i][:, 0] < 0)[0] for i in range(len(conf_intval))]
            numb_times = [[] for _ in range(len(conf_low))]
            for i in range(len(conf_low)):
                for idx in conf_low[i]:
                    numb_times[i].append(idx + 1 - conf_low[i][0])
            cut_off = [conf_low[i][np.where(np.array(numb_times[i]) == 1)][0] for i in range(len(conf_low))]
            time_steps = [20 * i for i in cut_off]
            if hasattr(df_serie.index, 'levels'):
                serie = [df_serie.iloc[:step,i] for i, step in enumerate(time_steps)]
            else:
                serie = [df_serie.iloc[:i] for i in time_steps]
        else:
            if hasattr(df_serie.index, 'levels'):
                time_steps = [time_steps] * df_serie.columns.size
                serie = [df_serie.iloc[:step, i] for i, step in enumerate(time_steps)]
            else:
                serie = [df_serie.iloc[:i] for i in time_steps]

        if select_years != None:
            if not isinstance(select_years, list):
                select_years = [select_years]
            if hasattr(df_serie.index, 'levels'):
                date_time = self.get_one_year(df_serie.index, *(select_years))
                serie = [df_serie.loc[date_time, col] for col in df_serie.columns]
                return serie
            else:
                date_time = self.get_one_year(df_serie, *(select_years))
            serie = df_serie.iloc[date_time]
        return serie

    def spectrum(self, y,  methods:Dict[str, object]=None, year_max=0.5):
  
        freq_dframe = None
        freq= None
        idx = None
        if hasattr(y.index, 'levels'):
            y = y.loc[0]
        if methods == None:
            methods = {'periodogram': self.periodogram}
        
        if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
            f = (y.index[1] - y.index[0]).days
            freq_dframe = self._calc_freq_df(f)
    
        results = {}
        for label, func in methods.items():
            freq, spec =  self.apply_concat_series(y, func=func, arg=1)
            periods = [self._freq_to_period(freq[i][1:],freq_dframe)  for i in range(len(freq))]
            idx = [int(np.argwhere(periods[i]-year_max ==min(abs(periods[i] - year_max)))[0]) for i in range(len(freq))]
            period  = [periods[i][:idx[i] + 1] for i in range(len(freq))]
            spec = [ spec[i][1:idx[i] + 2] for i in range(len(spec))]
            results[label] = [period, spec]          
        return [results, freq_dframe, freq, idx]

    def accuracy(self, y, sample='auto', auc_cutoff=None):
        # TODO FIX HARD CODED NUMBERS OF MULTIPLICATION SAMPLE SIZE OR CUT OFF
        if hasattr(y.index, 'levels'):
            y = y.loc[0]
        acc, conf_intval = self.apply_concat_series(y[y.columns[y.dtypes != bool]], self.autocorrelation_stats_meth)
        tfreq = ((y.index[1] - y.index[0]).days, 'time [days]') if isinstance(y.index , pd.core.indexes.datetimes.DatetimeIndex) else (1,'timesteps')

        if sample =='auto':
            conf_low = [np.where(conf_intval[i][:, 0] < 0)[0] for i in range(len(conf_intval))]
            numb_times = [[] for _ in range(len(conf_low))]
            
            for i in range(len(conf_low)):
                for idx in conf_low[i]:
                    numb_times[i].append(idx + 1 - conf_low[i][0])
            try:
                cut_off = [conf_low[i][np.where(np.array(numb_times[i]) == 1)][0] for i in range(len(conf_low))]

            except:
                cut_off = tfreq[0] * 20

            if isinstance(cut_off, list):
                sample = [2 * cut_off[i] for i in range(len(cut_off))]
            else:
                sample = 2 * cut_off
        else:
            cut_off = int(sample / 2)

        if auc_cutoff == None or isinstance(auc_cutoff, int):
            auc_cutoff = cut_off
            auc = [np.trapz(acc[i][:auc_cutoff[i]], x=range(auc_cutoff[i])) for i in range(len(conf_intval)) ]
            text = [f'AUC {auc[i]} range lag {auc_cutoff[i]}' for i in range(len(auc_cutoff))]

        elif isinstance(auc_cutoff, tuple):
            auc = [np.trapz(acc[i][auc_cutoff[i][0]:auc_cutoff[i][1]], x=range(auc_cutoff[i][0], auc_cutoff[i][0])) for i in range(len(conf_intval)) ]
            text = [f'AUC {auc} range lag {auc_cutoff[i][0]}-{auc_cutoff[i][1]}' for i in range(len(auc_cutoff)) ]
        return [auc, acc,  auc_cutoff, sample, conf_intval, text, tfreq]

    def _calc_freq_df(self, freq):
        if freq in [28, 29, 30, 31]:
            freq = 'month'
        elif isinstance(freq, int):
            freq = int(365 / freq)
        else:
            freq = 1
        return freq

    def _freq_to_period(self, xfreq, freq):
        if freq =='month':
            periods = 1 / (xfreq * 12)
        elif isinstance(freq, int):
            periods = 1 / ( xfreq * freq)
        return np.round(periods, 3)

    def _period_to_freq(self, periods, freq):
        if freq =='month':
            freq = 1 / (periods * 12)
        else:
            freq = 1 / (periods * freq)
        return np.round(freq, 1)

    def resample(self,df, window_size=20, lag=0, columns=List, to_freq_=False, freq='M'):
        if to_freq_ == True:
            return self._resample_per_freq(df, to_freq=freq)
        else:
            splits = df.index.levels[0]
            df_train_is_true = df['TrainIsTrue']
            test_list = [ df[columns].loc[split][df_train_is_true[split] == False] for split in range(splits.size) ]

            df_test = pd.concat(test_list).sort_index()
            # for pre_cursor in df_test.columns[1:]:
            #     df_test[df_test.columns[1:]] = df_test[df_test[pre_cursor].shift(periods=- lag)
            df_test[df_test.columns[1:]] = df_test[df_test[df_test.columns[1:]]].shift(periods=- lag)
            return df_test.resample(f'{window_size}D').mean()
    
    def _re_smaple_per_freq(self, df, to_freq='M'):
        return df.resample(to_freq).mean()

    def select_period(self, df, start_end_date, start_end_year, leap_year, targ_var_mask='RV_mask', rename=False):
        if hasattr(df.index, 'levels'):
            dates_target_var_origin = df.loc[0].index[df.loc[0][targ_var_mask] == True ]
            df_period  = get_subdates(dates_target_var_origin, start_end_date, start_end_year, leap_year)
        else:
            raise ValueError('Provided not multi index data', sys.exc_info())
        if rename:
             df_period = df_period.rename(rename, axis= 1)
             return df_period
        return df_period

    def apply_concat_dFrame(self, df, field, func, col_name, concat=False):
        if concat:
            return pd.concat((df, df[field].apply(lambda cell : pd.Series(func(cell), index=col_name))), axis=1)
        else:
            df_copy = df.copy()
            return df_copy[field].apply(lambda cell : pd.Series(func(cell), index=col_name))

    def apply_concat_series(self, series, func, arg=None):
        return zip(*series.apply(func, args=(arg,)))

    def multi_tape_spectr(self, time_serie, sampling_period=1, band_width=4, numb_tapers=4):
       spectrum, frequence, _, _, _ =  mtspec.mtspec(data=time_serie, delta=sampling_period, 
                                    time_bandwidth=band_width, number_of_tapers=numb_tapers, statistics=True)
       return [frequence, spectrum]

    def periodogram(self, time_serie, sampling_period=1):
        frequence = np.fft.rfftfreq(len(time_serie))
        spectrum = 2 * np.abs(np.fft.rfft(time_serie))**2 / len(time_serie)
        return [frequence, spectrum]

    def fft_np(self, data, sampling_period=1.0):
        yfft = sp.fftpack.fft(data)
        ypsd = np.abs(yfft)**2
        ypsd = 2.0 / len(data) * ypsd
        fft_frequence = sp.fftpack.fftfreq(len(ypsd), sampling_period)
        return [fft_frequence, ypsd]
        
    def cross_corr_p_val(self, data, alpha=0.05):
        def pearson_pval(x, y):
            return stats.pearsonr(x, y)[1]

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            pval_matrix = data.corr(method=pearson_pval).to_numpy()
            sig_mask = pval_matrix < alpha
            cross_cor = data.corr(method='pearson')
            return [cross_cor, sig_mask, pval_matrix]
        else:
            raise ValueError('Please provide correct datatype', sys.exc_info())

    def get_one_year(self, pd_date_time, *args):
        if hasattr(pd_date_time, 'levels'):
            pd_date_time = pd_date_time.levels[1]
        else:
            pd_date_time = pd.to_datetime(pd_date_time)
        first_year =  pd_date_time.year[0]
        if len(args) != 0:
            dates = [pd_date_time.where(pd_date_time.year == arg).dropna() for arg in args]
            return  pd.to_datetime(self.flatten(dates))
        else:
            return pd_date_time.where(pd_date_time.year == first_year).dropna() 

    def remove_leap_period(self, date_time_or_xr):
        no_leap_month = None
        mask = None
        try:
            if pd.to_datetime(date_time_or_xr, format='%Y-%b-%d', errors='coerce'):
                date_time = pd.to_datetime(date_time_or_xr.time.values)
                mask = np.logical_and((date_time.month == 2), (date_time.day == 29))
                no_leap_month = date_time[mask==False]
            else: 
                raise ValueError('Not dataframe datatype')
        except ValueError as v_err_1:
            try:
                if isinstance(date_time_or_xr, xarray):
                    mask = np.logical_and((date_time_or_xr.month == 2), (date_time_or_xr.day == 29))
                    no_leap_month = date_time_or_xr[mask==False]
                    no_leap_month = date_time_or_xr.sel(time=no_leap_month)
                else:
                    raise ValueError('Not xarray datatype')
            except ValueError as v_err_2:
                print("Twice ValueError generated ", sys.exc_info(), v_err_1, v_err_2)
        return no_leap_month
    
class VisualizAnalysisBase:
    pass

class VisualizeAnalysis:
    def __init__(self, col_wrap=3, sharex='col', sharey='row'):
        self.nice_colors = ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB']
        self.colors_nice = cycler('color',
                        self.nice_colors)
        self.colors_datasets = sns.color_palette('deep')

        plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
            axisbelow=True, grid=True, prop_cycle=self.colors_nice)
        plt.rc('grid', color='w', linestyle='solid')
        plt.rc('xtick', direction='out', color='black')
        plt.rc('ytick', direction='out', color='black')
        plt.rc('patch', edgecolor='#E6E6E6')
        plt.rc('lines', linewidth=2)

        mpl.rcParams['figure.figsize'] = [7.0, 5.0]
        mpl.rcParams['figure.dpi'] = 100
        mpl.rcParams['savefig.dpi'] = 600

        mpl.rcParams['font.size'] = 13
        mpl.rcParams['legend.fontsize'] = 'medium'
        mpl.rcParams['figure.titlesize'] = 'medium'
        self.col_wrap = col_wrap
        self.gridspec_kw = {'wspace':0.5, 'hspace':0.4}
        self.sharex = sharex
        self.sharey = sharey

    def __str__(self):
        return f'{self.__class__.__name__}{self.__dict__}'

    def __repr__(self):
        return f'{self.__class__.__name__}{self.__dict__!r}'

    def _column_wrap(self, df):
        if df.columns.size % self.col_wrap == 0:
            return int(df.columns.size / self.col_wrap)
        return int(df.columns.size / self.col_wrap) + 1

    def _subplots_func_adjustment(self, col=None):
        if col == None:
            return plt.subplots(constrained_layout=True)
        else:
            if col % self.col_wrap == 0:
                return plt.subplots(int(col/self.col_wrap),self.col_wrap, constrained_layout=True)
            else:
                return plt.subplots(int(col/self.col_wrap) + 1, self.col_wrap, constrained_layout=True)

    def subplots_fig_settings(self, df):
        row = self._column_wrap(df)
        return plt.subplots(row, self.col_wrap, sharex=self.sharex, 
        sharey=self.sharey, figsize= (3* self.col_wrap, 2.5* row), gridspec_kw=self.gridspec_kw, constrained_layout=True)

    def vis_dataframe(self, df):
        # _, ax = self.subplots_fig_settings(df)
        if hasattr(df.index, 'levels'):
            df.unstack(0).plot(subplots=True)
        else:
            df.plot(subplots=True)
        plt.show()

    def vis_accuracy(self, values, title):
        # TODO FIND OUT WHY DO WE NEED TO DIVIDE BY 5
        auc, aut_corr, auc_cutoffs, sample_cutoff, conf_intval, text, time_freq= values
        fig, ax = self._subplots_func_adjustment(col=len(sample_cutoff))
        x_labels = [[] for _ in range(len(sample_cutoff)) ]
        for i in range(len(sample_cutoff)): 
            for x in range(sample_cutoff[i]):
                x_labels[i].append(x* time_freq[0])
        
        # High confindence interval
        high_conf = [min(1, h) for i in range(len(sample_cutoff)) for h in conf_intval[i][:, 1][:sample_cutoff[i]]  ]
        # Low confidennce interval
        low_conf = [min(1, k) for i in range(len(sample_cutoff)) for k in conf_intval[i][:, 0][:sample_cutoff[i]] ]
        numb_labels = [max(1, int(sample_cutoff[i] / 5)) for i in range(len(sample_cutoff))]
  
        for i in range(len(sample_cutoff)):
            ax[i].plot(x_labels[i], high_conf[:len(x_labels[i])], color='orange')
            ax[i].plot(x_labels[i], low_conf[:len(x_labels[i])], color='orange')
            ax[i].plot(x_labels[i], aut_corr[i][:len(x_labels[i])] )
            ax[i].scatter(x_labels[i], aut_corr[i][:len(x_labels[i])] )
            ax[i].hlines(y= 0, xmin=min(x_labels[i]), xmax=max(x_labels[i]))
            ax[i].text(0.99, 0.99, text[i], transform=ax[i].transAxes, horizontalalignment='right', fontdict={'fontsize':8})
            ax[i].set_xticks(x_labels[i][::numb_labels[i]])
            ax[i].set_xticklabels(x_labels[i][::numb_labels[i]], fontsize=10)
            ax[i].set_xlabel(time_freq[1], fontsize=10)

            if title:
                ax[i].set_title(title[i], fontsize=10)
        plt.show()

    def vis_timeseries(self, list_of_pdseries):
        _, ax = self._subplots_func_adjustment(col=len(list_of_pdseries))
        for series in list_of_pdseries:
            with pd.plotting.plot_params.use('x_compat', True):
                if hasattr(series.index, 'levels'):
                    series.unstack(level=0).plot(subplots=True, ax=ax)
                else:
                    series.plot(subplots=True, ax =ax)
        plt.show()
    
    def vis_scatter(self, df, target_var, aggr, title):
        fig, ax = self._subplots_func_adjustment()

        if aggr == 'annual':
            df_gr = df.groupby(df.index.year).mean()
            target_var_gr = target_var.grouby(target_var.index.year).mean()
            ax.scatter(df_gr, target_var_gr)
        else:
            ax.scatter(df, target_var)
        if title:
            ax.set_title(title, fontsize=10)
        plt.show()
    
    def vis_time_serie_matrix(self, df_period, cross_corr, sig_mask, pval):
        _, ax = self._subplots_func_adjustment()
        plt.figure(figsize=(10, 10))

        # Generate mask for upper triangle matrix 
        mask_triangle = np.zeros_like(cross_corr, dtype=bool)
        mask_triangle[np.triu_indices_from(mask_triangle)] = True
        mask_signal = mask_triangle.copy()
        mask_signal[sig_mask == False] = True

        # Removing meaningless row and column 
        cross_corr = cross_corr.columns.drop(cross_corr.columns[0], axis=0).drop(cross_corr.columns[-1], axis=1)
        mask_signal = mask_signal[1: , :-1]
        mask_triangle = mask_triangle[1: , :-1]

        # Custom cmap for corr matrix plot
        cust_cmap = sns.diverging_palette(220, 10, n=9, l=30, as_cmap=True)

        signf_labels = self.significance_annotation(cross_corr, mask_signal)

        ax = sns.heatmap(cross_corr, ax=ax, mask=mask_triangle, cmap=cust_cmap, vmax=1, center=0,
        square=True, linewidths=.5, cbar_kws={'shrink': .8}, annot=signf_labels, annot_kws={'size':30}, cbar=False, fmt='s')

        ax.tick_params(axis='both', labelsize=15, bottom=True, top=False, left=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
        ax.set_xticklabels(cross_corr.columns, fontdict={'fontweight': 'bold',
                                                        'fontsize': 25})
        ax.set_yticklabels(cross_corr.index, fontdict={'fontweight':'bold',
                                                    'fontsize':25}, rotation=0)

        plt.show()

    def vis_spectrum(self, title, subtitle, results, freqdf, freq, idx_):
        fig, ax = self._subplots_func_adjustment(col=len(subtitle))
        # print(ax.shape)
        # sys.exit()
        # TODO Better way to plot all results tuples  instead of double for-loops
        counter = len(results)
        label = list(results.keys())

        for _,values in results.items():
            for idx in range(len(subtitle)):
                # ax = ax.flatten()[0]
                ax[idx].plot(values[0][idx], values[1][idx], ls='-', c=self.nice_colors[counter], label=label)

                ax[idx].set_title(subtitle[idx])
                ax[idx].set_xscale('log')
                ax[idx].set_xticks(values[0][idx][np.logical_or(values[0][idx] % 2 == 0, values[0][idx] % 1 == 0)])
                ax[idx].set_xticklabels(np.array(values[0][idx][np.logical_or(values[0][idx] % 2 == 0, values[0][idx] % 1 == 0)], dtype=int))
                ax[idx].set_xlim((values[0][idx][0], values[0][idx][-1]))
                ax[idx].set_xlabel('Periods [years]', fontsize=9)
                ax[idx].tick_params(axis='both', labelsize=8)

                ax2 = ax[idx].twiny()
                ax2.plot(values[0][idx], values[1][idx], ls='-', c=self.nice_colors[counter], label=label)
                ax2.set_xscale('log')
                ax2.set_xticks(values[0][idx][np.logical_or(values[0][idx] % 2 == 0, values[0][idx] % 1 == 0)])
                ax2.set_xticklabels(np.round(freq[idx][1:idx_[idx] + 2][np.logical_or(values[0][idx] % 2 == 0, values[0][idx] % 1 == 0)], 3))
                ax2.set_xlim((values[0][idx][0], values[0][idx][-1])) 
                ax2.tick_params(axis='both', labelsize=8)
                if freqdf == 'month':
                    ax2.set_xlabel('Frequency [1 /months]', fontsize=8)
                else:
                    ax2.set_xlabel(f'Frequency [1/ {freqdf} days]', fontsize=6)
                ax[idx].legend(loc=0, fontsize='xx-small')
                counter -=  1
        if title != None:
            fig.suptitle(title, fontsize=10)
        plt.show()
           
    def significance_annotation(self, corr, pvals):
        corr_str = np.zeros_like( corr, dtype=str ).tolist()
        for i1, r in enumerate(corr.values):
            for i2, c in enumerate(r):
                if pvals[i1, i2] <= 0.05 and pvals[i1, i2] > 0.01:
                    corr_str[i1][i2] = '{:.2f}*'.format(c)
                if pvals[i1, i2] <= 0.01:
                    corr_str[i1][i2]= '{:.2f}**'.format(c)
                elif pvals[i1, i2] > 0.05:
                    corr_str[i1][i2]= '{:.2f}'.format(c)
        return np.array(corr_str)
    
class DFA:
    pass

if __name__ == "__main__":
    # df_ana = DataFrameAnalysis()
    # df_vis = VisualizeAnalysis()
    # print(repr(df_ana))
    # print(repr(df_vis))
    pass
