#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 12:17:25 2019

@author: semvijverberg
"""
import sys, os, inspect
if 'win' in sys.platform and 'dar' not in sys.platform:
    sep = '\\' # Windows folder seperator
else:
    sep = '/' # Mac/Linux folder seperator

import func_models as fc_utils
import itertools, os, re
import numpy as np
import xarray as xr
import scipy
import pandas as pd
from statsmodels.sandbox.stats import multicomp
import functions_pp, core_pp
import find_precursors
from func_models import apply_shift_lag
from class_RV import RV_class
from typing import Union
import uuid
flatten = lambda l: list(itertools.chain.from_iterable(l))
from joblib import Parallel, delayed

try:
    from tigramite.independence_tests import ParCorr
except:
    pass


class BivariateMI:

    def __init__(self, name, func=None, kwrgs_func={}, alpha: float=0.05,
                 FDR_control: bool=True, lags=np.array([1]),
                 lag_as_gap: bool=False, distance_eps: int=400,
                 min_area_in_degrees2=3, group_lag: bool=False,
                 group_split : bool=True, calc_ts='region mean',
                 selbox: tuple=None, use_sign_pattern: bool=False,
                 use_coef_wghts: bool=True, path_hashfile: str=None,
                 hash_str: str=None, dailytomonths: bool=False, n_cpu=1,
                 n_jobs_clust=-1, verbosity=1):
        '''

        Parameters
        ----------
        name : str
            Name that links to a filepath pointing to a Netcdf4 file.
        func : function to apply to calculate the bivariate
            Mutual Informaiton (MI), optional
            The default is applying a correlation map.
        kwrgs_func : dict, optional
            Arguments for func. The default is {}.
        alpha : float, optional
            significance threshold
        FDR_control: bool, optional
            Control for multiple hypothesis testing
        lags : np.ndarray, optional
            lag w.r.t. the the target variable at which to calculate the MI.
            The default is np.array([1]).
        lag_as_gap : bool, optional
            Interpret the lag as days in between last day of precursor
            aggregation window and first day of target window.
        distance_eps : int, optional
            The maximum distance between two gridcells for one to be considered
            as in the neighborhood of the other, only gridcells with the same
            sign are grouped together.
            The default is 400.
        min_area_in_degrees2 : TYPE, optional
            The number of samples gridcells in a neighborhood for a
            region to be considered as a core point. The parameter is
            propotional to the average size of 1 by 1 degree gridcell.
            The default is 400.
        group_split : str, optional
            If True, then region labels will be equal between different train test splits.
            If False, splits will clustered separately.
            The default is 'together'.
        calc_ts : str, optional
            Choose 'region mean' or 'pattern cov'. If 'region mean', a
            timeseries is calculated for each label. If 'pattern cov', the
            spatial covariance of the whole pattern is calculated.
            The default is 'region_mean'.
        selbox : tuple, optional
            has format of (lon_min, lon_max, lat_min, lat_max)
        use_sign_pattern : bool, optional
            When calculating spatial covariance, do not use original pattern
            but focus on the sign of each region. Used for quantifying Rossby
            waves.
        use_coef_wghts : bool, optional
            When True, using (corr) coefficient values as weights when calculating
            spatial mean. (will always be area weighted).
        dailytomonths : bool, optional
            When True, the daily input data will be aggregated to monthly data,
            subsequently, the pre-processing steps are performed (detrend/anomaly).
        n_cpu : int, optional
            Calculate different train-test splits in parallel using Joblib.
        n_jobs_clust : int, optional
            Perform DBSCAN clustering calculation in parallel. Beware that for
            large memory precursor fields with many precursor regions, DBSCAN
            can become memory demanding. If all cpu's are used, there may not be
            sufficient working memory for each cpu left.
        verbosity : int, optional
            Not used atm. The default is 1.

        Returns
        -------
        Initialization of the BivariateMI class

        '''
        self.name = name
        if func is None:
            self.func = corr_map
        else:
            self.func = func
        self._name = name + '_'+ self.func.__name__

        self.kwrgs_func = kwrgs_func

        self.alpha = alpha
        self.FDR_control = FDR_control
        #get_prec_ts & spatial_mean_regions
        self.calc_ts = calc_ts
        self.selbox = selbox
        self.use_sign_pattern = use_sign_pattern
        self.use_coef_wghts = use_coef_wghts
        self.lags = lags
        self.lag_as_gap = lag_as_gap

        self.dailytomonths = dailytomonths

        # cluster_DBSCAN_regions
        self.distance_eps = distance_eps
        self.min_area_in_degrees2 = min_area_in_degrees2
        self.group_split = group_split
        self.group_lag = group_lag
        self.verbosity = verbosity
        self.n_cpu = n_cpu
        self.n_jobs_clust = n_jobs_clust
        if hash_str is not None:
            assert path_hashfile is not None, 'Give path to search hashfile'
            self.load_files(self, path_hashfile=str, hash_str=str)

        return


    def bivariateMI_map(self, precur_arr, df_splits, df_RVfull): #
        #%%
        # self=rg.list_for_MI[0] ; precur_arr=self.precur_arr ;
        # df_splits = rg.df_splits ; df_RVfull = rg.df_fullts
        """
        This function calculates the correlation maps for precur_arr for different lags.
        Field significance is applied to test for correltion.
        RV_period: indices that matches the response variable time series
        alpha: significance level

        A land sea mask is assumed from settin all the nan value to True (masked).
        For xrcorr['mask'], all gridcell which are significant are not masked,
        i.e. bool == False
        """

        if type(self.lags) is np.ndarray and type(self.lags[0]) is not np.ndarray:
            self.lags = np.array(self.lags, dtype=np.int16) # fix dtype
            self.lag_coordname = self.lags
        else:
            self.lag_coordname = np.arange(len(self.lags)) # for period_means
        n_lags = len(self.lags)
        lags = self.lags
        self.df_splits = df_splits # add df_splits to self
        dates = self.df_splits.loc[0].index

        TrainIsTrue = df_splits.loc[0]['TrainIsTrue']
        RV_train_mask = np.logical_and(df_splits.loc[0]['RV_mask'], TrainIsTrue)
        if hasattr(df_RVfull.index, 'levels'):
            RV_ts = df_RVfull.loc[0][RV_train_mask.values]
        else:
            RV_ts = df_RVfull[RV_train_mask.values]
        targetstepsoneyr = functions_pp.get_oneyr(RV_ts)
        if type(self.lags[0]) == np.ndarray and targetstepsoneyr.size>1:
            raise ValueError('Precursor and Target do not align.\n'
                             'One aggregated value taken for months '
                             f'{self.lags[0]}, while target timeseries has '
                             f'multiple timesteps per year:\n{targetstepsoneyr}')
        yrs_precur_arr = np.unique(precur_arr.time.dt.year)
        if np.unique(dates.year).size != yrs_precur_arr.size:
            raise ValueError('Numer of years between precursor and Target '
                             'not match. Check if precursor period is crossyr, '
                             'while target period is not. '
                             'Mannually ensure start_end_year is aligned.')

        oneyr = functions_pp.get_oneyr(dates)
        if oneyr.size == 1: # single val per year precursor
            self._tfreq = 365
        else:
            self._tfreq = (oneyr[1] - oneyr[0]).days

        n_spl = df_splits.index.levels[0].size
        # make new xarray to store results
        xrcorr = precur_arr.isel(time=0).drop('time').copy()
        orig_mask = np.isnan(precur_arr[1])
        if 'lag' not in xrcorr.dims:
            # add lags
            list_xr = [xrcorr.expand_dims('lag', axis=0) for i in range(n_lags)]
            xrcorr = xr.concat(list_xr, dim = 'lag')
            xrcorr['lag'] = ('lag', self.lag_coordname)
        # add train test split
        list_xr = [xrcorr.expand_dims('split', axis=0) for i in range(n_spl)]
        xrcorr = xr.concat(list_xr, dim = 'split')
        xrcorr['split'] = ('split', range(n_spl))
        xrpvals = xrcorr.copy()


        def MI_single_split(RV_ts, precur_train, s, alpha=.05, FDR_control=True):


            lat = precur_train.latitude.values
            lon = precur_train.longitude.values

            z = np.zeros((lat.size*lon.size,len(lags) ) )
            Corr_Coeff = np.ma.array(z, mask=z)
            pvals = np.ones((lat.size*lon.size,len(lags) ) )

            dates_RV = RV_ts.index
            for i, lag in enumerate(lags):
                if type(lag) is np.int16 and self.lag_as_gap==False:
                    # dates_lag = functions_pp.func_dates_min_lag(dates_RV, self._tfreq*lag)[1]
                    m = apply_shift_lag(self.df_splits.loc[s], lag)
                    dates_lag = m[np.logical_and(m['TrainIsTrue']==1, m['x_fit'])].index
                    corr_val, pval = self.func(precur_train.sel(time=dates_lag),
                                               RV_ts.values.squeeze(),
                                               **self.kwrgs_func)
                elif type(lag) == np.int16 and self.lag_as_gap==True:
                    # if only shift tfreq, then gap=0
                    datesdaily = RV_class.aggr_to_daily_dates(dates_RV, tfreq=self._tfreq)
                    dates_lag = functions_pp.func_dates_min_lag(datesdaily,
                                                                self._tfreq+lag)[1]

                    tmb = functions_pp.time_mean_bins
                    corr_val, pval = self.func(tmb(precur_train.sel(time=dates_lag),
                                                           to_freq=self._tfreq)[0],
                                               RV_ts.values.squeeze(),
                                               **self.kwrgs_func)
                elif type(lag) == np.ndarray:
                    corr_val, pval = self.func(precur_train.sel(lag=i),
                                               RV_ts.values.squeeze(),
                                               **self.kwrgs_func)



                mask = np.ones(corr_val.size, dtype=bool)
                if FDR_control == True:
                    # test for Field significance and mask unsignificant values
                    # FDR control:
                    adjusted_pvalues = multicomp.multipletests(pval, method='fdr_bh')
                    ad_p = adjusted_pvalues[1]
                    pvals[:,i] = ad_p
                    mask[ad_p <= alpha] = False

                else:
                    pvals[:,i] = pval
                    mask[pval <= alpha] = False

                Corr_Coeff[:,i] = corr_val[:]
                Corr_Coeff[:,i].mask = mask

            Corr_Coeff = np.ma.array(data = Corr_Coeff[:,:], mask = Corr_Coeff.mask[:,:])
            Corr_Coeff = Corr_Coeff.reshape(lat.size,lon.size,len(lags)).swapaxes(2,1).swapaxes(1,0)
            pvals = pvals.reshape(lat.size,lon.size,len(lags)).swapaxes(2,1).swapaxes(1,0)
            return Corr_Coeff, pvals

        print('\n{} - calculating correlation maps'.format(precur_arr.name))
        np_data = np.zeros_like(xrcorr.values)
        np_mask = np.zeros_like(xrcorr.values)
        np_pvals = np.zeros_like(xrcorr.values)


        #%%
        # start_time = time()

        def calc_corr_for_splits(self, splits, df_RVfull, np_precur_arr, df_splits, output):
            '''
            Wrapper to divide calculating a number of splits per core, instead of
            assigning each split to a seperate worker.
            '''
            n_spl = df_splits.index.levels[0].size
            # reload numpy array to xarray (xarray not always picklable by joblib)
            precur_arr = core_pp.back_to_input_dtype(np_precur_arr[0], np_precur_arr[1],
                                                     np_precur_arr[2])
            RV_mask = df_splits.loc[0]['RV_mask']
            for s in splits:
                progress = int(100 * (s+1) / n_spl)
                # =============================================================================
                # Split train test methods ['random'k'fold', 'leave_'k'_out', ', 'no_train_test_split']
                # =============================================================================
                TrainIsTrue = df_splits.loc[s]['TrainIsTrue'].values==True
                RV_train_mask = np.logical_and(RV_mask, TrainIsTrue)
                if hasattr(df_RVfull.index, 'levels'):
                    RV_ts = df_RVfull.loc[s][RV_train_mask.values]
                else:
                    RV_ts = df_RVfull[RV_train_mask.values]

                if self.lag_as_gap: # no clue why selecting all datapoints, changed 26-01-2021
                    train_dates = df_splits.loc[s]['TrainIsTrue'][TrainIsTrue].index
                    precur_train = precur_arr.sel(time=train_dates)
                else:
                    precur_train = precur_arr[TrainIsTrue] # only train data

                n = RV_ts.size ; r = int(100*n/RV_mask[RV_mask].size)
                print(f"\rProgress traintest set {progress}%, trainsize=({n}dp, {r}%)", end="")

                output[s] = MI_single_split(RV_ts, precur_train, s,
                                            alpha=self.alpha,
                                            FDR_control=self.FDR_control)
            return output

        output = {}
        np_precur_arr = core_pp.to_np(precur_arr)
        if self.n_cpu == 1:
            splits = df_splits.index.levels[0]
            output = calc_corr_for_splits(self, splits, df_RVfull, np_precur_arr,
                                          df_splits, output)
        elif self.n_cpu > 1:
            splits = df_splits.index.levels[0]
            futures = []
            for _s in np.array_split(splits, self.n_cpu):
                futures.append(delayed(calc_corr_for_splits)(self, _s, df_RVfull,
                                                             np_precur_arr, df_splits,
                                                             output))
            futures = Parallel(n_jobs=self.n_cpu, backend='loky')(futures)
            [output.update(d) for d in futures]

        # unpack results
        for s in xrcorr.split.values:
            ma_data, pvals = output[s]
            np_data[s] = ma_data.data
            np_mask[s] = ma_data.mask
            np_pvals[s]= pvals
        print("\n")
        # print(f'Time: {(time()-start_time)}')
        #%%


        xrcorr.values = np_data
        xrpvals.values = np_pvals
        mask = (('split', 'lag', 'latitude', 'longitude'), np_mask )
        xrcorr.coords['mask'] = mask
        # fill nans with mask = True
        xrcorr['mask'] = xrcorr['mask'].where(orig_mask==False, other=orig_mask).drop('time')
        #%%
        return xrcorr, xrpvals

    # def check_exception_time_mean_period(df_splits, precur_train)

    def adjust_significance_threshold(self, alpha):
        self.alpha = alpha
        self.corr_xr.mask.values = (self.pval_xr > self.alpha).values

    def load_and_aggregate_precur(self, kwrgs_load):
        '''
        Wrapper to load in Netcdf and aggregated to n-mean bins or a period
        mean, e.g. DJF mean (see seasonal_mode.ipynb).

        Parameters
        ----------
        kwrgs_load : TYPE
            dictionary passed to functions_pp.import_ds_timemeanbins or
            to functions_pp.time_mean_periods.
        df_splits : pd.DataFrame, optional
            See class_RGCPD. The default is using the df_splits that was used
            for calculating the correlation map.

        Returns
        -------
        None.

        '''
        # self = rg.list_for_MI[0] ; df_splits = rg.df_splits ; kwrgs_load = rg.kwrgs_load
        name = self.name
        filepath = self.filepath

        # for name, filepath in list_precur_pp: # loop over all variables
            # =============================================================================
            # Unpack non-default arguments
            # =============================================================================
        kwrgs = {'selbox':self.selbox, 'dailytomonths':self.dailytomonths}
        for key, value in kwrgs_load.items():
            if type(value) is list and name in value[1].keys():
                kwrgs[key] = value[1][name]
            elif type(value) is list and name not in value[1].keys():
                kwrgs[key] = value[0] # plugging in default value
            elif hasattr(self, key):
                # Overwrite RGCPD parameters with MI specific parameters
                kwrgs[key] = self.__dict__[key]
            else:
                kwrgs[key] = value
        if self.lag_as_gap: kwrgs['tfreq'] = 1
        self.kwrgs_load = kwrgs.copy()
        #===========================================
        # Precursor field
        #===========================================
        self.precur_arr = functions_pp.import_ds_timemeanbins(filepath, **kwrgs)

        if type(self.lags[0]) == np.ndarray:
            tmp = functions_pp.time_mean_periods
            self.precur_arr = tmp(self.precur_arr, self.lags,
                                  kwrgs_load['start_end_year'])
        return

    def load_and_aggregate_ts(self, df_splits: pd.DataFrame=None):
        if df_splits is None:
            df_splits = self.df_splits
        # =============================================================================
        # Load external timeseries for partial_corr_z
        # =============================================================================
        kwrgs = self.kwrgs_load
        if hasattr(self, 'kwrgs_z') == False: # copy so info remains stored
            self.kwrgs_z = self.kwrgs_func.copy() # first time copy
        if self.func.__name__ == 'parcorr_z':
            if type(self.kwrgs_z['filepath']) is str:
                print('Loading and aggregating {}'.format(self.kwrgs_z['keys_ext']))
                f = find_precursors.import_precur_ts
                self.df_z = f([('z', self.kwrgs_z['filepath'])],
                              df_splits,
                              start_end_date=kwrgs['start_end_date'],
                              start_end_year=kwrgs['start_end_year'],
                              start_end_TVdate=kwrgs['start_end_TVdate'],
                              cols=self.kwrgs_z['keys_ext'],
                              precur_aggr=kwrgs['tfreq'])

                if hasattr(self.df_z.index, 'levels'): # has train-test splits
                    f = functions_pp
                    self.df_z = f.get_df_test(self.df_z.merge(df_splits,
                                                              left_index=True,
                                                              right_index=True)).iloc[:,:1]
                k = list(self.kwrgs_func.keys())
                [self.kwrgs_func.pop(k) for k in k if k in ['filepath','keys_ext']]
                self.kwrgs_func.update({'z':self.df_z}) # overwrite kwrgs_func
                k = [k for k in list(self.kwrgs_z.keys()) if k not in ['filepath','keys_ext']]

                equal_dates = all(np.equal(self.df_z.index,
                                           pd.to_datetime(self.precur_arr.time.values)))
                if equal_dates==False:
                    raise ValueError('Dates of timeseries z not equal to dates of field')
            elif type(self.kwrgs_z['filepath']) is pd.DataFrame:
                self.df_z = self.kwrgs_z['filepath']
                k = list(self.kwrgs_func.keys())
                [self.kwrgs_func.pop(k) for k in k if k in ['filepath','keys_ext']]
                self.kwrgs_func.update({'z':self.df_z}) # overwrite kwrgs_func
        return


    def get_prec_ts(self, precur_aggr=None, kwrgs_load=None): #, outdic_precur #TODO
        # tsCorr is total time series (.shape[0]) and .shape[1] are the correlated regions
        # stacked on top of each other (from lag_min to lag_max)

        n_tot_regs = 0
        splits = self.corr_xr.split
        if hasattr(self, 'prec_labels') == False:
            print(f'{self.name} is not clustered yet')
        else:
            if np.isnan(self.prec_labels.values).all():
                self.ts_corr = np.array(splits.size*[[]])
            else:
                self.ts_corr = calc_ts_wrapper(self,
                                               precur_aggr=precur_aggr,
                                               kwrgs_load=kwrgs_load)
                n_tot_regs += max([self.ts_corr[s].shape[1] for s in range(splits.size)])

        return

    def store_netcdf(self, path: str=None, f_name: str=None, add_hash=True):
        assert hasattr(self, 'corr_xr'), 'No MI map calculated'
        if path is None:
            path = functions_pp.get_download_path()
        hash_str  = uuid.uuid4().hex[:6]
        if f_name is None:
            f_name = '{}_a{}'.format(self._name, self.alpha)
        self.corr_xr.attrs['alpha'] = self.alpha
        self.corr_xr.attrs['FDR_control'] = int(self.FDR_control)
        self.corr_xr['lag'] = ('lag', range(self.lags.shape[0]))
        if 'mask' in self.precur_arr.coords:
                self.precur_arr = self.precur_arr.drop('mask')
        # self.precur_arr.attrs['_tfreq'] = int(self._tfreq)
        if hasattr(self, 'prec_labels'):
            self.prec_labels['lag'] = self.corr_xr['lag'] # must be same
            self.prec_labels.attrs['distance_eps'] = self.distance_eps
            self.prec_labels.attrs['min_area_in_degrees2'] = self.min_area_in_degrees2
            self.prec_labels.attrs['group_lag'] = int(self.group_lag)
            self.prec_labels.attrs['group_split'] = int(self.group_split)
            if f_name is None:
                f_name += '_{}_{}'.format(self.distance_eps,
                                          self.min_area_in_degrees2)

            ds = xr.Dataset({'corr_xr':self.corr_xr,
                             'prec_labels':self.prec_labels,
                             'precur_arr':self.precur_arr})
        else:
            ds = xr.Dataset({'corr_xr':self.corr_xr,
                             'precur_arr':self.precur_arr})
        if add_hash:
            f_name += f'_{hash_str}'
        self.filepath_experiment = os.path.join(path, f_name+ '.nc')
        ds.to_netcdf(self.filepath_experiment)
        print(f'Dataset stored with hash: {hash_str}')

    def load_files(self, path_hashfile=str, hash_str: str=None):
        #%%
        if hash_str is None:
            hash_str = '{}_a{}_{}_{}'.format(self._name, self.alpha,
                                           self.distance_eps,
                                           self.min_area_in_degrees2)
        if path_hashfile is None:
            path_hashfile = functions_pp.get_download_path()
        f_name = None
        for root, dirs, files in os.walk(path_hashfile):
            for file in files:
                if re.findall(f'{hash_str}', file):
                    print(f'Found file {file}')
                    f_name = file
        if f_name is not None:
            filepath = os.path.join(path_hashfile, f_name)
            self.ds = core_pp.import_ds_lazy(filepath)
            self.corr_xr = self.ds['corr_xr']
            self.alpha = self.corr_xr.attrs['alpha']
            self.FDR_control = bool(self.corr_xr.attrs['FDR_control'])
            self.precur_arr = self.ds['precur_arr']
            # self._tfreq = self.precur_arr.attrs['_tfreq']
            if 'prec_labels' in self.ds.variables.keys():
                self.prec_labels = self.ds['prec_labels']
                self.distance_eps = self.prec_labels.attrs['distance_eps']
                self.min_area_in_degrees2 = self.prec_labels.attrs['min_area_in_degrees2']
                self.group_lag = bool(self.prec_labels.attrs['group_lag'])
                self.group_split = bool(self.prec_labels.attrs['group_split'])
            loaded = True
        else:
            print('No file that matches the hash_str or instance settings in '
                  f'folder {path_hashfile}')
            loaded = False
        return loaded


        #%%
def check_NaNs(field, ts):
    '''
    Return shortened timeseries of both field and ts if a few NaNs are detected
    at boundary due to large lag. At boundary time-axis, large lags
    often result in NaNs due to missing data.
    Removing timesteps from timeseries if
    1. Entire field is filled with NaNs
    2. Number of timesteps are less than a single year
       of datapoints.
    '''
    t = functions_pp.get_oneyr(field).size # threshold NaNs allowed.
    field = np.reshape(field.values, (field.shape[0],-1))
    i = 0 ; # check NaNs in first year
    if bool(np.isnan(field[i]).all()):
        i+=1
        while bool(np.isnan(field[i]).all()):
            i+=1
            if i > t:
                raise ValueError('More NaNs detected then # of datapoints in '
                                 'single year')
    j = -1 ; # check NaNs in last year
    if bool(np.isnan(field[j]).all()):
        j-=1
        while bool(np.isnan(field[j]).all()):
            j-=1
            if j < t:
                raise ValueError('More NaNs detected then # of datapoints in '
                                 'single year')
    else:
        j = field.shape[0]
    return field[i:j], ts[i:j]


def corr_map(field, ts):
    """
    This function calculates the correlation coefficent r and
    the pvalue p for each grid-point of field vs response-variable ts
    If more then a single year of NaNs is detected, a NaN will
    be returned, otherwise corr is calculated over non-NaN values.

    """
    # if more then one year is filled with NaNs -> no corr value calculated.
    field, ts = check_NaNs(field, ts)
    x = np.ma.zeros(field.shape[1])
    corr_vals = np.array(x)
    pvals = np.array(x)

    fieldnans = np.array([np.isnan(field[:,i]).any() for i in range(x.size)])
    nonans_gc = np.arange(0, fieldnans.size)[fieldnans==False]

    for i in nonans_gc:
        corr_vals[i], pvals[i] = scipy.stats.pearsonr(ts,field[:,i])
    # restore original nans
    corr_vals[fieldnans] = np.nan
    # correlation map and pvalue at each grid-point:

    return corr_vals, pvals

def parcorr_map_time(field: xr.DataArray, ts: np.ndarray, lag_y=0, lag_x=0):
    '''
    Only works for subseasonal data (more then 1 datapoint per year).
    Lag must be >= 1. Warning!!! what about gap between years and shifting data

    Parameters
    ----------
    field : xr.DataArray
        (time, lat, lon) field.
    ts : np.ndarray
        Target timeseries.
    lag : int, optional
        DESCRIPTION. The default is 1.
    target : TYPE, optional
        DESCRIPTION. The default is True.
    precursor : TYPE, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    corr_vals : np.ndarray
    pvals : np.ndarray

    '''
    # field = precur_train.sel(time=dates_lag) ; ts = RV_ts.values.squeeze()

    if type(lag_y) is int:
        lag_y = [lag_y]
    if type(lag_x) is int:
        lag_x = [lag_x]

    max_lag = max(max(lag_y), max(lag_x))
    assert max_lag>0, 'lag_x or lag_y must be >= 1'
    # if more then one year is filled with NaNs -> no corr value calculated.
    field, ts = check_NaNs(field, ts)
    x = np.ma.zeros(field.shape[1])
    corr_vals = np.array(x)
    pvals = np.array(x)

    fieldnans = np.array([np.isnan(field[:,i]).any() for i in range(x.size)])
    nonans_gc = np.arange(0, fieldnans.size)[fieldnans==False]


    if max(lag_y) > 0:
        zy = [np.expand_dims(ts[max_lag-l:-l], axis=1) for l in lag_y if l != 0]
        zy = np.concatenate(zy, axis=1)

    y = np.expand_dims(ts[max_lag:], axis=1)
    for i in nonans_gc:
        cond_ind_test = ParCorr()
        if max(lag_x) > 0:
            zx = [np.expand_dims(field[max_lag-l:-l, i], axis=1) for l in lag_x if l != 0]
            zx = np.concatenate(zx, axis=1)
        if max(lag_x) > 0 and max(lag_y) > 0: # both zy and zx defined
            z = np.concatenate((zy,zx), axis=1)
        elif max(lag_x) > 0 and max(lag_y) == 0: # only zx defined
            z = zx
        elif max(lag_x) == 0 and max(lag_y) > 0:
            z = zy
        field_i = np.expand_dims(field[max_lag:,i], axis=1)
        a, b = cond_ind_test.run_test_raw(field_i, y, z)
        corr_vals[i] = a
        pvals[i] = b
    # restore original nans
    corr_vals[fieldnans] = np.nan
    return corr_vals, pvals

def parcorr_z(field: xr.DataArray, ts: np.ndarray, z: pd.DataFrame, lag_z: int=0):
    '''
    Regress out influence of 1-d timeseries z. if lag_z==0, dates of z will match
    dates of field. Note, lag_z >= 1 probably only makes sense when using
    subseasonal data (more then 1 value per year).

    Parameters
    ----------
    field : xr.DataArray
        (time, lat, lon) field.
    ts : np.ndarray
        Target timeseries.
    z : pd.DataFrame
        1-d timeseries.

    Returns
    -------
    corr_vals : np.ndarray
    pvals : np.ndarray

    '''
    if type(lag_z) is int:
        lag_z = [lag_z]

    max_lag = max(lag_z)
    # if more then one year is filled with NaNs -> no corr value calculated.
    dates = pd.to_datetime(field.time.values)
    field, ts = check_NaNs(field, ts)
    x = np.ma.zeros(field.shape[1])
    corr_vals = np.array(x)
    pvals = np.array(x)
    fieldnans = np.array([np.isnan(field[:,i]).any() for i in range(x.size)])
    nonans_gc = np.arange(0, fieldnans.size)[fieldnans==False]

    # ts = np.expand_dims(ts[:], axis=1)
    # adjust to shape (samples, dimension) and remove first datapoints if
    # lag_z != 0.
    y = np.expand_dims(ts[max_lag:], axis=1)
    if len(z.values.squeeze().shape)==1:
        z = np.expand_dims(z.loc[dates].values.squeeze(), axis=1)
    else:
        z = z.loc[dates].values.squeeze() # lag_z shifted wrt precursor dates

    list_z = []
    if 0 in lag_z:
        list_z = [z[max_lag:]]

    if max(lag_z) > 0:
        [list_z.append(z[max_lag-l:-l]) for l in lag_z if l != 0]
        z = np.concatenate(list_z, axis=1)

    # if lag_z >= 1:
        # z = z[:-lag_z] # last values are 'removed'
    for i in nonans_gc:
        cond_ind_test = ParCorr()
        field_i = np.expand_dims(field[max_lag:,i], axis=1)
        a, b = cond_ind_test.run_test_raw(field_i, y, z)
        corr_vals[i] = a
        pvals[i] = b
    # restore original nans
    corr_vals[fieldnans] = np.nan
    return corr_vals, pvals

def pp_calc_ts(precur, precur_aggr=None, kwrgs_load: dict=None,
                      force_reload: bool=False, lags: list=None):
    '''
    Pre-process for calculating timeseries of precursor regions or pattern.
    '''
    #%%
    corr_xr         = precur.corr_xr
    prec_labels     = precur.prec_labels

    if lags is not None:
        lags        = np.array(lags) # ensure lag is np.ndarray
        corr_xr     = corr_xr.sel(lag=lags).copy()
        prec_labels = prec_labels.sel(lag=lags).copy()
    else:
        lags        = prec_labels.lag.values
    dates           = pd.to_datetime(precur.precur_arr.time.values)
    oneyr = functions_pp.get_oneyr(dates)
    if oneyr.size == 1: # single val per year precursor
        tfreq = 365
    else:
        tfreq = (oneyr[1] - oneyr[0]).days


    if precur_aggr is None and force_reload==False:
        precur_arr = precur.precur_arr
    else:
        if precur_aggr is not None:
            precur.tfreq = precur_aggr
        precur.load_and_aggregate_precur(kwrgs_load.copy())
        precur_arr = precur.precur_arr

    if type(precur.lags[0]) is np.ndarray and precur_aggr is None:
        precur.period_means_array = True
    else:
        precur.period_means_array = False

    if precur_arr.shape[-2:] != corr_xr.shape[-2:]:
        print('shape loaded precur_arr != corr map, matching coords')
        corr_xr, prec_labels = functions_pp.match_coords_xarrays(precur_arr,
                                          *[corr_xr, prec_labels])
    #%%
    return precur_arr, corr_xr, prec_labels

# def loop_get_spatcov(precur, precur_aggr=None, kwrgs_load: dict=None,
#                      force_reload: bool=False, lags: list=None):
#     '''
#     Calculate spatial covariance between significantly correlating gridcells
#     and observed (time, lat, lon) data.
#     '''
#     #%%

#     precur_arr, corr_xr, prec_labels = pp_calc_ts(precur, precur_aggr,
#                                                   kwrgs_load,
#                                                   force_reload, lags)

#     lags        = prec_labels.lag.values
#     precur.area_grid = find_precursors.get_area(precur_arr)
#     splits          = corr_xr.split
#     use_sign_pattern = precur.use_sign_pattern



#     ts_sp = np.zeros( (splits.size), dtype=object)
#     for s in splits:
#         ts_list = np.zeros( (lags.size), dtype=list )
#         track_names = []
#         for il,lag in enumerate(lags):

#             # if lag represents aggregation period:
#             if type(precur.lags[il]) is np.ndarray and precur_aggr is None:
#                 precur_arr = precur.precur_arr.sel(lag=il)

#             corr_vals = corr_xr.sel(split=s).isel(lag=il)
#             mask = prec_labels.sel(split=s).isel(lag=il)
#             pattern = corr_vals.where(~np.isnan(mask))
#             if use_sign_pattern == True:
#                 pattern = np.sign(pattern)
#             if np.isnan(pattern.values).all():
#                 # no regions of this variable and split
#                 nants = np.zeros( (precur_arr.time.size, 1) )
#                 nants[:] = np.nan
#                 ts_list[il] = nants
#                 pass
#             else:
#                 xrts = find_precursors.calc_spatcov(precur_arr, pattern,
#                                                     area_wght=True)
#                 ts_list[il] = xrts.values[:,None]
#             track_names.append(f'{lag}..0..{precur.name}' + '_sp')

#         # concatenate timeseries all of lags
#         tsCorr = np.concatenate(tuple(ts_list), axis = 1)

#         dates = pd.to_datetime(precur_arr.time.values)
#         ts_sp[s] = pd.DataFrame(tsCorr,
#                                 index=dates,
#                                 columns=track_names)
#     # df_sp = pd.concat(list(ts_sp), keys=range(splits.size))
#     #%%
#     return ts_sp

def single_split_calc_spatcov(precur, precur_arr: np.ndarray, corr: np.ndarray,
                              labels: np.ndarray, a_wghts: np.ndarray,
                              lags: np.ndarray, use_sign_pattern: bool):
    ts_list = np.zeros( (lags.size), dtype=list )
    track_names = []
    for il,lag in enumerate(lags):

        # if lag represents aggregation period:
        if precur.period_means_array == True:
            precur_arr = precur.precur_arr.sel(lag=il)

        pattern = np.copy(corr[il]) # copy to fix ValueError: assignment destination is read-only
        mask = labels[il]
        pattern[np.isnan(mask)] = np.nan
        if use_sign_pattern == True:
            pattern = np.sign(pattern)
        if np.isnan(pattern).all():
            # no regions of this variable and split
            nants = np.zeros( (precur_arr.shape[0], 1) )
            nants[:] = np.nan
            ts_list[il] = nants
            pass
        else:
            xrts = find_precursors.calc_spatcov(precur_arr, pattern,
                                                area_wght=a_wghts)
            ts_list[il] = xrts[:,None]
        track_names.append(f'{lag}..0..{precur.name}' + '_sp')
    return ts_list, track_names


def single_split_spatial_mean_regions(precur, precur_arr: np.ndarray,
                                      corr: np.ndarray,
                                      labels: np.ndarray, a_wghts: np.ndarray,
                                      lags: np.ndarray,
                                      use_coef_wghts: bool):
    '''
    precur : class_BivariateMI

    precur_arr : np.ndarray
        of shape (time, lat, lon). If lags define period_means;
        of shape (lag, time, lat, lon).
    corr : np.ndarray
        if shape (lag, lat, lon).
    labels : np.ndarray
        of shape (lag, lat, lon).
    a_wghts : np.ndarray
        if shape (lat, lon).
    use_coef_wghts : bool
        Use correlation coefficient as weights for spatial mean.

    Returns
    -------
    ts_list : list of splits with numpy timeseries
    '''
    ts_list = np.zeros( (lags.size), dtype=list )
    track_names = []
    for l_idx, lag in enumerate(lags):
        labels_lag = labels[l_idx]

        # if lag represents aggregation period:
        if precur.period_means_array == True:
            precur_arr = precur.precur_arr[:,l_idx].values

        regions_for_ts = list(np.unique(labels_lag[~np.isnan(labels_lag)]))

        if use_coef_wghts:
            coef_wghts = abs(corr[l_idx]) / abs(np.nanmax(corr[l_idx]))
            wghts = a_wghts * coef_wghts # area & corr. value weighted
        else:
            wghts = a_wghts

        # this array will be the time series for each feature
        ts_regions_lag_i = np.zeros((precur_arr.shape[0], len(regions_for_ts)))

        # track sign of eacht region

        # calculate area-weighted mean over features
        for r in regions_for_ts:
            idx = regions_for_ts.index(r)
            # start with empty lonlat array
            B = np.zeros(labels_lag.shape)
            # Mask everything except region of interest
            B[labels_lag == r] = 1
            # Calculates how values inside region vary over time
            ts = np.nanmean(precur_arr[:,B==1] * wghts[B==1], axis =1)

            # check for nans
            if ts[np.isnan(ts)].size !=0:
                print(ts)
                perc_nans = ts[np.isnan(ts)].size / ts.size
                if perc_nans == 1:
                    # all NaNs
                    print(f'All timesteps were NaNs for split'
                        f' for region {r} at lag {lag}')

                else:
                    print(f'{perc_nans} NaNs for split'
                        f' for region {r} at lag {lag}')

            track_names.append(f'{lag}..{int(r)}..{precur.name}')

            ts_regions_lag_i[:,idx] = ts
            # get sign of region
            # sign_ts_regions[idx] = np.sign(np.mean(corr.isel(lag=l_idx).values[B==1]))

        ts_list[l_idx] = ts_regions_lag_i

    return ts_list, track_names

def calc_ts_wrapper(precur, precur_aggr=None, kwrgs_load: dict=None,
                    force_reload: bool=False, lags: list=None):
    '''
    Wrapper for calculating 1-d spatial mean timeseries per precursor region
    or a timeseries of the spatial pattern (only significantly corr. gridcells).

    Parameters
    ----------
    precur : class_BivariateMI instance
    precur_aggr : int, optional
        If None, same precur_arr is used as for the correlation maps.
    kwrgs_load : dict, optional
        kwrgs to load in timeseries. See functions_pp.import_ds_timemeanbins or
        functions_pp.time_mean_period. The default is None.
    force_reload : bool, optional
        Force reload a different precursor array (precur_arr). The default is
        False.

    Returns
    -------
    ts_corr : TYPE
        DESCRIPTION.

    '''
    #%%
    # precur=rg.list_for_MI[0];precur_aggr=None;kwrgs_load=None;force_reload=False;lags=None
    # start_time  = time()
    precur_arr, corr_xr, prec_labels = pp_calc_ts(precur, precur_aggr,
                                                  kwrgs_load,
                                                  force_reload, lags)
    lags        = prec_labels.lag.values
    use_coef_wghts  = precur.use_coef_wghts
    a_wghts         = precur.area_grid / precur.area_grid.mean()
    splits          = corr_xr.split.values
    dates = pd.to_datetime(precur_arr.time.values)



    if precur.calc_ts == 'pattern cov':
        kwrgs = {'use_sign_pattern':precur.use_sign_pattern}
        _f = single_split_calc_spatcov
    elif precur.calc_ts == 'region mean':
        kwrgs = {'use_coef_wghts':precur.use_coef_wghts}
        _f = single_split_spatial_mean_regions


    def splits_spatial_mean_regions(_f,
                                    splits: np.ndarray,
                                    precur, precur_arr: np.ndarray,
                                    corr_np: np.ndarray,
                                    labels_np: np.ndarray, a_wghts: np.ndarray,
                                    lags: np.ndarray,
                                    use_coef_wghts: bool):
        '''
        Wrapper to divide calculating a number of splits per core, instead of
        assigning each split to a seperate worker (high overhead).
        '''
        for s in splits:
            corr = corr_np[s]
            labels = labels_np[s]
            output[s] = _f(precur, precur_arr, corr, labels, a_wghts, lags,
                           **kwrgs)
        return output


    precur_arr = precur_arr.values
    corr_np = corr_xr.values
    labels_np = prec_labels.values
    output = {}
    if precur.n_cpu == 1:
        output = splits_spatial_mean_regions(_f, splits, precur, precur_arr,
                                             corr_np, labels_np,
                                             a_wghts, lags, use_coef_wghts)

    elif precur.n_cpu > 1:
        futures = []
        for _s in np.array_split(splits, precur.n_cpu):
            futures.append(delayed(splits_spatial_mean_regions)(_f, _s,
                                                                precur,
                                                                precur_arr,
                                                                corr_np,
                                                                labels_np,
                                                                a_wghts, lags,
                                                                use_coef_wghts))
        futures = Parallel(n_jobs=precur.n_cpu, backend='loky')(futures)
        [output.update(d) for d in futures]


    ts_corr = np.zeros( (splits.size), dtype=object)
    for s in range(splits.size):
        ts_list, track_names = output[s] # list of ts at different lags
        tsCorr = np.concatenate(tuple(ts_list), axis = 1)
        df_tscorr = pd.DataFrame(tsCorr,
                                 index=dates,
                                columns=track_names)
        df_tscorr.name = str(s)
        if any(df_tscorr.isna().values.flatten()):
            print('Warnning: nans detected')
        ts_corr[s] = df_tscorr

    # print(f'End time: {time() - start_time} seconds')

    #%%
    return ts_corr