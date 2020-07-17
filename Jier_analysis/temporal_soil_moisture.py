# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('matplotlib', 'inline')
import os, sys, inspect, warnings
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
sub_dir = os.path.join(main_dir, 'RGCPD/')
# core_pp = os.path.join(main_dir, 'RGCPD/core')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(sub_dir)
import numpy as np 
from collections import Counter
# import statsmodels.api as sm 
import pandas as pd 
import math
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (20.0, 10.0)
import itertools as it
import pywt as wv
from scipy.fftpack import fft
from copy import deepcopy
# from statsmodels.tsa.arima_process import  arma_generate_sample, ArmaProcess
from pprint import pprint as pp 
from pandas.plotting import register_matplotlib_converters
from RGCPD import RGCPD
from RGCPD import BivariateMI
import core_pp
import plot_signal_decomp
import plot_coeffs
from visualize_cwt import *
register_matplotlib_converters()
np.random.seed(12345)
plt.style.use('seaborn')


# %%
path_data = os.path.join(main_dir, 'data')
# temporal_data  = os.path.join(main_dir, 'data/Ftemporal')
current_analysis_path = os.path.join(main_dir, 'Jier_analysis')
target= 3
target_path = os.path.join(path_data, 'tf5_nc5_dendo_80d77.nc')
precursor_path = os.path.join(path_data,'sm2_1979-2018_1_12_daily_1.0deg.nc')
list_of_name_path = [(target, target_path), 
                    ('sm', precursor_path )]
list_for_MI = [BivariateMI(name='sm', func=BivariateMI.corr_map, 
                          kwrgs_func={'alpha':.0001, 'FDR_control':True}, 
                          distance_eps=700, min_area_in_degrees2=5)]
rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           start_end_TVdate=('06-01', '08-31'),
           path_outmain=os.path.join(main_dir,'data'))


# %%
rg.pp_precursors(detrend=True, anomaly=True, selbox=None)
rg.pp_TV()
rg.traintest(method='no_train_test_split')
rg.calc_corr_maps()
rg.cluster_list_MI()
rg.get_ts_prec(precur_aggr=1)


# %%
rg.dates_TV


# %%
rg.PCMCI_df_data()
rg.PCMCI_get_links()
rg.df_MCIc
rg.PCMCI_plot_graph()


# %%
rg_data  = rg.df_data[['3ts', '0..1..sm', '0..2..sm']]
rg_data = rg_data.rename(columns={'0..1..sm':'prec1', '0..2..sm':'prec2'})
rg_index = rg_data.index.levels[1]
prec1 = rg_data['prec1'].values
prec2 = rg_data['prec2'].values
target = rg_data['3ts'].values
wave  = wv.Wavelet('db4')
mode=wv.Modes.periodic


# %%
print(len(prec1), len(prec1)/2)


# %%
plt.figure(figsize=(19,8), dpi=120)
plt.plot(rg_index, prec1)
plt.show()


# %%
plt.figure(figsize=(19,8), dpi=120)
plt.plot(rg_index, prec2)
plt.show()


# %%

fig, ax = plt.subplots(wv.dwt_max_level(len(prec1), wave.dec_len), 2, figsize=(19, 8))
fig.suptitle('Using Discrete Wavelet transform', fontsize=14)
ap = rg_data['prec1'].values
for i in range(wv.dwt_max_level(len(prec1), wave.dec_len)):
   ap, det =  wv.dwt(ap, 'db4')
   ax[i, 0].plot(ap, 'r')
   ax[i, 1].plot(det, 'g')
   ax[i, 0].set_ylabel('Level {}'.format(i + 1), fontsize=9, rotation=90)
   if i == 0:
        ax[i, 0].set_title('Approximation coeffs', fontsize=14)
        ax[i, 1].set_title('Details coeffs', fontsize=14)
plt.tight_layout()
plt.show()


# %%

fig, ax = plt.subplots(wv.dwt_max_level(len(prec2), wave.dec_len), 2, figsize=(19, 8))
fig.suptitle('Using Discrete Wavelet transform', fontsize=14)
ap = rg_data['prec2'].values
for i in range(wv.dwt_max_level(len(prec2), wave.dec_len)):
   ap, det =  wv.dwt(ap, 'db4')
   ax[i, 0].plot(ap, 'r')
   ax[i, 1].plot(det, 'g')
   ax[i, 0].set_ylabel('Level {}'.format(i + 1), fontsize=9, rotation=90)
   if i == 0:
        ax[i, 0].set_title('Approximation coeffs', fontsize=14)
        ax[i, 1].set_title('Details coeffs', fontsize=14)
plt.tight_layout()
plt.show()


# %%
2**6


# %%
# Using recursion we obtain all of our approximation coefficients with just dwt
s_prec1 = prec1
cA = []
# wv.dwt_max_level(len(s_prec1), wave.dec_len)
for i in range(6): # Using recursion to overwrite signal to go level deepeer
   s_prec1, det =  wv.dwt(s_prec1, wave , mode=mode)
   print('Len Sign ', len(s_prec1), 'Lenght detail ', len(det))
   cA.append(s_prec1)

print('Inspecting approximations length')
for i, c in enumerate(cA):
    print(i+1, len(c))


# %%
s_target = target
cA_t= []
# wv.dwt_max_level(len(s_target), wave.dec_len)
for i in range(6): # Using recursion to overwrite signal to go level deepeer
   s_target, det =  wv.dwt(s_target, wave , mode=mode)
   print('Len Sign ', len(s_target), 'Lenght detail ', len(det))
   cA_t.append(s_target)

print('Inspecting approximations length')
for i, c in enumerate(cA_t):
    print(i, len(c))


# %%
obj_rgcpd = []
for i in range(0,len(cA)):    
    idx_lvl_t = pd.DatetimeIndex(pd.date_range(rg_index[0] ,end=rg_index[-1], periods=len(cA_t[i]) ).strftime('%Y-%m-%d') )
    idx_prec = pd.DatetimeIndex(pd.date_range(rg_index[0], rg_index[-1], periods=len(cA[i]) ).strftime('%Y-%m-%d') )
    dates = core_pp.get_subdates(dates=idx_lvl_t, start_end_date=('06-01', '08-31'), start_end_year=None, lpyr=False)
    full_time  = idx_lvl_t
    RV_time  = dates
    RV_mask = pd.Series(np.array([True if d in RV_time else False for d in full_time]), index=pd.MultiIndex.from_product(([0], idx_lvl_t)), name='RV_mask')
    trainIsTrue = pd.Series(np.array([True for _ in range(len(cA_t[i]))]), index=pd.MultiIndex.from_product(([0], idx_lvl_t)), name='TrainIsTrue')
    ts_ca1 = pd.Series(cA[i], index=pd.MultiIndex.from_product(([0], idx_prec)),name='p_1_lvl_'+ str(i)+'_dec')
    # ts_ca2= pd.Series(cA_2[i], index=pd.MultiIndex.from_product(([0], idx_prec)), name='p_2_lvl_'+str(i)+'_dec')
    ts_tca1 = pd.Series(cA_t[i], index=pd.MultiIndex.from_product(([0],idx_lvl_t)), name='3ts')
    df = pd.concat([ts_tca1, ts_ca1, trainIsTrue, RV_mask], axis=1)
    rg.df_data = df
    rg.PCMCI_df_data()
    rg.PCMCI_get_links()
    rg.df_MCIc
    obj_rgcpd.append(deepcopy(rg.df_MCIc))
    rg.PCMCI_plot_graph()


# %%
np.exp2(x_as)


# %%
x_as = np.arange(1, len(cA)+1)
x_as = np.exp2(x_as)
lag_0_1 = [lags.values[:,0][1] for _, lags in enumerate(obj_rgcpd)]

plt.figure(figsize=(19,8))
# plt.plot(x_as, lag_0_t, label='target ')
plt.plot(x_as, lag_0_1, label='precrursor 1 ')
plt.xticks(x_as)
plt.title('Relation MCI on scale wavelet on lag 0')
plt.xlabel('Scales in days mean')
plt.ylabel('MCI')
plt.legend(loc=0)
plt.show()


# %%
# Using recursion we obtain all of our approximation coefficients with just dwt
s_prec2 = prec2
cA2 = []
# wv.dwt_max_level(len(s_prec2), wave.dec_len)
for i in range(6): # Using recursion to overwrite signal to go level deepeer
   s_prec2, det =  wv.dwt(s_prec2, wave , mode=mode)
   print('Len Sign ', len(s_prec2), 'Lenght detail ', len(det))
   cA2.append(s_prec2)

print('Inspecting approximations length')
for i, c in enumerate(cA2):
    print(i, len(c))


# %%
obj_rgcpd2 = []
for i in range(0,len(cA2)):    
    idx_lvl_t = pd.DatetimeIndex(pd.date_range(rg_index[0] ,end=rg_index[-1], periods=len(cA_t[i]) ).strftime('%Y-%m-%d') )
    idx_prec = pd.DatetimeIndex(pd.date_range(rg_index[0], rg_index[-1], periods=len(cA2[i]) ).strftime('%Y-%m-%d') )
    dates = core_pp.get_subdates(dates=idx_lvl_t, start_end_date=('06-01', '08-31'), start_end_year=None, lpyr=False)

    full_time  = idx_lvl_t
    RV_time  = dates
    RV_mask = pd.Series(np.array([True if d in RV_time else False for d in full_time]), index=pd.MultiIndex.from_product(([0], idx_lvl_t)), name='RV_mask')

    trainIsTrue = pd.Series(np.array([True for _ in range(len(cA_t[i]))]), index=pd.MultiIndex.from_product(([0], idx_lvl_t)), name='TrainIsTrue')
    ts_ca2= pd.Series(cA2[i], index=pd.MultiIndex.from_product(([0], idx_prec)), name='p_2_lvl_'+str(i)+'_dec')
    ts_tca1 = pd.Series(cA_t[i], index=pd.MultiIndex.from_product(([0],idx_lvl_t)), name='3ts')

    # df = pd.concat([ts_tca1.reset_index(drop=True), ts_ca2.reset_index(drop=True), trainIsTrue.reset_index(drop=True), RV_mask.reset_index(drop=True)], axis=1)
    df  = pd.concat([ts_tca1, ts_ca2, trainIsTrue, RV_mask], axis=1)

    rg.df_data = df
    rg.PCMCI_df_data()
    rg.PCMCI_get_links()
    rg.df_MCIc
    obj_rgcpd2.append(deepcopy(rg.df_MCIc))
    rg.PCMCI_plot_graph()


# %%
x_as = np.arange(1, len(cA2)+1)
x_as = np.exp2(x_as)
lag_0_1 = [lags.values[:,0][1] for _, lags in enumerate(obj_rgcpd2)]

plt.figure(figsize=(19,8))
# plt.plot(x_as, lag_0_t, label='target ')
plt.xticks(x_as)
plt.plot(x_as, lag_0_1, label='precrursor 2 ')
plt.title('Relation MCI on scale wavelet on lag 0')
plt.xlabel('Scales in days mean')
plt.ylabel('MCI')
plt.legend(loc=0)
plt.show()


# %%


