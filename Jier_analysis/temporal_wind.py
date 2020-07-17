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
precursor_path = os.path.join(path_data,'z500hpa_1979-2018_1_12_daily_2.5deg.nc')
list_of_name_path = [(target, target_path), 
                    ('wd', precursor_path )]
list_for_MI = [BivariateMI(name='wd', func=BivariateMI.corr_map, 
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
rg.df_data

# %%
rg_data  = rg.df_data[['3ts', '0..1..wd', '0..2..wd','0..3..wd','0..4..wd']]
rg_data = rg_data.rename(columns={'0..1..wd':'prec1', '0..2..wd':'prec2', '0..3..wd':'prec3', '0..4..wd':'prec4'})
rg_index = rg_data.index.levels[1]
prec1 = rg_data['prec1'].values
prec2 = rg_data['prec2'].values
prec3 = rg_data['prec3'].values 
prec4 = rg_data['prec4'].values 
target = rg_data['3ts'].values
wave  = wv.Wavelet('db4')
mode=wv.Modes.periodic

# %%
fig , ax = plt.subplots(4, 1, figsize=(19,8), dpi=90)
rg_data[['prec1', 'prec2', 'prec3', 'prec4']].plot(subplots=True, ax=ax)
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
fig, ax = plt.subplots(wv.dwt_max_level(len(prec1), wave.dec_len), 2, figsize=(19, 8))
fig.suptitle('Using Discrete Wavelet transform', fontsize=14)
ap = rg_data['prec2'].values
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
fig, ax = plt.subplots(wv.dwt_max_level(len(prec1), wave.dec_len), 2, figsize=(19, 8))
fig.suptitle('Using Discrete Wavelet transform', fontsize=14)
ap = rg_data['prec3'].values
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
fig, ax = plt.subplots(wv.dwt_max_level(len(prec1), wave.dec_len), 2, figsize=(19, 8))
fig.suptitle('Using Discrete Wavelet transform', fontsize=14)
ap = rg_data['prec4'].values
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
# Using recursion we obtain all of our approximation coefficients with just dwt
s_prec1 = prec1
cA = []
# wv.dwt_max_level(len(prec1), wave.dec_len)
for i in range(6): # Using recursion to overwrite signal to go level deepeer
   s_prec1, det =  wv.dwt(s_prec1, wave , mode=mode)
   print('Len Sign ', len(s_prec1), 'Lenght detail ', len(det))
   cA.append(s_prec1)

print('Inspecting approximations length')
for i, c in enumerate(cA):
    print(i+1, len(c))

# %%
# wv.dwt_max_level(len(prec1), wave.dec_len)# Using recursion we obtain all of our approximation coefficients with just dwt
s_prec2 = prec2
cA2 = []
# wv.dwt_max_level(len(prec2), wave.dec_len)
for i in range(6): # Using recursion to overwrite signal to go level deepeer
   s_prec2, det =  wv.dwt(s_prec2, wave , mode=mode)
   print('Len Sign ', len(s_prec2), 'Lenght detail ', len(det))
   cA2.append(s_prec2)

print('Inspecting approximations length')
for i, c in enumerate(cA2):
    print(i+1, len(c))

# %%
# Using recursion we obtain all of our approximation coefficients with just dwt
s_prec3 = prec3
cA3 = []
# wv.dwt_max_level(len(prec3), wave.dec_len)
for i in range(6): # Using recursion to overwrite signal to go level deepeer
   s_prec3, det =  wv.dwt(s_prec3, wave , mode=mode)
   print('Len Sign ', len(s_prec3), 'Lenght detail ', len(det))
   cA3.append(s_prec3)

print('Inspecting approximations length')
for i, c in enumerate(cA3):
    print(i+1, len(c))

# %%
# Using recursion we obtain all of our approximation coefficients with just dwt
s_prec4 = prec4
cA4 = []
# wv.dwt_max_level(len(prec4), wave.dec_len)
for i in range(6): # Using recursion to overwrite signal to go level deepeer
   s_prec4, det =  wv.dwt(s_prec4, wave , mode=mode)
   print('Len Sign ', len(s_prec4), 'Lenght detail ', len(det))
   cA4.append(s_prec4)

print('Inspecting approximations length')
for i, c in enumerate(cA4):
    print(i+1, len(c))


# %%
# Using recursion we obtain all of our approximation coefficients with just dwt
s_tar = target
cA_t = []
# wv.dwt_max_level(len(target), wave.dec_len)
for i in range(6): # Using recursion to overwrite signal to go level deepeer
   s_tar, det =  wv.dwt(s_tar, wave , mode=mode)
   print('Len Sign ', len(s_tar), 'Lenght detail ', len(det))
   cA_t.append(s_tar)

print('Inspecting approximations length')
for i, c in enumerate(cA_t):
    print(i+1, len(c))

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
    # rg.PCMCI_plot_graph()

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
    # ts_ca1 = pd.Series(cA[i], index=pd.MultiIndex.from_product(([0], idx_prec)),name='p_1_lvl_'+ str(i)+'_dec')
    ts_ca2= pd.Series(cA2[i], index=pd.MultiIndex.from_product(([0], idx_prec)), name='p_2_lvl_'+str(i)+'_dec')
    ts_tca1 = pd.Series(cA_t[i], index=pd.MultiIndex.from_product(([0],idx_lvl_t)), name='3ts')
    df = pd.concat([ts_tca1, ts_ca2, trainIsTrue, RV_mask], axis=1)
    rg.df_data = df
    rg.PCMCI_df_data()
    rg.PCMCI_get_links()
    rg.df_MCIc
    obj_rgcpd2.append(deepcopy(rg.df_MCIc))
    # rg.PCMCI_plot_graph()

# %%
obj_rgcpd3 = []
for i in range(0,len(cA3)):    
    idx_lvl_t = pd.DatetimeIndex(pd.date_range(rg_index[0] ,end=rg_index[-1], periods=len(cA_t[i]) ).strftime('%Y-%m-%d') )
    idx_prec = pd.DatetimeIndex(pd.date_range(rg_index[0], rg_index[-1], periods=len(cA3[i]) ).strftime('%Y-%m-%d') )
    dates = core_pp.get_subdates(dates=idx_lvl_t, start_end_date=('06-01', '08-31'), start_end_year=None, lpyr=False)
    full_time  = idx_lvl_t
    RV_time  = dates
    RV_mask = pd.Series(np.array([True if d in RV_time else False for d in full_time]), index=pd.MultiIndex.from_product(([0], idx_lvl_t)), name='RV_mask')
    trainIsTrue = pd.Series(np.array([True for _ in range(len(cA_t[i]))]), index=pd.MultiIndex.from_product(([0], idx_lvl_t)), name='TrainIsTrue')
    # ts_ca1 = pd.Series(cA[i], index=pd.MultiIndex.from_product(([0], idx_prec)),name='p_1_lvl_'+ str(i)+'_dec')
    ts_ca3= pd.Series(cA3[i], index=pd.MultiIndex.from_product(([0], idx_prec)), name='p_2_lvl_'+str(i)+'_dec')
    ts_tca1 = pd.Series(cA_t[i], index=pd.MultiIndex.from_product(([0],idx_lvl_t)), name='3ts')
    df = pd.concat([ts_tca1, ts_ca3, trainIsTrue, RV_mask], axis=1)
    rg.df_data = df
    rg.PCMCI_df_data()
    rg.PCMCI_get_links()
    rg.df_MCIc
    obj_rgcpd3.append(deepcopy(rg.df_MCIc))
    # rg.PCMCI_plot_graph()

# %%
obj_rgcpd4 = []
for i in range(0,len(cA4)):    
    idx_lvl_t = pd.DatetimeIndex(pd.date_range(rg_index[0] ,end=rg_index[-1], periods=len(cA_t[i]) ).strftime('%Y-%m-%d') )
    idx_prec = pd.DatetimeIndex(pd.date_range(rg_index[0], rg_index[-1], periods=len(cA4[i]) ).strftime('%Y-%m-%d') )
    dates = core_pp.get_subdates(dates=idx_lvl_t, start_end_date=('06-01', '08-31'), start_end_year=None, lpyr=False)
    full_time  = idx_lvl_t
    RV_time  = dates
    RV_mask = pd.Series(np.array([True if d in RV_time else False for d in full_time]), index=pd.MultiIndex.from_product(([0], idx_lvl_t)), name='RV_mask')
    trainIsTrue = pd.Series(np.array([True for _ in range(len(cA_t[i]))]), index=pd.MultiIndex.from_product(([0], idx_lvl_t)), name='TrainIsTrue')
    # ts_ca1 = pd.Series(cA[i], index=pd.MultiIndex.from_product(([0], idx_prec)),name='p_1_lvl_'+ str(i)+'_dec')
    ts_ca4= pd.Series(cA4[i], index=pd.MultiIndex.from_product(([0], idx_prec)), name='p_2_lvl_'+str(i)+'_dec')
    ts_tca1 = pd.Series(cA_t[i], index=pd.MultiIndex.from_product(([0],idx_lvl_t)), name='3ts')
    df = pd.concat([ts_tca1, ts_ca4, trainIsTrue, RV_mask], axis=1)
    rg.df_data = df
    rg.PCMCI_df_data()
    rg.PCMCI_get_links()
    rg.df_MCIc
    obj_rgcpd4.append(deepcopy(rg.df_MCIc))
    # rg.PCMCI_plot_graph()

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
x_as = np.arange(1, len(cA2)+1)
x_as = np.exp2(x_as)
lag_0_1 = [lags.values[:,0][1] for _, lags in enumerate(obj_rgcpd2)]

plt.figure(figsize=(19,8))
# plt.plot(x_as, lag_0_t, label='target ')
plt.plot(x_as, lag_0_1, label='precrursor 2 ')
plt.xticks(x_as)
plt.title('Relation MCI on scale wavelet on lag 0')
plt.xlabel('Scales in days mean')
plt.ylabel('MCI')
plt.legend(loc=0)
plt.show()

# %%
x_as = np.arange(1, len(cA3)+1)
x_as = np.exp2(x_as)
lag_0_1 = [lags.values[:,0][1] for _, lags in enumerate(obj_rgcpd3)]

plt.figure(figsize=(19,8))
# plt.plot(x_as, lag_0_t, label='target ')
plt.plot(x_as, lag_0_1, label='precrursor 3 ')
plt.xticks(x_as)
plt.title('Relation MCI on scale wavelet on lag 0')
plt.xlabel('Scales in days mean')
plt.ylabel('MCI')
plt.legend(loc=0)
plt.show()

# %%
x_as = np.arange(1, len(cA4)+1)
x_as = np.exp2(x_as)
lag_0_1 = [lags.values[:,0][1] for _, lags in enumerate(obj_rgcpd4)]

plt.figure(figsize=(19,8))
# plt.plot(x_as, lag_0_t, label='target ')
plt.plot(x_as, lag_0_1, label='precrursor 4 ')
plt.xticks(x_as)
plt.title('Relation MCI on scale wavelet on lag 0')
plt.xlabel('Scales in days mean')
plt.ylabel('MCI')
plt.legend(loc=0)
plt.show()

# %%
