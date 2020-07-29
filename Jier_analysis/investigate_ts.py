import os, sys, inspect, warnings
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
sub_dir = os.path.join(main_dir, 'RGCPD/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(sub_dir)
import numpy as np 
import statsmodels.api as sm 
from statsmodels.tsa.api import VAR
import matplotlib.pyplot as plt 
plt.style.use('seaborn')
import pandas as pd 
import pywt as wv
from pprint import pprint as pp 
import synthetic_data as sd 
import wave_ana as wa 
from RGCPD import RGCPD
from RGCPD import BivariateMI
import wave_ana as wa 
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
# TODO USE ARTIFICIAL DATA COEFFICIENT AND WAVELET TIME SCALE VS RGCPD DATA TO BE EVALUATED WITH PCMCI, SHOW IN PLOTS 

current_analysis_path = os.path.join(main_dir, 'Jier_analysis')


path_data = os.path.join(main_dir, 'data')
current_analysis_path = os.path.join(main_dir, 'Jier_analysis')

rg_sst  = wa.generate_rgcpd()
rg_wind = wa.generate_rgcpd(prec_path='z500hpa_1979-2018_1_12_daily_2.5deg.nc')
rg_sm = wa.generate_rgcpd(prec_path='sm2_1979-2018_1_12_daily_1.0deg.nc')

rg_sst_obj = wa.create_rgcpd_obj(rg=rg_sst, precur_aggr=10)
rg_sm_obj = wa.create_rgcpd_obj(rg=rg_sm, precur_aggr=10)
rg_wind_obj = wa.create_rgcpd_obj(rg=rg_wind, precur_aggr=10)

rg_sst_data = wa.setup_wavelets_rgdata(rg=rg_sst_obj)
rg_wind_data = wa.setup_wavelets_rgdata(rg=rg_wind_obj)
rg_sm_data = wa.setup_wavelets_rgdata(rg=rg_sm_obj)
(rg_data_sm, rg_index_sm), (precursor_list_sm, target_sm), (wave, mode) = rg_sm_data
(rg_data_sst, rg_index_sst), (precursor_list_sst, target_sst), const = rg_sst_data
(rg_data_zh, rg_index_zh), (precursor_list_zh, target_zh), const = rg_wind_data

sd.stationarity_test(serie=rg_data_sst['3ts'].values)
print("sst target")
sd.stationarity_test(serie=rg_data_sst['prec1'].values)
print("sst prec1")
sd.stationarity_test(serie=rg_data_sst['prec2'].values)
print("sst prec2")

sd.stationarity_test(serie=rg_data_sm['3ts'].values)
print("sm target")
sd.stationarity_test(serie=rg_data_sm['prec1'].values)
print("sm prec1")
sd.stationarity_test(serie=rg_data_sm['prec2'].values)
print("sm prec2")

sd.stationarity_test(serie=rg_data_zh['3ts'].values)
print("zh target")
sd.stationarity_test(serie=rg_data_zh['prec1'].values)
print("zh prec1")
sd.stationarity_test(serie=rg_data_zh['prec2'].values)
print("zh prec2")
sd.stationarity_test(serie=rg_data_zh['prec3'].values)
print("zh prec3")
sd.stationarity_test(serie=rg_data_zh['prec4'].values)
print("zh prec4")

const_sst_ts, ar_sst_ts  = sd.evaluate_data_ar(data=rg_data_sst['3ts'].values)
const_sst_p1, ar_sst_p1 = sd.evaluate_data_ar(data=rg_data_sst['prec1'].values) 
const_sst_p2, ar_sst_p2 = sd.evaluate_data_ar(data=rg_data_sst['prec2'].values)

const_sm_ts, ar_sm_ts  = sd.evaluate_data_ar(data=rg_data_sm['3ts'].values)
const_sm_p1, ar_sm_p1 = sd.evaluate_data_ar(data=rg_data_sm['prec1'].values)
const_sm_p2, ar_sm_p2 = sd.evaluate_data_ar(data=rg_data_sm['prec2'].values)

const_zh_ts, ar_zh_ts  = sd.evaluate_data_ar(data=rg_data_zh['3ts'].values)
const_zh_p1, ar_zh_p1 = sd.evaluate_data_ar(data=rg_data_zh['prec1'].values)
const_zh_p2, ar_zh_p2 = sd.evaluate_data_ar(data=rg_data_zh['prec2'].values)
const_zh_p3, ar_zh_p3 = sd.evaluate_data_ar(data=rg_data_zh['prec3'].values)
const_zh_p4, ar_zh_p4 = sd.evaluate_data_ar(data=rg_data_zh['prec4'].values)

# target_sst = pd.read_csv(os.path.join(current_analysis_path, 'sst_3ts.csv'), engine='python', index_col=[0,1])
# first_sst = pd.read_csv(os.path.join(current_analysis_path, 'sst_prec1.csv'), engine='python', index_col=[0,1])
# second_sst = pd.read_csv(os.path.join(current_analysis_path, 'sst_prec2.csv'), engine='python', index_col=[0, 1])
# # var_sst = np.var(first_sst['values'].values)
# ar_sst_t = np.load('ar_sst_t.npy')
# ar_sst_p1 = np.load('ar_sst_p1.npy')
# ar_sst_p2 = np.load('ar_sst_p2.npy')

# target_sm = pd.read_csv(os.path.join(current_analysis_path, 'sm_3ts.csv'), engine='python', index_col=[0,1])
# first_sm = pd.read_csv(os.path.join(current_analysis_path, 'sm_prec1.csv'), engine='python',index_col=[0,1])
# second_sm = pd.read_csv(os.path.join(current_analysis_path, 'sm_prec2.csv'), engine='python',index_col=[0,1])

# # COEFFICIENTS
# ar_t = np.load('ar_sm_t_c.npz')
# ar_p1 = np.load('ar_sm_p1_c.npz')
# ar_p2 = np.load('ar_sm_p2_c.npz')
# ar_sm_t, const_sm_t = ar_t['x'] , ar_t['y']
# ar_sm_p1, const_sm_p1 = ar_p1['x'], ar_p1['y']
# ar_sm_p2, cosnt_sm_p2 = ar_p2['x'] ,ar_p2['y']

# # N_ = len(target_sm)
# # index = target_sm.index.levels[1]
# # dep_t = sd.create_polynomial_fit_ar_depence(alpha=ar_sm_t, beta=ar_sm_p1, sigma=np.var(target_sm), data=target_sm, const=const_sm_t)
# # poly_p1 = sd.create_polynomial_fit_ar(ar=ar_sm_p1, sigma=np.var(first_sm), data=first_sm, const=const_sm_p1)
# # plt.plot(dep_t, 'r')
# # plt.plot(poly_p1, 'k')
# # plt.show()
# # # CONTAINER

# ar_t = np.load('ar_sst_t_c.npz')
# ar_p1 = np.load('ar_sst_p1_c.npz')
# ar_p2 = np.load('ar_sst_p2_c.npz')
# ar_sst_t, const_sst_t = ar_t['x'] , ar_t['y']
# ar_sst_p1, const_sst_p1 = ar_p1['x'], ar_p1['y']
# ar_sst_p2, cosnt_sst_p2 = ar_p2['x'] ,ar_p2['y']
poly_p1= sd.create_polynomial_fit_ar(ar_sst_p1, sigma=np.var(rg_data_sst['prec1'].values), data=rg_data_sst['prec1'], const=const_sst_p1)
poly_ts = sd.create_polynomial_fit_ar(ar=ar_sst_ts, sigma=np.var(rg_data_sst['3ts'].values), data=target_sst, const=const_sst_ts )
poly_dep = sd.create_polynomial_fit_ar_depence(x0=poly_p1, alpha=ar_sst_p1, x1=poly_ts, beta=ar_sst_ts, data=target_sst)
sd.display_poly_data_ar(simul_data=poly_dep, ar=[ar_sst_ts[0], ar_sst_p1[0]], signal=rg_data_sst['3ts'])
plt.show()
poly_p1_sm= sd.create_polynomial_fit_ar(ar_sm_p1, sigma=np.var(rg_data_sm['prec1'].values), data=rg_data_sm['prec1'], const=const_sm_p1)
poly_ts_sm = sd.create_polynomial_fit_ar(ar=ar_sm_ts, sigma=np.var(rg_data_sm['3ts'].values), data=target_sm, const=const_sm_ts )
poly_dep_sm = sd.create_polynomial_fit_ar_depence(x0=poly_p1_sm, alpha=ar_sm_p1, x1=poly_ts_sm, beta=ar_sm_ts, data=target_sm)
sd.display_poly_data_ar(simul_data=poly_dep_sm, ar=[ar_sm_ts[0], ar_sm_p1[0]], signal=rg_data_sm['3ts'])
plt.show()
# # poly_p1, eps_p2 = sd.create_polynomial_fit_ar(ar_p1, sigma=np.var(rg_data['prec1'].values), data=rg_data['prec1'].values, const=const_p1)
# # poly_p2, eps_p2 = sd.create_polynomial_fit_ar(ar_p2, sigma=np.var(rg_data['prec2'].values), data=rg_data['prec2'].values, const=const_p2)
# # np.random.seed(42)
# # plt.plot(poly_ts)
# # plt.show()
sst_p1_cA = wa.create_low_freq_components(pd.Series(poly_p1, index=rg_index_sst), level=6, wave=wave, mode=mode , debug=True)
sst_dep_cA = wa.create_low_freq_components(pd.Series(poly_dep, index=rg_index_sst), level=6, wave=wave, mode=mode, debug=True)

sm_p1_cA = wa.create_low_freq_components(pd.Series(poly_p1_sm, index=rg_index_sm), level=6, wave=wave, mode=mode, debug=True)
sm_dep_cA = wa.create_low_freq_components(pd.Series(poly_dep_sm, index=rg_index_sm), level=6, wave=wave, mode=mode, debug=True)

sst_obj_rgcpd = wa.create_mci_coeff(cA=sst_p1_cA, cA_t=sst_dep_cA, rg_index=rg_index_sm, rg=rg_sst_obj, debug=False)
sm_obj_rgcpd = wa.create_mci_coeff(cA=sm_p1_cA, cA_t=sm_dep_cA, rg_index=rg_index_sm, rg=rg_sm_obj, debug=False)

_, sst_prec_lag = wa.extract_mci_lags(to_clean_mci_df=sst_obj_rgcpd, lag=0)
_, sm_prec_lag = wa.extract_mci_lags(to_clean_mci_df=sm_obj_rgcpd, lag=0)

wa.plot_mci_pred_relation(cA=sst_p1_cA, prec_lag=sst_prec_lag, savefig=False)
wa.plot_mci_pred_relation(cA=sm_p1_cA, prec_lag=sm_prec_lag, savefig=False)

wa.plot_discr_wave_decomp(data=pd.Series(poly_p1, index=rg_index_sst), wave=wave)
plt.show()
wa.plot_discr_wave_decomp(data=pd.Series(poly_ts, index=rg_index_sst), wave=wave)
plt.show()
wa.plot_discr_wave_decomp(data=pd.Series(poly_dep, index=rg_index_sst), wave=wave)
plt.show()

wa.plot_discr_wave_decomp(data=pd.Series(poly_p1_sm, index=rg_index_sst), wave=wave)
plt.show()
wa.plot_discr_wave_decomp(data=pd.Series(poly_ts_sm, index=rg_index_sst), wave=wave)
plt.show()
wa.plot_discr_wave_decomp(data=pd.Series(poly_dep_sm, index=rg_index_sst), wave=wave)
plt.show()