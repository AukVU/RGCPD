import os, sys, inspect, warnings
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
sub_dir = os.path.join(main_dir, 'RGCPD/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(sub_dir)
import numpy as np 
import statsmodels.api as sm 
import matplotlib.pyplot as plt 
plt.style.use('seaborn')
import pandas as pd 
import pandas.util.testing as testing
import itertools as it
from statsmodels.tsa.arima_process import  arma_generate_sample, ArmaProcess
import pywt as wv
from pprint import pprint as pp 
import timesynth as ts
import tsfel as ts_extract
import pygasp.dwt as dwt 
import scipy.signal as signal
np.random.seed(12345)
current_analysis_path = os.path.join(main_dir, 'Jier_analysis')
# TODO Investigate family of wavelet packages
# TODO Apply wavelet to arma process
# TODO Apply wavelet on actual rgcpd data
# TODO Apply correlation investigation of decomposition with observed series, the original one
# TODO Extract time scale obtained from time serie by plotting


# arma_process = ArmaProcess(ar=[0.74485, -0.2297], ma=[0.69816, 0.38910], nobs=500)
# # print(arma_process.isstationary, arma_process.isinvertible)
# y = arma_process.generate_sample(250)

# model = sm.tsa.ARMA(y, (2, 2)).fit(trend='c')
# # print(model.summary().tables[1])
# db3 = wv.Wavelet('db3')
# a3, d1, d2, d3 = wv.wavedec(model.resid, db3, level=3)
# pp(dwt, indent=4)
# print(a3)
# pp([d1, d2, d3], indent=4)
# target = pd.read_csv(os.path.join(current_analysis_path, 'target.csv'), engine='python', index_col=[0,1])
# second_sst = pd.read_csv(os.path.join(current_analysis_path, 'second_sst_prec.csv'), engine='python', index_col=[0, 1])
# first_sst = pd.read_csv(os.path.join(current_analysis_path, 'first_sst_prec.csv'), engine='python', index_col=[0, 1])
# time_sampler= ts.TimeSampler(stop_time=20)
# irregular_time_samples = time_sampler.sample_irregular_time(num_points=500, keep_percentage=50)
# white_noise = ts.noise.GaussianNoise(std=0.3)
# sinusoid = ts.signals.Sinusoidal(frequency=0.25)
# time_serie = ts.TimeSeries(sinusoid, noise_generator=white_noise)
# samples, signals, errors = time_serie.sample(irregular_time_samples)
# plt.plot(irregular_time_samples, samples, marker='o')


rows, col = 365, 1
# data = np.random.rand(rows, col)
# t_idx = pd.date_range('1980-01-01', periods=rows, freq='MS')
# df = pd.DataFrame(data=data, columns=['value'], index=t_idx)
# df.plot()
testing.N, testing.K = rows, col 
df_t = testing.makeTimeDataFrame(freq='MS')
# db3 = wv.Wavelet('db3')
# a3= wv.wavedec(df_t.values, db3, level=2)
# d1 = np.array(d1)
# d2 = np.array(d2)
# d3 = np.array(d3)
# print( a3.shape, a3[:,0], a3[:,1], a3[:,2], sep='\n')
# print(a3.shape,d1.shape, d2.shape, d3.shape, sep='\n')
# print(np.array_equal(d1[:, 2], d1[:, 3]), (d1[:, 2] == d1[:, 3]).all(), sep='\n\n')
# print(a3, d1, d2, d3, sep='\n\n')
# pp(d1, indent=4)
# fig, ax = plt.subplots(5, 1, figsize=(14, 8), dpi=130)
# ax[0].plot(a3[:,0], 'red', label='DA')
# ax[0].legend()
# ax[1].plot(d1[:,0], 'green', label='D1')
# ax[1].legend()
# ax[2].plot(d1[:,1], 'blue', label='D2')
# ax[2].legend()
# ax[3].plot(d1[:,3], 'black', label='D3')
# ax[3].legend()
# ax[4].plot(df_t.values, label='Original')
# ax[4].legend()
# plt.show()
# coef = len(a3)
# # print(coef, a3)
# fig, axis = plt.subplots(2*coef+2, 1, figsize=(14, 8), dpi=120)

# for i in range(coef):
#     for j in range(2):
#         axis[3 * i + j].plot(a3[i][:,j])

# # #     axis[i].plot(a3[)
# # plt.tight_layout()
# df_t.plot()
widths = range(1, 5)
cwtmatr, freq = wv.cwt(df_t.values, widths, 'morl')
plt.matshow(cwtmatr)
# plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto', vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

# cfg = ts_extract.get_features_by_domain()

# # X_s = ts_extract.time_series_features_extractor(cfg, samples)
# X_df_n = ts_extract.time_series_features_extractor(cfg, df)
# # X_df_t = ts_extract.time_series_features_extractor(cfg, df_t)
# # print(X_s)
# features_s = ts_extract.correlation_report(X_df_n)
# print(list(features_s))