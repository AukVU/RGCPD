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
# TODO Apply wavelet on actual rgcpd data, precursor and target
# TODO Apply MCI on different scale with target(not decomposed)
# TODO If strong MCI presents and curve data is visible use fitted data representation of target and precursor to see effect of autocorelation given previously found maximum
# TODO Apply correlation investigation of decomposition with observed series, the original one, similar to next point
# TODO Extract time scale obtained from time serie by filtering on the different scales
 
# To noisy to use 
# data = np.random.rand(rows, col)
# t_idx = pd.date_range('1980-01-01', periods=rows, freq='MS')
# df = pd.DataFrame(data=data, columns=['value'], index=t_idx)
# df.plot()
rows, col = 365, 1
testing.N, testing.K = rows, col 
df_t = testing.makeTimeDataFrame(freq='MS')
db3 = wv.Wavelet('db3')
a3= wv.wavedec(df_t.values, db3, level=2)
pp(a3, indent=4, depth=4)

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