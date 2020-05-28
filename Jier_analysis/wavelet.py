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
import itertools as it
from statsmodels.tsa.arima_process import  arma_generate_sample, ArmaProcess
import pywt as wv
from pprint import pprint as pp 
np.random.seed(12345)
current_analysis_path = os.path.join(main_dir, 'Jier_analysis')
# TODO Investigate family of wavelet packages
# TODO Apply wavelet to arma process
# TODO Apply wavelet on actual rgcpd data
# TODO Apply correlation investigation of decomposition with observed series, the original one
# TODO Extract time scale obtained from time serie by plotting


arma_process = ArmaProcess(ar=[0.74485, -0.2297], ma=[0.69816, 0.38910], nobs=500)
# print(arma_process.isstationary, arma_process.isinvertible)
y = arma_process.generate_sample(250)

model = sm.tsa.ARMA(y, (2, 2)).fit(trend='c')
# print(model.summary().tables[1])
db3 = wv.Wavelet('db3')
a3, d1, d2, d3 = wv.wavedec(model.resid, db3, level=3)
# pp(dwt, indent=4)
# print(a3)
# pp([d1, d2, d3], indent=4)
# target = pd.read_csv(os.path.join(current_analysis_path, 'target.csv'), engine='python', index_col=[0,1])
# second_sst = pd.read_csv(os.path.join(current_analysis_path, 'second_sst_prec.csv'), engine='python', index_col=[0, 1])
# first_sst = pd.read_csv(os.path.join(current_analysis_path, 'first_sst_prec.csv'), engine='python', index_col=[0, 1])

