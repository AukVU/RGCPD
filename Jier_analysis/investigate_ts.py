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
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
# TODO USE ARTIFICIAL DATA COEFFICIENT AND WAVELET TIME SCALE VS RGCPD DATA TO BE EVALUATED WITH PCMCI, SHOW IN PLOTS 

current_analysis_path = os.path.join(main_dir, 'Jier_analysis')

target_sst = pd.read_csv(os.path.join(current_analysis_path, 'sst_3ts.csv'), engine='python')
first_sst = pd.read_csv(os.path.join(current_analysis_path, 'sst_prec1.csv'), engine='python')
second_sst = pd.read_csv(os.path.join(current_analysis_path, 'sst_prec2.csv'), engine='python')
# var_sst = np.var(first_sst['values'].values)
ar_sst_t = np.load('ar_sst_t.npy')
ar_sst_p1 = np.load('ar_sst_p1.npy')
ar_sst_p2 = np.load('ar_sst_p2.npy')

target_sm = pd.read_csv(os.path.join(current_analysis_path, 'sm_3ts.csv'), engine='python', index_col=[0,1])
first_sm = pd.read_csv(os.path.join(current_analysis_path, 'sm_prec1.csv'), engine='python',index_col=[0,1])
second_sm = pd.read_csv(os.path.join(current_analysis_path, 'sm_prec2.csv'), engine='python',index_col=[0,1])

# COEFFICIENTS
ar_sm_t = np.load('ar_sm_t.npy')
ar_sm_p1 = np.load('ar_sm_p1.npy')
ar_sm_p2 = np.load('ar_sm_p2.npy')

N_ = len(target_sm)
index = target_sm.index.levels[1]

# CONTAINER
# data = np.zeros((N, 3))

# TODO THIS IS ONLY TO PLOT
# poly_ts, eps_ts = sd.create_polynomial_fit_ar(ar_ts, sigma=np.var(rg_data['3ts'].values), data=rg_data['3ts'].values, const=const_ts)
# poly_p1, eps_p2 = sd.create_polynomial_fit_ar(ar_p1, sigma=np.var(rg_data['prec1'].values), data=rg_data['prec1'].values, const=const_p1)
# poly_p2, eps_p2 = sd.create_polynomial_fit_ar(ar_p2, sigma=np.var(rg_data['prec2'].values), data=rg_data['prec2'].values, const=const_p2)

# NOT CORRECT
# for t in range(0, N):
#     data[t, 0] += ar_sm_t[0]* data[t, 0]
#     data[t, 1] += ar_sm_p1[0] * data[t, 1] + ar_sm_p1[0]*data[t-1, 1]
#     data[t, 2] += ar_sm_p2[0] * data[t, 2] + ar_sm_p2[0]*data[t-1, 2] + ar_sm_p2[0]*data[t-2, 2]
print(np.log(ar_sm_p1[0]), np.log(ar_sm_t[0]), ar_sm_p1[0], ar_sm_t[0])



# np.random.seed(42)     # Fix random seed
links_coeffs = {0: [((0, -1), np.log(ar_sst_t[0]) ), ((1, -1), np.log(ar_sst_p1[0]) )],
                1: [((1, -1), np.log(ar_sst_p1[0]) )]
                }
data, true_parents_neighbors = pp.var_process(links_coeffs, T=N_, verbosity=1)
df = pd.DataFrame(data=data, index=index, columns=['3ts', 'prec1'])
wa.choose_wavelet_signal(data=df['3ts'])
# TODO CONTINUE EQUATIONS
# T, N = data.shape
# var_names = [r'$X^0$', r'$X^1$', ]
# df = pp.DataFrame(data, datatime = index, var_names=var_names)
# print(true_parents_neighbors)
# tp.plot_timeseries(df)
# plt.show()
# df  = pd.concat([target_sst['value'], first_sst['value'], second_sst['value']], axis=1)
# df['prec2'] = df.iloc[:,2].rename(columns={'value':'prec2'})
# df['prec1'] = df.iloc[:,1].rename(columns={'value':'prec1'})
# df['3ts']= df.iloc[:,0].rename(columns={'value':'3ts'})
# df = df.drop(columns=['value', 'value', 'value'], axis=1)
# df.index = index 
# # data = np.log(df).diff().dropna()
# model = VAR(df)
# result = model.fit(maxlags=2, ic='aic')
# print(result.summary())