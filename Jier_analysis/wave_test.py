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
import pywt as wv
from pprint import pprint as pp 
import pygasp.dwt as dwt 
# np.random.seed(12345)
current_analysis_path = os.path.join(main_dir, 'Jier_analysis')

rows, col = 1000, 1
# data = np.random.rand(rows, col)
# t_idx = pd.date_range('1980-01-01', periods=rows, freq='MS')
# df = pd.DataFrame(data=data, columns=['value'], index=t_idx)
# df.plot()
testing.N, testing.K = rows, col 
df = testing.makeTimeDataFrame(freq='MS')
families = ['haar', 'db1', 'db2', 'db3', 'db4', 'db5']
levels = [1, 2, 3, 4, 5, 6, 7, 8]
#calculate the number of necessary decompositions
# NbrDecomp= pywt.dwt_max_level(len(D), db1)+1
# wave = wv.Wavelet('db3')
# coefs= wv.wavedec(df_t.values, wave, level=3)
#series - input data
#wave   - current wavelet

def shanon_entropy(signal):
    if len(signal) == 1:
        return len(signal)
    else:
        energy_scale = np.sum(np.abs(signal), axis=1)
        t_energy = np.sum(energy_scale)
        prob = energy_scale / t_energy
        w_entropy = -np.sum(prob * np.log(prob))
        return w_entropy


def energy_at_level(coeffs, level):
    return np.sqrt(np.sum(np.array(coeffs[-level]) ** 2)) / len(coeffs[-level])

def choose_family(signal, families, levels):
    entropy_score = []
    for wave in families:
        for level in levels:
            coefs = wv.wavedec(signal, wave, level)
            if len(coefs) > 1:
                for i in range(len(coefs)):
                    print('More dan one')
                    print(energy_at_level(coefs[i],level), level, wave)
                    entropy_score.append((shanon_entropy(coefs[i]), wave))
            else:
                print(energy_at_level(coefs[0], level), level, wave)
                entropy_score.append((shanon_entropy(coefs[0]), wave))
        
    print(entropy_score)
    entropy_score = np.array(entropy_score)

choose_family(df.values, families, levels)
    

