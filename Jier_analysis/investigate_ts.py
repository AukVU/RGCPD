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

# TODO COMBINE DATA GENERATION FROM RGCPD DATA AND FIT THIS, ESTIMATE APPROXIMATIONS 
# TODO EXTRACT WAVELETS FROM RGCPD DATA 
# TODO FIND A WAY TO INCORPORATE WAVELETS TIME-SCALE REPRESENTATION
# TODO USE ARTIFICIAL DATA COEFFICIENT AND WAVELET TIME SCALE VS RGCPD DATA TO BE EVALUATED WITH PCMCI, SHOW IN PLOTS 