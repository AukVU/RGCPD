# -*- coding: utf-8 -*-
import sys 
from  df_ana import *
from creating_time_series import *
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
plt.style.use('ggplot')
import sklearn
import pandas as pd
import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb

np.random.seed(42)     # Fix random seed
# links_coeffs = {0: [((0, -1), 0.7), ((1, -1), -0.8)],
#                 1: [((1, -1), 0.8), ((3, -1), 0.8)],
#                 2: [((2, -1), 0.5), ((1, -2), 0.5), ((3, -3), 0.6)],
#                 3: [((3, -1), 0.4)],
#                 }
# T = 1000     # time series length
# data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)
# T, N = data.shape

# # Initialize dataframe object, specify time axis and variable names
# var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']
# dataframe = pp.DataFrame(data, 
#                          datatime = np.arange(len(data)), 
#                          var_names=var_names)
# def mult(x):
#     return x * 2
# a = pd.Series(data=np.random.randint(10, size=10), name='A-Serie')
# b = pd.Series(data=np.random.randint(10, size=10), name='B-Serie')
# df = pd.concat([a, b], axis=1).astype('float32')

# res = loop_df_ana(df, mult )
# !USE APPLY FUNCTION instead of loop_df_ana
# test = df.apply(mult)

#! USE BUILT-IN PLOT FUNCTIONS
# res = loop_df(df, mult)

# EXPECT time serie dataframe for autocorrelation
# settings = {}
# settings['N'] = 5
# settings['nx'], settings['ny'], settings['T'] = 30, settings['N'] * 30, 5000
# settings['spatial_covariance'] = 0.3
# settings['random_modes'] = False
# settings['noise_use_mean'] = False
# settings['transient'] = 200
# settings['spatial_factor'] = 0.1
# settings['plot_points'] = 500


# links_coeffs = 'model2'

# ts = create_time_series(settings, links_coeffs,  verbose=True,
#                                             plot_modes=False,
#                                             plot_timeseries=True,
#                                             draw_network=True)
# res = autocorr_sm(ts)
