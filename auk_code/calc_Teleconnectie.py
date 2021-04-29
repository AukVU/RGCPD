# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:23:07 2021

@author: Auk
"""

import os, inspect, sys
# main_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) 
main_dir ='C:/Users/Auk/Documents/GitHub/RGCPD'
print(main_dir)
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering')
if RGCPD_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    
from RGCPD import RGCPD
from RGCPD import BivariateMI
import class_BivariateMI, functions_pp
from IPython.display import Image
import numpy as np

# define input: 
path_test = os.path.join(main_dir, 'data') # path of test data
# format list_of_name_path = [('TVname', 'TVpath'), ('prec_name', 'prec_path')]
list_of_name_path = [(3, os.path.join(path_test, 'C:/Users/Auk/Documents/GitHub/RGCPD/data/tp_world_SPI_12_clustered.nc')),
                    ('sst', os.path.join(path_test,'C:/Users/Auk/Documents/GitHub/RGCPD/data/sst_1950-2020_1_12_monthly_1.0deg.nc'))]

list_for_MI = [BivariateMI(name='sst', func=class_BivariateMI.corr_map, 
                           alpha=.01, FDR_control=True, 
                           lags=np.array([['01-01', '09-30']]), # <- selecting time periods to aggregate
                           distance_eps=700, min_area_in_degrees2=5)]

rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           tfreq=None, # <- seasonal forecasting mode, set tfreq to None! 
           start_end_TVdate=([['10-01', '12-30']]), # <- defining DJF target period 
           path_outmain=os.path.join(main_dir,'data'))

rg.pp_TV(TVdates_aggr=True)
