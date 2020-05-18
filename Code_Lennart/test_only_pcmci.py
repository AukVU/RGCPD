#!/usr/bin/env python
# coding: utf-8

# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

import os, inspect, sys


if sys.platform == 'linux':
    import matplotlib as mpl
    mpl.use('Agg')
user_dir = os.path.expanduser('~')
user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/') 
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    
import numpy as np

print(sys.path)
from RGCPD import RGCPD
from RGCPD import EOF
from class_BivariateMI import BivariateMI
from functions_pp import csv_to_df

local_base_path = "/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD"
local_script_dir = os.path.join(local_base_path, "ERA5" )

old_CPPA = [('sst_CPPA', local_script_dir + '/era5_24-09-19_07hr_lag_0.h5')]
CPPA_s30 = [('sst_CPPAs30', local_script_dir + '/era5_21-01-20_10hr_lag_10_Xzkup1.h5' )]
CPPA_s5  = [('sst_CPPAs5', local_script_dir + '/ERA5_15-02-20_15hr_lag_10_Xzkup1.h5')]



list_of_name_path = [#('test_target', local_base_path + '/Code_Lennart/NC/test.npy'),
                     ('test_target', local_base_path + '/Code_Lennart/NC/test_target2.nc')
]

# # list_for_EOFS = [EOF(name='test_precur', neofs=1, selbox=[-180, 360, -15, 30])]
# list_for_MI   = [BivariateMI(name='test_precur', func='corr', kwrgs_func={'alpha':.05, 'FDR_control':True})]
list_for_MI = []

# start_end_TVdate = ('06-24', '08-22')
start_end_TVdate = None

#start_end_TVdate = ('06-15', '08-31')
# start_end_date = ('1-1', '12-31')
start_end_date = None

output = 'small'
csv_to_df(path=local_base_path + f'/Code_Lennart/results/{output}/time_series/timeseries.csv')

list_import_ts = [('test_precur', local_base_path + f'/Code_Lennart/results/{output}/time_series/timeseries.h5')]

rg = RGCPD(#list_of_name_path=list_of_name_path,
            list_import_ts=list_import_ts,
            # list_for_MI=list_for_MI,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=10, lags_i=np.array([1]),
           path_outmain=user_dir+'/ERA5/clustered/output_RGCPD')

#selbox = [None, {'sst':[-180,360,-10,90]}]
selbox = None
#anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False}]
anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False, 'st2':False}]

rg.pp_TV()

kwrgs_events=None
rg.traintest(method='random10', kwrgs_events=kwrgs_events)

rg.get_ts_prec(precur_aggr=None)

rg.PCMCI_df_data(pc_alpha=None, 
                 tau_max=2,
                 max_combinations=2)

rg.PCMCI_get_links(alpha_level=0.1)

rg.PCMCI_plot_graph(s=1)

rg.quick_view_labels()

rg.plot_maps_sum()