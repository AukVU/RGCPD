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
# ## Initialize RGCPD class
# args:
# - list_of_name_path
# - start_end_TVdate
# 
#         list_of_name_path : list of name, path tuples. 
#         Convention: first entry should be (name, path) of target variable (TV).
#         list_of_name_path = [('TVname', 'TVpath'), ('prec_name1', 'prec_path1')]
#         
#         TV period : tuple of start- and enddate in format ('mm-dd', 'mm-dd')

print(sys.path)
from RGCPD import RGCPD
from RGCPD import EOF
from class_BivariateMI import BivariateMI

local_base_path = "/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD"
local_script_dir = os.path.join(local_base_path, "ERA5" )

old_CPPA = [('sst_CPPA', local_script_dir + '/era5_24-09-19_07hr_lag_0.h5')]
CPPA_s30 = [('sst_CPPAs30', local_script_dir + '/era5_21-01-20_10hr_lag_10_Xzkup1.h5' )]
CPPA_s5  = [('sst_CPPAs5', local_script_dir + '/ERA5_15-02-20_15hr_lag_10_Xzkup1.h5')]

# list_of_name_path = [#('t2mmmax', local_script_dir + '/era5_t2mmax_US_1979-2018_averAggljacc0.25d_tf1_n4__to_t2mmax_US_tf1_selclus4_okt19_Xzkup1.npy'),
#                         # ('sm1', local_script_dir + '/sm1_1979-2018_1_12_daily_1.0deg.nc'),
#                         # ('sm2', local_script_dir + '/sm2_1979-2018_1_12_daily_1.0deg.nc')                     
#                         # ('sm3', local_script_dir + '/sm3_1979-2018_1_12_daily_1.0deg.nc'),  
#                         ('mx2t', '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD/ERA5/clustered/output_RGCPD_dendo_20491.nc'),
#                         ('st2', local_script_dir + '/st2_1979-2018_1_12_daily_1.0deg.nc')
#                         # ('OLR', local_script_dir + '/OLRtrop_1979-2018_1_12_daily_2.5deg.nc')

# #                        ('u500', local_script_dir + '/u500hpa_1979-2018_1_12_daily_2.5deg.nc'),
# #                         ('v200', local_script_dir + '/input_raw/v200hpa_1979-2018_1_12_daily_2.5deg.nc'),
# #                         ('v500', local_script_dir + '/input_raw/v500hpa_1979-2018_1_12_daily_2.5deg.nc'),
# #                        ('sst', local_script_dir + '/sst_1979-2018_1_12_daily_1.0deg.nc'),
# #                        ('sm123', local_script_dir + '/sm_123_1979-2018_1_12_daily_1.0deg.nc')
# ]

# list_for_EOFS = [EOF(name='st2', neofs=1, selbox=[-180, 360, -15, 30])]
# list_for_MI   = [BivariateMI(name='st2', func='corr', kwrgs_func={'alpha':.05, 'FDR_control':True})]



# test = BivariateMI(name='st2', func=BivariateMI.corr_map, kwrgs_func={'alpha':.05, 'FDR_control':True})
# print(test)

list_of_name_path = [#('test_target', local_base_path + '/Code_Lennart/NC/test.npy'),
                     ('test_target', local_base_path + '/Code_Lennart/NC/test_target2.nc'),
                     ('test_precur', local_base_path + '/Code_Lennart/NC/test.nc')
]

list_for_EOFS = [EOF(name='test_precur', neofs=1, selbox=[-180, 360, -15, 30])]
list_for_MI   = [BivariateMI(name='test_precur', func='corr', kwrgs_func={'alpha':.05, 'FDR_control':True})]

start_end_TVdate = ('06-24', '08-22')
start_end_TVdate = ('07-06', '08-11')

#start_end_TVdate = ('06-15', '08-31')
start_end_date = ('1-1', '12-31')

rg = RGCPD(list_of_name_path=list_of_name_path, 
           list_for_EOFS=list_for_EOFS,
           list_for_MI=list_for_MI,
           import_prec_ts=None,
           start_end_TVdate=start_end_TVdate,
           start_end_date=start_end_date,
           tfreq=10, lags_i=np.array([1]),
           path_outmain=user_dir+'/ERA5/clustered/output_RGCPD')

#selbox = [None, {'sst':[-180,360,-10,90]}]
selbox = None
#anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False}]
anomaly = [True, {'sm1':False, 'sm2':False, 'sm3':False, 'st2':False}]

rg.pp_precursors(selbox=selbox, anomaly=anomaly)

rg.pp_TV()

#kwrgs_events={'event_percentile':66}
kwrgs_events=None
rg.traintest(method='random10', kwrgs_events=kwrgs_events)

rg.calc_corr_maps()

rg.cluster_list_MI()
print('hoi')

rg.quick_view_labels() 

rg.get_EOFs()

rg.get_ts_prec(precur_aggr=None)

rg.PCMCI_df_data(pc_alpha=None, 
                 tau_max=2,
                 max_combinations=2)

rg.PCMCI_get_links(alpha_level=0.1)

rg.plot_maps_sum()