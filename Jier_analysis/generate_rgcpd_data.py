import os, sys, inspect
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
sub_dir = os.path.join(main_dir, 'RGCPD/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(sub_dir)

import pandas as pd 
from RGCPD import RGCPD
from RGCPD import BivariateMI

path_data = os.path.join(main_dir, 'data')
current_analysis_path = os.path.join(main_dir, 'Jier_analysis')
target= 3
target_path = os.path.join(path_data, 'tf5_nc5_dendo_80d77.nc')
precursor_path = os.path.join(path_data,'sst_1979-2018_2.5deg_Pacific.nc')
list_of_name_path = [(target, target_path), 
                    ('sst', precursor_path )]
list_for_MI = [BivariateMI(name='sst', func=BivariateMI.corr_map, 
                          kwrgs_func={'alpha':.0001, 'FDR_control':True}, 
                          distance_eps=700, min_area_in_degrees2=5)]
rg = RGCPD(list_of_name_path=list_of_name_path,
           list_for_MI=list_for_MI,
           path_outmain=os.path.join(main_dir,'data'))
rg.pp_precursors(detrend=True, anomaly=True, selbox=None)
rg.pp_TV()
rg.traintest(method='no_train_test_split')
rg.calc_corr_maps()
rg.cluster_list_MI()
# rg.get_ts_prec(precur_aggr=1)
rg.get_ts_prec()
data = rg.df_data
target_region = data['3ts']
prec_1 = data['0..1..sst']
prec_2 = data['0..2..sst']
target_region.to_csv(os.path.join(current_analysis_path, 'target.csv'), header=['values'])
prec_1.to_csv(os.path.join(current_analysis_path, 'first_sst_prec.csv'), header=['values'])
prec_2.to_csv(os.path.join(current_analysis_path, 'second_sst_prec.csv'), header=['values'])