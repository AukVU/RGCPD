import os, sys, inspect, time
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
sub_dir = os.path.join(main_dir, 'RGCPD/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(sub_dir)

import pandas as pd 
import numpy as np 
from RGCPD import RGCPD
from RGCPD import BivariateMI
import wave_ana as wa 
import synthetic_data as sd 
from tqdm import tqdm

path_data = os.path.join(main_dir, 'data')
current_analysis_path = os.path.join(main_dir, 'Jier_analysis')

rg_sst  = wa.generate_rgcpd()
rg_wind = wa.generate_rgcpd(prec_path='z500hpa_1979-2018_1_12_daily_2.5deg.nc')
rg_sm = wa.generate_rgcpd(prec_path='sm2_1979-2018_1_12_daily_1.0deg.nc')

rg_sst_obj = wa.create_rgcpd_obj(rg=rg_sst)
rg_sm_obj = wa.create_rgcpd_obj(rg=rg_sm)
rg_wind_obj = wa.create_rgcpd_obj(rg=rg_wind)

rg_sst_data = wa.setup_wavelets_rgdata(rg=rg_sst_obj)
rg_wind_data = wa.setup_wavelets_rgdata(rg=rg_wind_obj)
rg_sm_data = wa.setup_wavelets_rgdata(rg=rg_sm_obj)
(rg_data_sm, rg_index_sm), (precursor_list_sm, target_sm), (wave, mode) = rg_sm_data
(rg_data_sst, rg_index_sst), (precursor_list_sst, target_sst), const = rg_sst_data
(rg_data_zh, rg_index_zh), (precursor_list_zh, target_zh), const = rg_wind_data


def to_csv(rg_data, prec='sst_'):
    cols = rg_data.columns.tolist()
    for i in range(len(cols)):
        rg_data[cols[i]].to_csv(os.path.join(current_analysis_path, prec+str(cols[i]+'.csv') ), header=['value'])



if __name__ == "__main__":

    
    to_csv(rg_data_sst)
    to_csv(rg_data_sm, prec='sm_')
    to_csv(rg_data_zh, prec='zh_')

    sd.stationarity_test(serie=rg_data_sst['3ts'].values)
    print("sst target")
    sd.stationarity_test(serie=rg_data_sst['prec1'].values)
    print("sst prec1")
    sd.stationarity_test(serie=rg_data_sst['prec2'].values)
    print("sst prec2")

    sd.stationarity_test(serie=rg_data_sm['3ts'].values)
    print("sm target")
    sd.stationarity_test(serie=rg_data_sm['prec1'].values)
    print("sm prec1")
    sd.stationarity_test(serie=rg_data_sm['prec2'].values)
    print("sm prec2")

    sd.stationarity_test(serie=rg_data_zh['3ts'].values)
    print("zh target")
    sd.stationarity_test(serie=rg_data_zh['prec1'].values)
    print("zh prec1")
    sd.stationarity_test(serie=rg_data_zh['prec2'].values)
    print("zh prec2")
    sd.stationarity_test(serie=rg_data_zh['prec3'].values)
    print("zh prec3")
    sd.stationarity_test(serie=rg_data_zh['prec4'].values)
    print("zh prec4")

    # # TODO THIS IS TIME CONSUMING PICKLE THIS 
# with tqdm( desc="Writing to file", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
    start = time.time()
    const_ts, ar_ts  = sd.evaluate_data_ar(data=rg_data_sst['3ts'].values)
    print("Elapsed target ", time.time() -start)
    const_p1, ar_p1 = sd.evaluate_data_ar(data=rg_data_sst['prec1'].values) 
    const_p2, ar_p2 = sd.evaluate_data_ar(data=rg_data_sst['prec2'].values)
    
    np.savez('ar_sst_t_c.npz', x=ar_ts, y=const_ts)
    np.savez('ar_sst_p1_c.npz', x=ar_p1, y=const_p1)
    np.savez('ar_sst_p2_c.npz', x=ar_p2, y=const_p2)

    print('Done saving sst ar')
    const_ts, ar_ts  = sd.evaluate_data_ar(data=rg_data_sm['3ts'].values)
    const_p1, ar_p1 = sd.evaluate_data_ar(data=rg_data_sm['prec1'].values)
    const_p2, ar_p2 = sd.evaluate_data_ar(data=rg_data_sm['prec2'].values)

    np.savez('ar_sm_t_c.npz', x=ar_ts, y=const_ts)
    np.savez('ar_sm_p1_c.npz', x=ar_p1, y=const_p1)
    np.savez('ar_sm_p2_c.npz', x=ar_p2,y=const_p2)
    print('Done saving sm ar')
    const_ts, ar_ts  = sd.evaluate_data_ar(data=rg_data_zh['3ts'].values)
    const_p1, ar_p1 = sd.evaluate_data_ar(data=rg_data_zh['prec1'].values)
    const_p2, ar_p2 = sd.evaluate_data_ar(data=rg_data_zh['prec2'].values)
    const_p3, ar_p3 = sd.evaluate_data_ar(data=rg_data_zh['prec3'].values)
    const_p4, ar_p4 = sd.evaluate_data_ar(data=rg_data_zh['prec4'].values)

    np.savez('ar_zh_t_c.npz', x=ar_ts, y=const_ts)
    np.savez('ar_zh_p1_c.npz', x=ar_p1,y=const_p1)
    np.savez('ar_zh_p2_c.npz', x=ar_p2,y=const_p2)
    np.savez('ar_zh_p3_c.npz', x=ar_p3,y=const_p3)
    np.savez('ar_zh_p4_c.npz', x=ar_p4,y=const_p4)
    print('Done saving zh ar')