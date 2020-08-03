import os, sys, inspect, time
from pathlib import Path
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
import multiprocessing as mp 
from multiprocessing import Process

path_data = os.path.join(main_dir, 'data')
current_analysis_path = os.path.join(main_dir, 'Jier_analysis/Data')
ar_data_path = os.path.join(main_dir, 'Jier_analysis/Fitted/AR/AR_data')

rg_sst  = wa.generate_rgcpd()
rg_wind = wa.generate_rgcpd(prec_path='z500hpa_1979-2018_1_12_daily_2.5deg.nc')
rg_sm = wa.generate_rgcpd(prec_path='sm2_1979-2018_1_12_daily_1.0deg.nc')

rg_sst_obj = wa.create_rgcpd_obj(rg=rg_sst, precur_aggr=1)
rg_sm_obj = wa.create_rgcpd_obj(rg=rg_sm, precur_aggr=1)
rg_wind_obj = wa.create_rgcpd_obj(rg=rg_wind,precur_aggr=1)

rg_sst_data = wa.setup_wavelets_rgdata(rg=rg_sst_obj)
rg_wind_data = wa.setup_wavelets_rgdata(rg=rg_wind_obj)
rg_sm_data = wa.setup_wavelets_rgdata(rg=rg_sm_obj)
(rg_data_sm, rg_index_sm),  (wave, mode) = rg_sm_data
(rg_data_sst, rg_index_sst), _  = rg_sst_data
(rg_data_zh, rg_index_zh), _ = rg_wind_data


def to_csv(rg_data, prec='sst_', prec_aggr='1'):
    print('[INFO] Starting writing csv file... ')
    cols = rg_data.columns.tolist()
    csv_path = os.path.join(current_analysis_path ,prec[:-1])
    Path(csv_path).mkdir(parents=True, exist_ok=True )
    for i in range(len(cols)):
        rg_data[cols[i]].to_csv(os.path.join(csv_path, prec+str(cols[i]+'_'+prec_aggr+'.csv') ), header=[cols[i]])
    print('[INFO] Done writing csv file of columns')

def stationarity_test(rg_data):
    print(f'[INFO] Stationary test on the following columns {rg_data.columns.tolist()}\n')
    return rg_data.apply(lambda serie: sd.stationarity_test(serie.values), axis=0)

def evaluate_data_ar(rg_data, title, path=ar_data_path, prec_aggr='1'):
    print('[INFO] Evaluating AR data to save...')
    cols = rg_data.columns.tolist()
    ar_path = os.path.join(path , title)
    Path(ar_path).mkdir(parents=True, exist_ok=True )
    for _, col in enumerate(cols):
        const_, ar_ = sd.evaluate_data_ar(rg_data[col], col=col)
        np.savez(os.path.join(ar_path, 'ar_'+title+'_'+col+'_c_'+prec_aggr+'.npz'), const=const_, ar=ar_)
    print('[INFO] Evaluation successfull \n')

if __name__ == "__main__":

    for data, prec in zip([rg_data_sst, rg_data_sm, rg_data_zh],['sst_', 'sm_', 'zh_']):
        to_csv(data, prec=prec)
    for data in [rg_data_sst, rg_data_sm, rg_data_zh]:
        stationarity_test(data)

    # Process does not run any faster
    # for data, title in zip([rg_data_sst, rg_data_sm, rg_data_zh],['sst', 'sm', 'zh']):
    #     p = Process(target=evaluate_data_ar, args=(data, title,))
    #     p.start()
    #     p.join()

    p = None
    if mp.cpu_count() >= 4:
        p = mp.Pool(mp.cpu_count()-2)
    else:
        p = mp.Pool(mp.cpu_count()-1)
    p.starmap(evaluate_data_ar,zip([rg_data_sst, rg_data_sm, rg_data_zh],['sst', 'sm', 'zh']) )
    p.close()
    p.join()
