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
from RGCPD import RGCPD
from RGCPD import BivariateMI
import wave_ana as wa 
import multiprocessing as mp 


# TODO Fix fitting on target it is not right.


def generate_rgcpd_default(default=True, doc=""):
    if default == True:
        return wa.generate_rgcpd()
    else:
        return wa.generate_rgcpd(prec_path=doc)

def generate_rgcpd_object_prec_aggr(rg, precur_aggr):
    return wa.create_rgcpd_obj(rg=rg, precur_aggr=precur_aggr)
   
def generate_rgcpd_data(rg_obj_aggr):
    rg_data_tuples = wa.setup_wavelets_rgdata(rg=rg_obj_aggr)
    (rg_data, rg_index),  (wave, mode) = rg_data_tuples
    return rg_data, rg_index , wave, mode

def verify_stationarity(rg_data, target ):
    print(f"[INFO] Evaluating stationarity of {target}")
    rg_data.apply(lambda x : sd.stationarity_test(x.values), axis=0)

def extract_ar_data(rg_data, col):
    rg_data = rg_data.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    const, ar  = sd.evaluate_data_ar(data=rg_data[col], col = col)
    return const, ar 

def polynomial_fit(ar, rg_data, col, sigma, const, dependance, x1=np.zeros((100)), gamma=0.1):
    return sd.create_polynomial_fit_ar(ar, sigma=sigma, data=rg_data[col], const=const, dependance=dependance, gamma=gamma, x1=x1)

def polynomial_fit_dependance(x0, x1, gamma, data):
    return sd.create_polynomial_fit_ar_depence(x0=x0, x1=x1, gamma=gamma, data=data)

def display_polynome(poly, ar_list, rg_data, col, title,  save_fig, dependance):
    sd.display_poly_data_ar(simul_data=poly, ar=ar_list, signal=rg_data[col], save_fig=save_fig, dep=dependance, title=title)
    

def create_wavelet_details(poly, index_poly, wave, level, mode, debug=True):
    return wa.create_low_freq_components(pd.Series(poly, index=index_poly), level=level, wave=wave, mode=mode , debug=debug)

def  get_mci_coeffs_lag(details_prec, details_target, index, rg_obj, debug=False):
    mci_obj = wa.create_mci_coeff(cA=details_prec, cA_t=details_target, rg_index=index, rg=rg_obj, debug=debug)
    _ , prec_lag = wa.extract_mci_lags(to_clean_mci_df=mci_obj, lag=0)
    return prec_lag

def plot_mci_prediction(detail_prec, prec_lag, savefig=False):
    wa.plot_mci_pred_relation(cA=detail_prec, prec_lag=prec_lag, savefig=savefig)

def display_wavelet_decomposition(poly, index, wave):
    wa.plot_discr_wave_decomp(data=pd.Series(poly, index=index), wave=wave)
    plt.show()



if __name__ == "__main__":
    # rg_wind = wa.generate_rgcpd(prec_path='z500hpa_1979-2018_1_12_daily_2.5deg.nc')
    # rg_sm = wa.generate_rgcpd(prec_path='sm2_1979-2018_1_12_daily_1.0deg.nc')

    # sst_p1_cA = wa.create_low_freq_components(pd.Series(poly_p1, index=rg_index_sst), level=3, wave=wave, mode=mode , debug=True)
    # sst_dep_cA = wa.create_low_freq_components(pd.Series(poly_dep, index=rg_index_sst), level=3, wave=wave, mode=mode, debug=True)
    # sst_obj_rgcpd = wa.create_mci_coeff(cA=sst_p1_cA, cA_t=sst_dep_cA, rg_index=rg_index_sm, rg=rg_sst_obj, debug=False)
    # # sm_obj_rgcpd = wa.create_mci_coeff(cA=sm_p1_cA, cA_t=sm_dep_cA, rg_index=rg_index_sm, rg=rg_sm_obj, debug=False)
    # _, sst_prec_lag = wa.extract_mci_lags(to_clean_mci_df=sst_obj_rgcpd, lag=0)
    # _, sm_prec_lag = wa.extract_mci_lags(to_clean_mci_df=sm_obj_rgcpd, lag=0)
    # wa.plot_mci_pred_relation(cA=sst_p1_cA, prec_lag=sst_prec_lag, savefig=False)
    # wa.plot_mci_pred_relation(cA=sm_p1_cA, prec_lag=sm_prec_lag, savefig=False)

    if(len(sys.argv) <= 2):
        print("Parameter error, this is how you should call this")
        print("python " + sys.argv[0] + "<prec_aggr><gamma><target><precursor><data><precursor_path>")
        sys.exit(1)
    else:   
        prec_aggr = int(sys.argv[1])
        gamma = float(sys.argv[2])
        target = sys.argv[3]
        precursor = sys.argv[4]
        data = sys.argv[5]
        prec_path = sys.argv[6]


    rg = generate_rgcpd_default(default=False, doc=prec_path)
    rg_obj_aggr =  generate_rgcpd_object_prec_aggr(rg=rg, precur_aggr=prec_aggr)
    rg_data, rg_index , wave, mode = generate_rgcpd_data(rg_obj_aggr=rg_obj_aggr)
    verify_stationarity(rg_data=rg_data, target=data )

    
    p = mp.Pool(mp.cpu_count()-1)
    answer  = p.starmap_async(extract_ar_data,zip([rg_data, rg_data],[target, precursor]))
    result_3ts, result_prec1 =  answer.get()
    p.close()
    p.join()
    const_ts, ar_ts = result_3ts
    const_p1, ar_p1 = result_prec1

    poly_p1 = polynomial_fit(ar=ar_p1, rg_data=rg_data, col=precursor, sigma=np.var(rg_data[precursor].values), const=const_p1, dependance=False)
    poly_ts = polynomial_fit(ar=ar_ts, rg_data=rg_data, col=target, sigma=np.var(rg_data[target].values), const=const_ts, dependance=False)
    poly_dep = polynomial_fit_dependance(x0=poly_ts,  gamma=gamma, data=rg_data[target],  x1=poly_p1)

    display_polynome(poly=poly_dep, ar_list=[ar_ts[:2][0], ar_ts[:2][1], gamma], rg_data=rg_data, col=target, title=f'AR fit dependance {gamma} gamma', save_fig=False, dependance=True)
    display_polynome(poly=poly_ts, ar_list=ar_ts, rg_data=rg_data, col=target, title=f'AR fit on {target} target', save_fig=False, dependance=False)
    display_polynome(poly=poly_p1, ar_list=ar_p1, rg_data=rg_data, col=precursor, title=f'AR fit on {precursor} precursor', save_fig=False, dependance=False)
    plt.show()
