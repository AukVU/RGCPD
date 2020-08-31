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
import investigate_ts as inv
prec_paths= [' ', 'z500hpa_1979-2018_1_12_daily_2.5deg.nc','sm2_1979-2018_1_12_daily_1.0deg.nc' ]

rg = inv.generate_rgcpd_default(doc=prec_paths[0])
rg_obj_aggr =  inv.generate_rgcpd_object_prec_aggr(rg=rg, precur_aggr=1)
rg_data, rg_index , wave, mode = inv.generate_rgcpd_data(rg_obj_aggr=rg_obj_aggr)

#    target = 
# p = mp.Pool(mp.cpu_count()-1)
# answer  = p.starmap_async(extract_ar_data,zip([rg_data_, rg_data_],[target_, precursor]))
# result_3ts, result_prec1 =  answer.get()
# p.close()
# p.join()
# const_ts, ar_ts = result_3ts
# const_p1, ar_p1 = result_prec1

cols =  rg_data.columns.tolist()
filter_  = np.load(os.path.join(main_dir, 'Jier_analysis/la8_filter.npz'))
filter_bank = filter_['filter_bank']
la8 = wv.Wavelet(name="LA8", filter_bank=filter_bank)
la8.orthogonal = True
la8.biorthogonal = True 
db4 = wv.Wavelet('db4')
sym8 = wv.Wavelet('sym8')
db4_normalized = wv.Wavelet(
    'db4_normalized',
    filter_bank=[np.asarray(f)/np.sqrt(2) for f in db4.filter_bank]
)

db4_normalized.orthogonal = True
db4_normalized.biorthogonal = True
wave = sym8

gammas = np.arange(0, 1.1, 0.1)
nus = np.arange(0, 1.1, 0.1)
thetas = np.arange(0, 1.1 , 0.1)
# for gamma in gammas:
tests = ['avg', 'std', 'var']
elements = ['prec', 'target', 'dep', 'mci']
poly_stats ={key:{key: [] for  key in tests} for key in elements}
# prec_tests ={key:{key: [] for key in tests} for key in elements}

for col in cols[1:]:
    for nu in nus:
    # for-loop iteratie for consistency
        for it in np.arange(0, 11):
            ar_ts, const_ts = inv.extract_ar_data(rg_data, cols[0])
            ar, const = inv.extract_ar_data(rg_data, col) 

            poly_prec = inv.polynomial_fit_turbulance(ar=ar, col=col,  sigma=np.std(rg_data[col].values), rg_data=rg_data, const=const, theta=1, nu=nu)
            poly_ts = inv.polynomial_fit(ar=ar_ts, rg_data=rg_data, col=cols[0], sigma=np.std(rg_data[cols[0]].values), const=const_ts, dependance=False)
            poly_dep = inv.polynomial_fit(ar=ar_ts, rg_data=rg_data, col=cols[0], sigma=np.std(rg_data[cols[0]].values), const=const_ts, gamma=1, dependance=True, x1=poly_prec)

            poly_stats['prec']['std'].append(np.std(poly_prec))
            poly_stats['prec']['avg'].append(np.average(poly_prec))
            poly_stats['prec']['var'].append(np.var(poly_prec))

            poly_stats['target']['std'].append(np.std(poly_ts))
            poly_stats['target']['avg'].append(np.average(poly_ts))
            poly_stats['target']['var'].append(np.var(poly_ts))


            poly_stats['dep']['std'].append(np.std(poly_dep))
            poly_stats['dep']['avg'].append(np.average(poly_dep))
            poly_stats['dep']['var'].append(np.var(poly_dep))

            result_var_target = inv.wavelet_variance_levels(data=pd.Series(poly_ts, index=rg_index), wavelet=wave, mode=mode, levels=9)
            result_var_prec_dep = inv.wavelet_variance_levels(data=pd.Series(poly_dep, index=rg_index),  wavelet=wave, mode=mode,  levels=9)
            result_var_prec = inv.wavelet_variance_levels(data=pd.Series(poly_prec, index=rg_index),  wavelet=wave, mode=mode,  levels=9)

            _, prec_cD = inv.create_wavelet_details(poly_prec, index_poly=rg_index, level=15, wave=wave, mode=mode , debug=False)
            _, target_cD = inv.create_wavelet_details(poly_dep, index_poly=rg_index, level=15, wave=wave, mode=mode , debug=False)

            target_prec_lag  = inv.get_mci_coeffs_lag(details_prec=prec_cD, details_target=target_cD, index=rg_index, rg_obj=rg_obj_aggr, debug=False)
            poly_stats['mci']['std'].append(np.std(target_prec_lag))
            poly_stats['mci']['avg'].append(np.average(target_prec_lag))
            poly_stats['mci']['var'].append(np.var(target_prec_lag))

            
            if it % 5 == 0:

                inv.plot_mci_prediction(detail_prec=prec_cD, prec_lag=target_prec_lag, title=f'Dependance with {str(np.round(nu, 2))} variation with precursor {col} on {str(it)} iteration', savefig=True)
                
                inv.plot_wavelet_variance(var_result=result_var_target, title=cols[0], savefig=True)
                inv.plot_wavelet_variance(var_result=result_var_prec, title=col, savefig=True)
                inv.plot_wavelet_variance(var_result=result_var_prec_dep, title='dependance', savefig=True)

                inv.display_polynome(poly=poly_dep, ar_list=[ar_ts[0], ar_ts[1], 1.0], rg_data=rg_data, col=cols[0], title=f'target {cols[0]} and precursor {col} with nu {str(np.round(nu, 2))} and {1} gamma', save_fig=True, dependance=True)
                inv.display_polynome(poly=poly_ts, ar_list=ar_ts, rg_data=rg_data, col=cols[0], title=f'AR fit on {cols[0]} target with {str(it)} iteration', save_fig=True, dependance=False)
                inv.display_polynome(poly=poly_prec, ar_list=ar, rg_data=rg_data, col=col, title=f'AR fit on {col} precursor with {str(nu)} variation on {str(it)} iteration ', save_fig=True, dependance=False)


# TODO PLOT ANALYSIS FOR SENSITIVITY OF SIMULATION
inv.display_sensitivity(tests=poly_stats, subjects=elements[-1] )