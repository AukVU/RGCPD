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
import argparse 
# TODO CREATE DATASTRUCTURE TO CAPTURE VARIATION ON NU ITERATIONS OF A GIVEN PRECURSOR 
# TODO WRITE A FUNCTION TO GENERALLY PLOT SENSITIVITY OF EITHER ITERATIONS OR NU  ITERATIONS OR PER COLUMNS STATS 
parser = argparse.ArgumentParser()
parser.add_argument('-it', '--iteration', type=int, required=True, help='The number of iterations to run the experiments')
parser.add_argument('-w', '--wavelet', type=str, default='sym4', help='The name of the wavelets to work with DWT')
parser.add_argument('-v', '--variable', type=str, default='nu', help='The variable to run experiment on, nu, gamma, thetas')
args = parser.parse_args()



prec_paths_total= [' ', 'z500hpa_1979-2018_1_12_daily_2.5deg.nc','sm2_1979-2018_1_12_daily_1.0deg.nc' ]
prec_paths = [' ']
rg = inv.generate_rgcpd_default(doc=prec_paths[0])
rg_obj_aggr =  inv.generate_rgcpd_object_prec_aggr(rg=rg, precur_aggr=1)
rg_data, rg_index , wave, mode = inv.generate_rgcpd_data(rg_obj_aggr=rg_obj_aggr)

dataset = 'sst' if " " in prec_paths else prec_paths[:4]
dataset +='_'+args.variable

#    target = 
# p = mp.Pool(mp.cpu_count()-1)
# answer  = p.starmap_async(extract_ar_data,zip([rg_data_, rg_data_],[target_, precursor]))
# result_3ts, result_prec1 =  answer.get()
# p.close()
# p.join()
# const_ts, ar_ts = result_3ts
# const_p1, ar_p1 = result_prec1
# db4_normalized = wv.Wavelet(
#     'db4_normalized',
#     filter_bank=[np.asarray(f)/np.sqrt(2) for f in db4.filter_bank]
# )

# db4_normalized.orthogonal = True
# db4_normalized.biorthogonal = True
# la8 = wa.create_least_asymmetric_filter()
# db4 = wv.Wavelet('db4')

cols =  rg_data.columns.tolist()

wave = wv.Wavelet(args.wavelet)
gammas = np.arange(0.1, 1.1, 0.1)
nus = np.arange(0.1, 1.1, 0.1)


tests = ['avg', 'perc', 'var']
elements = ['wvar', 'dep', 'mci']

end_iter = args.iteration
poly_stats_iter ={key:{key: [] for  key in tests} for key in elements} 

daily_mean_peak = np.empty((args.iteration, len(nus)))
exp_col =  {key: daily_mean_peak for key in cols[1:]}
level_decomposition = None
for i, col in enumerate(cols[1:]):
    
    for nu in nus: # all xis [0.1, .... 1]
        poly_stats ={key:[] for key in elements} 
        # for-loop iteratie for consistency
        for it in np.arange(0, end_iter): #all experiments M = end_iter+1
            ar_ts, const_ts = inv.extract_ar_data(rg_data, cols[0])
            ar, const = inv.extract_ar_data(rg_data, col) 

            poly_prec = inv.polynomial_fit_turbulance(ar=ar, col=col,  sigma=np.std(rg_data[col].values), rg_data=rg_data, const=const,  nu=nu)
            poly_ts = inv.polynomial_fit(ar=ar_ts, rg_data=rg_data, col=cols[0], sigma=np.std(rg_data[cols[0]].values), const=const_ts, dependance=False)
            poly_dep = inv.polynomial_fit(ar=ar_ts, rg_data=rg_data, col=cols[0], sigma=np.std(rg_data[cols[0]].values), const=const_ts, gamma=1, dependance=True, x1=poly_prec)


            poly_stats['dep'].append(poly_dep)
            poly_stats_iter['dep']['perc'].append(np.percentile(poly_dep, 95))
            poly_stats_iter['dep']['avg'].append(np.average(poly_dep))
            poly_stats_iter['dep']['var'].append(np.var(poly_dep))   

            # result_var_target = inv.wavelet_variance_levels(data=pd.Series(poly_ts, index=rg_index), wavelet=wave, mode=mode, levels=9) 
            # result_var_prec = inv.wavelet_variance_levels(data=pd.Series(poly_prec, index=rg_index),  wavelet=wave, mode=mode,  levels=9)
            result_var_prec_dep = inv.wavelet_variance_levels(data=pd.Series(poly_dep, index=rg_index),  wavelet=wave, mode=mode,  levels=9)


            poly_stats['wvar'].append(result_var_prec_dep)
            poly_stats_iter['wvar']['perc'].append(np.percentile(result_var_prec_dep, 95))
            poly_stats_iter['wvar']['avg'].append(np.average(result_var_prec_dep))
            poly_stats_iter['wvar']['var'].append(np.var(result_var_prec_dep)) 

            _, prec_cD, lvl = inv.create_wavelet_details(poly_prec, index_poly=rg_index, level=15, wave=wave, mode=mode , debug=False)
            _, target_cD, lvl2 = inv.create_wavelet_details(poly_dep, index_poly=rg_index, level=15, wave=wave, mode=mode , debug=False)

            target_prec_lag  = inv.get_mci_coeffs_lag(details_prec=prec_cD, details_target=target_cD, index=rg_index, rg_obj=rg_obj_aggr, debug=False)
      

            poly_stats['mci'].append(target_prec_lag)
            poly_stats_iter['mci']['perc'].append(np.percentile(target_prec_lag, 95))
            poly_stats_iter['mci']['avg'].append(np.average(target_prec_lag))
            poly_stats_iter['mci']['var'].append(np.var(target_prec_lag))

            day_means_scales = np.arange(1, len(prec_cD)+1)
            day_means_scales = np.exp2(day_means_scales).astype(np.float64)
            exp_col[col][it,np.argwhere(nus==nu)[0][0]] = day_means_scales[np.argmax(target_prec_lag)] #  This will be the holy grail for boxplot
            if lvl == lvl2:
                level_decomposition = lvl 
                

 
            
            if it in [0, end_iter-1//2, end_iter-1] :

                inv.plot_mci_prediction(detail_prec=prec_cD, prec_lag=target_prec_lag, title=f'MCI with nu {str(np.round(nu, 2))} variation with precursor {col} \n on {str(it)} iteration',path=dataset, savefig=True)
                inv.plot_wavelet_variance(var_result=result_var_prec_dep, title=f'Wavelet scale2scale variance on {str(it)} variation \n with nu {str(np.round(nu, 2))}', path=dataset, savefig=True)
                inv.display_polynome(poly=poly_dep, ar_list=[ar_ts[0], ar_ts[1], 1.0], rg_data=rg_data, col=cols[0], title=f'Target dependance {cols[0]} and precursor {col} \n with nu {str(np.round(nu, 2))} and {1} gamma', path=dataset, save_fig=True, dependance=True)
                
            #     # inv.plot_wavelet_variance(var_result=result_var_target, title=cols[0] +'_iteration'+ str(it), savefig=True)
            #     # inv.plot_wavelet_variance(var_result=result_var_prec, title=col +'_ iteration'+ str(it), savefig=True)
            #     # inv.display_polynome(poly=poly_ts, ar_list=ar_ts, rg_data=rg_data, col=cols[0], title=f'AR fit on {cols[0]} target with {str(it)} iteration', save_fig=True, dependance=False)
            #     inv.display_polynome(poly=poly_prec, ar_list=ar, rg_data=rg_data, col=col, title=f'AR fit on {col} precursor with {str(nu)} variation on {str(it)} iteration ', save_fig=True, dependance=False)  

            if it == end_iter - 1 :
                inv.plot_mci_prediction(detail_prec=prec_cD, prec_lag=poly_stats['mci'], title=f'MCI ensemble with nu {str(np.round(nu, 2))} variation with precursor {col} \n on {str(end_iter)} total iterations',path=dataset, savefig=True, ensemble=True)
                inv.display_sensitivity_in_iter(tests=poly_stats_iter, subjects=elements[-1], title='Totale run of '+str(end_iter)+' iteration of nu for of experiment '+elements[-1].upper()+' '+col, path=dataset, savefig=True)
                inv.display_sensitivity_in_iter(tests=poly_stats_iter, subjects=elements[-2],title='Totale run of '+str(end_iter)+' iteration of nu for of experiment '+elements[-2].upper()+' '+col, path=dataset, savefig=True)
                inv.display_sensitivity_in_iter(tests=poly_stats_iter, subjects=elements[-3],title='Totale run of '+str(end_iter)+' iteration of nu for of experiment '+elements[-3].upper()+' '+col, path=dataset, savefig=True)



inv.display_boxplot_sensitivity(tests=exp_col, subject=cols[1], sens_vars=nus, depth=level_decomposition, path=dataset+'box_plots', title=r'Evaluation of sensitivity $\nu$ '+cols[1], savefig=True)


    

         


                

                    
