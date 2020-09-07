import os, sys, inspect, warnings
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
sub_dir = os.path.join(main_dir, 'RGCPD/')
# core_pp = os.path.join(main_dir, 'RGCPD/core')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(sub_dir)
import numpy as np 
from collections import Counter

import pandas as pd 
import math, scipy
import seaborn as sns 
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (20.0, 10.0)
import itertools as it
import pywt as wv
from scipy.fftpack import fft
import scipy.stats as st
from scipy.special import *
from copy import deepcopy
from pprint import pprint as pp 
from pandas.plotting import register_matplotlib_converters
import  statsmodels.stats.api as stats
from RGCPD import RGCPD
from RGCPD import BivariateMI
import core_pp
from plot_signal_decomp import *
import plot_coeffs
from visualize_cwt import *
register_matplotlib_converters()
# np.random.seed(12345)
plt.style.use('seaborn')
from pathlib import Path
current_analysis_path = os.path.join(main_dir, 'Jier_analysis')

# rows, col = 1000, 1
# data = np.random.rand(rows, col)
# t_idx = pd.date_range('1980-01-01', periods=rows, freq='MS')
# df = pd.DataFrame(data=data, columns=['value'], index=t_idx)
# df.plot()
# testing.N, testing.K = rows, col 
# df = testing.makeTimeDataFrame(freq='MS')
families = ['haar',  'db2', 'db4', 'db8', 'sym4', 'sym8']
g_la8 = [-0.0757657147893407,-0.0296355276459541,
    0.4976186676324578,0.8037387518052163,0.2978577956055422,
    -0.0992195435769354,-0.0126039672622612,0.0322231006040713
    ]
def energy(coeffs): 
    return np.sqrt(np.sum(np.array(coeffs) ** 2) / len(coeffs) )

def entropy(signal):
    counts = Counter(signal).most_common()
    probs = [float(count[1]) / len(signal) for count in counts]
    w_entropy =scipy.stats.entropy(probs)
    return w_entropy

def renyi_entropy(X, alpha):
    assert alpha >= 0, f"Error: renyi_entropy only accepts values of alpha >= 0, but alpha = {alpha}."  # DEBUG
    if np.isinf(alpha):
        #  Min entropy!
        return - np.log2(np.max(X))
    elif np.isclose(alpha, 0):
        # Max entropy!
        return np.log2(len(X))
    elif np.isclose(alpha, 1):
        #  Shannon entropy!
        return entropy(X)
    else:
        counts = Counter(X).most_common()
        probs = np.array([float(count[1]) / len(X) for count in counts])
        return (1.0 / (1.0 - alpha)) * np.log2(np.sum(probs ** alpha))

def npess(data):
    # https://pdfs.semanticscholar.org/0c8b/e141c9092ed389b9931ac09ec2e852d437c6.pdf appendix A3
    'Normalized partial energy sequence'
    U  = np.empty(len(data))

    for i, elem in np.ndenumerate(data):
        U[i] = elem * elem

    U = np.array(sorted(U, reverse=True))

    return np.divide(np.cumsum(U),sum(U))

def plot_npess(npess, wave_coeff, wave_name,  col='Test', savefig=False):
  
    fig, ax = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    fig.sup_title('NPESS '+col)
    ax[0].plot(np.arange(len(npess)), npess, '--', color='k', label='NPESS of '+col)
    ax[0].ylabel('Partial Energy')
    ax[0].legend(loc=0)
    ax[1].plot(wave_coeff, '-x', label='NPESS Wave Coeff with '+ wave_name)
    ax[1].legend(loc=0)
    ax[1].xlabel('# datapoints')
    ax[1].ylabel('Parital Energy')

    
    if savefig == True:
        plt.savefig('Wavelet/wave_choice_npess'+ col +'_analysis .pdf', dpi=120)
        plt.savefig('Wavelet/wave_choice_npess'+ col +'_analysis .png', dpi=120)
    plt.show()

def  test(coeffs, j_0):
    # TODO FIX EITHER THIS OR MULTI RES INFO
    #! approx = vj_0
    #! details = wj

    det_sum = np.sum(np.asarray([ np.dot(coef, coef) for _, coef in enumerate(coeffs[0][1:j_0]) ]))
    approx, details = coeffs[0][0], coeffs[0][1:]
    E = np.dot(approx, approx) * det_sum
    k_list = range(len(approx))
    j_list = range(j_0)

    for k in k_list:
        A = (approx[k])**2
        for j in j_list:
            A += (details[j][k])**2
        A = np.log2(A)

        B = np.log2(E)

        for j in j_list:
            C = 0
            for k in k_list:
                C += approx[k]**2 + details[j][k]**2
            C = np.log2(C)

            D = np.log2(approx[k]**2 / j_0 + details[j][k]**2)

            F = B + D - C - A

            G = (approx[k]**2 / j_0 + details[j][k]**2) / E

            H = G * F
        
    print('done?')

def multires_info(coeffs, j_0, j):
    # TODO POOP PERCENTAGES
    #  approx = vj_0, details = wj
    # https://arxiv.org/ftp/arxiv/papers/1502/1502.05879.pdf eqn 26, 27
    approx , details = coeffs[0][0], coeffs[0][1:]
    pp(approx)
    print('---------------\n')
    pp(details)
    sys.exit()
    det_sum  = np.sum(np.asarray([np.dot(detail, detail) for detail in details]))
    det_sum_lvl = np.sum(np.asarray([ np.dot(coef, coef) for _, coef in enumerate(coeffs[0][1:j]) ]))

    E = np.dot(approx, approx) + det_sum_lvl

    # wj = [ np.dot(coef, coef) for _, coef in enumerate(coeffs[0][1:j]) ]
    # temp = [np.dot(detail, detail) for detail in details]
    # vJ = np.dot(approx, approx)
    # vJ_ = vJ/j_0
    # wj.append(vJ_)
    # p1 = np.sum(np.asarray(wj))/E 
    # A = np.log2(np.sum(np.asarray(wj)))
    # B = np.log2(E)
    # p2 = A + np.log2()
    part1 = ((np.dot(approx, approx)) + det_sum)/E 
 

    part2 = np.log2((E * part1)/(E * (np.dot(approx, approx) + det_sum_lvl )))
    print('------\n')
    print(part2)
    print('-----\n')
    print(part1*part2)
    return part1*part2

def ridge_regress(X, Y):
    pass 

def choose_wavelet_signal(data, families=families, debug=False):
    assert isinstance(data, pd.Series) , f"Expect pandas Series, {type(data)} given"
    ap = data.values
    tests = ['energy', 'shanon', 'renyi', 'r_shanon', 'r_renyi', 'bit']
    info ={fam:{i:[] for i in tests } for fam in families}
    if debug == True:
        print("Original signal  Entropy", entropy(ap))

    for fam in families:
        rennies = []
        for i in range(wv.dwt_max_level(len(data.values), fam)):
            ap, det =  wv.dwt(ap, fam)
            e_ap = energy(ap)
            ren = renyi_entropy(ap, 3)
            rennies.append(ren)
            entr = entropy(ap)
            ratio = e_ap/ entr if entr else 0.01
            r_ren = e_ap/ ren if ren else 0.01
            bit = rennies[0] - rennies[i-1] if len(rennies) > 2 else 0.0
            if debug == True:
                print('[DBEUG] index', i,'wave ', fam, 'energy', np.log10(e_ap), 'entropy sh', entr, 'renyi', ren,  'ratio shanny', np.log10(ratio), "ratio renny", np.log10(r_ren),'Renyi bit of info', np.exp2(round(bit)), sep='\n\n' ) 
            info[fam]['energy'].append(np.log10(e_ap) if abs(np.log10(e_ap)) != np.inf else 0)
            info[fam]['shanon'].append(entr)
            info[fam]['renyi'].append(ren)
            info[fam]['r_shanon'].append(np.log10(ratio) if abs(np.log10(ratio)) != np.inf else 0)
            info[fam]['r_renyi'].append(np.log10(r_ren) if abs(np.log10(r_ren)) != np.inf else 0)
            info[fam]['bit'].append(np.exp2(round(bit)))
        if debug == True:
            print('\n*-------------------------------------*\n')
    # plot_choice_wavelet_signal(data=info, columns=tests)

def plot_choice_wavelet_signal(data, columns, savefig=False):
    # TODO FIX THIS PLOT
    df = pd.DataFrame.from_dict(data=data, orient='index').stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index) 
    index = pd.MultiIndex.from_tuples(df.index, names=['wave', 'analysis'])
    df.index = index 
    df  = df.rename(columns={i:'level '+str(i) for i, _ in enumerate(df.columns.tolist())})
    df = df.T
  
    for col in columns:
        df.xs(col, level=('analysis'), axis=1).plot(subplots=True, layout=(4, 4), figsize=(16, 8), title='Analysis  of '+ col+' per wavelet on decomposition level')
        if savefig == True:
            plt.savefig('Wavelet/wave_choice'+ col +'_analysis .pdf', dpi=120)
            # plt.savefig('Wavelet/wave_choice'+ col +'_analysis .png', dpi=120)
    plt.show()  

def wavelet_var(data, wavelet, mode, levels, method='modwt'):
    assert isinstance(data, pd.Series) , f"Expect pandas Series, {type(data)} given"
    print(f'[INFO] Wavelet variance per scale analysis..')
    ap = data
    temp = wv.dwt_max_level(len(data), wavelet.dec_len)
    lvl_decomp = levels if temp > levels else temp
    result_var_level = np.zeros((lvl_decomp, 3))
    if method == 'dwt':
        for i in range(lvl_decomp):
            ap, det = wv.dwt(ap, wavelet, mode=mode)
            result_var_level[i][0] =  np.dot(det[1:-1], det[1:-1])/(len(data) - 2**(i - 1) + 1)
            result_var_level[i][1] = np.var(data)/(2**i+1)
        print('[INFO] Wavelet variant scale analysis done using DWT recursion ')
        return result_var_level
    if method == 'wavedec':
        coeffs = wv.wavedec(ap, wavelet, mode=mode, level=lvl_decomp)
        details = coeffs[1:]
        for i in range(levels):
            result_var_level[i][0] = np.dot(details[i], details[i])/(len(data) - 2**(i - 1) + 1)
            result_var_level[i][1] = np.var(data)/(2**i+1)
        print('[INFO] Wavelet variant scale analysis done using WAVEDEC  ')
        return result_var_level

    if method == 'modwt':
        data = get_pad_data(data=data)
        new_wavelet = convert_to_modwt_filter(wavelet)
        coeffs = wv.swt(data, new_wavelet, level=lvl_decomp, trim_approx=True, norm=True)
        details = coeffs[1:]
        for i in range(lvl_decomp):
            Mj = len(data) - 2**(i - 1) + 1
            result_var_level[i][0] = np.dot(details[i], details[i])/(Mj)
            result_var_level[i][1] = np.var(data)/(2*(2**(i-1)))
            result_var_level[i][2] = max(Mj/2**(i) , 1)
            # print(f'eta3 {result_var_level[i][2]}')
            # print(f' Going to  results {result_var_level[i][1]} scale to scale {result_var_level[i][0]}')
        print('[INFO] Wavelet variant scale analysis done using MODWT')
        return result_var_level

def conf_interval_wave_var(data, method='scale', alpha=0.05):
    conf_intv = np.zeros((len(data),2))
    if method == 'var':
        conf_intv[:,0] = (data[:,2]*data[:,1])/chdtri(data[:,2], 1 - alpha)
        conf_intv[:,1] = (data[:,2]*data[:,1])/chdtri(data[:,2], alpha)
        # pp(conf_intv)
        return conf_intv
    if method =='scale':
        conf_intv[:,0] = (data[:,0]*data[:,2])/chdtri(data[:,2], 1 - alpha)
        conf_intv[:,1] = (data[:,0]*data[:,2])/chdtri(data[:,2], alpha)
        return conf_intv

def plot_wavelet_var(var_result, title, path,  mode='scale', alpha=0.05, savefig=False):
    plt.figure(figsize=(16,8), dpi=90)
    ci_low, ci_high  = None, None
    scales = np.arange(1, len(var_result)+1)
    if mode == 'var':
        # Gaussian normality CI
        # ci_low, ci_high  =    stats.DescrStatsW(var_result[:,1]).zconfint_mean()
        # st.t.interval(0.95, len(var_result)-1, loc=np.mean(var_result), scale=st.sem(var_result))
        # Chi2 CI
        conf  =  conf_interval_wave_var(var_result, method=mode,alpha=alpha )
        ci_low, ci_high = conf[:,0], conf[:,1]
        plt.plot(scales, var_result[:,1], 'o-', color='k', alpha=0.6, label=r'Var result of $\tau$')
        plt.fill_between(scales, (abs(var_result[:,1] - ci_low)), (var_result[:,1] + ci_high), color='r', alpha=0.3, label=r'95 % confidence interval')
    if mode == 'scale':
        conf  =  conf_interval_wave_var(var_result, method=mode,alpha=alpha )
        ci_low, ci_high = conf[:,0], conf[:,1]
        plt.plot(scales, var_result[:,0], 'o-', color='k', alpha=0.6, label=r'Var result of $\tau$')
        plt.fill_between(scales, (abs(var_result[:,0] - ci_low)), (var_result[:,0] + ci_high), color='r', alpha=0.3, label=r'95 % confidence interval')
    plt.xlabel(r'Scales $\tau$')
    
 
    plt.ylabel(r'Wavelet variance $\nu^2$')
    plt.title(f'Wavelet variance per level  of {str(title)} ')
    plt.yscale('log',basey=10) 
    plt.xscale('log',basex=2)
    plt.tight_layout()
    plt.legend(loc=0)
    if savefig == True:
        Path('Wavelet/variance/'+path).mkdir(parents=True, exist_ok=True)
        plt.savefig('Wavelet/variance/'+path+'/wave_var_scale'+ str(title) +'_analysis .pdf', dpi=120)
        # plt.savefig('Wavelet/variance/'+path+'/wave_var_scale'+ str(title) +'_analysis .png', dpi=120)
    else:
        plt.show()

def generate_rgcpd(target=3, prec_path='sst_1979-2018_2.5deg_Pacific.nc'):
    path_data = os.path.join(main_dir, 'data')
    current_analysis_path = os.path.join(main_dir, 'Jier_analysis')
    target= target
    target_path = os.path.join(path_data, 'tf5_nc5_dendo_80d77.nc')
    precursor_path = os.path.join(path_data,prec_path)
    list_of_name_path = [(target, target_path), 
                        (prec_path[:3], precursor_path )]
    list_for_MI = [BivariateMI(name=prec_path[:3], func=BivariateMI.corr_map, 
                            kwrgs_func={'alpha':.0001, 'FDR_control':True}, 
                            distance_eps=700, min_area_in_degrees2=5)]
    rg = RGCPD(list_of_name_path=list_of_name_path,
            list_for_MI=list_for_MI,
            path_outmain=os.path.join(main_dir,'data'))
    return rg 

def create_rgcpd_obj(rg, precur_aggr=1):
    rg.pp_precursors(detrend=True, anomaly=True, selbox=None)
    rg.pp_TV()
    rg.traintest(method='no_train_test_split')
    rg.calc_corr_maps()
    rg.cluster_list_MI()
    rg.get_ts_prec(precur_aggr=precur_aggr)
    return rg 

def setup_wavelets_rgdata(rg, wave='db4', modes=wv.Modes.periodic):
    cols = rg.df_data.columns.tolist()[:-2]
    rg_data  = rg.df_data[cols]
    rg_data = rg_data.rename(columns={cols[i]:'prec'+str(i) for i in range(1, len(cols)) })
    rg_index = rg_data.index.levels[1]
    # precursor_list = [rg_data['prec'+str(i)].values for i in range(1, len(cols))]
    # target = rg_data[cols[0]]
    wave  = wv.Wavelet(wave)
    mode=modes 
    return (rg_data, rg_index),  (wave, mode)

def plot_discr_wave_decomp(data, wave, name, savefig=False):
    assert isinstance(data, pd.Series) , f"Expect pandas Series, {type(data)} given"
    lvl_decomp = wv.dwt_max_level(len(data), wave.dec_len)
    fig, ax = plt.subplots(lvl_decomp, 2, figsize=(19, 8))
    fig.suptitle('Using Discrete Wavelet transform', fontsize=18)
    ap = data.values
    for i in range(lvl_decomp):
        ap, det =  wv.dwt(ap, wave)
        ax[i, 0].plot(ap, 'r')
        ax[i, 1].plot(det, 'g')
        ax[i, 0].set_ylabel('Level {}'.format(i + 1), fontsize=14, rotation=90)
        if i == 0:
                ax[i, 0].set_title('Approximation coeffs', fontsize=14)
                ax[i, 1].set_title('Details coeffs', fontsize=14)
    plt.tight_layout()
    if savefig == True:
            plt.savefig('Wavelet/wave_decompose'+ name +'_analysis .pdf', dpi=120)
            # plt.savefig('Wavelet/wave_decompose'+ name +'_analysis .png', dpi=120)
    else:
        plt.show()

def create_low_freq_components(data, level=6, wave='db4', mode=wv.Modes.periodic, debug=False):
    assert isinstance(wave, wv.Wavelet)
    assert isinstance(data, pd.Series) , f"Expect pandas Series, {type(data)} given"
    s = data
    cA = []
    cD = []
    lvl_decomp = wv.dwt_max_level(len(data), wave.dec_len)
    lvl_decomp = level if lvl_decomp > level else lvl_decomp
    for i in range(lvl_decomp): # Using recursion to overwrite signal to go level deepeer
        s, det =  wv.dwt(s, wave , mode=mode)
        cA.append(s)
        cD.append(det)
    
    if debug == True:
        print('[DEBUG] Inspecting approximations length of low freq')
        for i, c in enumerate(cD):
            print('Vj Level: ', i,'Size: ', len(c))
        for i, x in enumerate(cA):
            print('Wj Level: ', i, 'Size: ', len(c))
    return cA, cD

def create_signal_recontstruction(data, wave, level, mode=wv.Modes.periodic):
    w = wave
    assert isinstance(data, pd.Series) , f"Expect pandas Series, {type(data)} given"
    assert isinstance(w, wv.Wavelet)
    a = data
    ca = []
    cd = []
    level_ = wv.dwt_max_level(len(data), w.dec_len)
    lvl_decomp = level if level_ > level else level_
    for i in range(lvl_decomp):
        (a, d) = wv.dwt(a, w, mode)
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(wv.waverec(coeff_list, w))

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        rec_d.append(wv.waverec(coeff_list, w)) 

    return rec_a, rec_d

def convert_to_filter_bank(g, mode='modwt'):
    if mode !='modwt':
        return wv.orthogonal_filter_bank(g)
    else:
        g_normalized = g/np.sqrt(2)
        return wv.orthogonal_filter_bank(g_normalized)

def create_least_asymmetric_filter(g=g_la8, mode='modwt'):
    if mode != 'modwt':
        filter_bank = convert_to_filter_bank(g, mode=mode)
        la = wv.Wavelet(name='la8', filter_bank=filter_bank)
        la.orthogonal = True
        la.biorthogonal = True 
        return la 
    else:
        filter_bank = convert_to_filter_bank(g=g, mode=mode)
        la = wv.Wavelet(name='la8_modwt', filter_bank=filter_bank)
        la.orthogonal = True
        la.biorthogonal = True 
        return la

def convert_to_modwt_filter(wave):
    new_wave  = wv.Wavelet(name=wave.name+'norm', filter_bank=[np.asarray(f)/np.sqrt(2) for f in wave.filter_bank])
    new_wave.orthogonal = True
    new_wave.biorthogonal = True 
    return new_wave

def create_modwt_decomposition(data, wave, level, la=False):

    assert isinstance(data, pd.Series) , f"Expect pandas Series, {type(data)} given"
    assert isinstance(wave, wv.Wavelet)
    if not la :
        w = convert_to_modwt_filter(wave)
    else:
        w = create_least_asymmetric_filter()
    a = get_pad_data(data=data)
    coeffs =  wv.swt(a, w, level=level, trim_approx=True, norm=True) #[(cAn, (cDn, ...,cDn-1, cD1)]
    return coeffs[0], coeffs[1:]

def create_mci_coeff(cA, cA_t, rg_index, rg, debug=False):

    obj_rgcpd = []
    for i in range(0,len(cA)):    
        idx_lvl_t = pd.DatetimeIndex(pd.date_range(rg_index[0] ,end=rg_index[-1], periods=len(cA_t[i]) ).strftime('%Y-%m-%d') )
        idx_prec = pd.DatetimeIndex(pd.date_range(rg_index[0], rg_index[-1], periods=len(cA[i]) ).strftime('%Y-%m-%d') )
        dates = core_pp.get_subdates(dates=idx_lvl_t, start_end_date=('06-01', '08-31'), start_end_year=None, lpyr=False)
        full_time  = idx_lvl_t
        RV_time  = dates
        RV_mask = pd.Series(np.array([True if d in RV_time else False for d in full_time]), index=pd.MultiIndex.from_product(([0], idx_lvl_t)), name='RV_mask')
        trainIsTrue = pd.Series(np.array([True for _ in range(len(cA_t[i]))]), index=pd.MultiIndex.from_product(([0], idx_lvl_t)), name='TrainIsTrue')
        ts_ca1 = pd.Series(cA[i], index=pd.MultiIndex.from_product(([0], idx_prec)),name='p_1_lvl_'+ str(i)+'_dec')
        ts_tca1 = pd.Series(cA_t[i], index=pd.MultiIndex.from_product(([0],idx_lvl_t)), name='3ts')
        df = pd.concat([ts_tca1, ts_ca1, trainIsTrue, RV_mask], axis=1)
        rg.df_data = df
        rg.PCMCI_df_data()
        rg.PCMCI_get_links()
        rg.df_MCIc
        obj_rgcpd.append(deepcopy(rg.df_MCIc))
        if debug == True:
            rg.PCMCI_plot_graph()
            plt.show()
    return obj_rgcpd

def extract_mci_lags(to_clean_mci_df, lag=0):

    lag_target = [lags.values[:,lag][1] for _, lags in enumerate(to_clean_mci_df)]
    lag_precurs = [lags.values[:,lag][1] for _, lags in enumerate(to_clean_mci_df)]
    return lag_target, lag_precurs

def plot_mci_pred_relation(cA, prec_lag, path, title, savefig=False):
    # TODO RECOGNISABLE WAY TO SAVE DISTINCTS PLOTS
    x_as = np.arange(1, len(cA)+1)
    x_as = np.exp2(x_as)
    plt.figure(figsize=(16,8), dpi=120)
    plt.plot(x_as, prec_lag, label='precrursor ')
    plt.xticks(x_as)
    plt.title(title)
    plt.xlabel('Scales in daily means')
    plt.ylabel('MCI')
    plt.legend(loc=0)
    if savefig ==True:
        Path('Wavelet/Mci/'+path).mkdir(parents=True, exist_ok=True)
        plt.savefig('Wavelet/Mci/'+path+'/MCI on scale wavelet on lag 0 of '+str(title)+' iteration.pdf', dpi=120)
        # plt.savefig('Wavelet/Mci/'+path+'/MCI on scale wavelet on lag 0 of '+str(title)+' iteration.png', dpi=120)
    else:
        plt.show()

