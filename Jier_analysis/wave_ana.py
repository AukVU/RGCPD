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
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = (20.0, 10.0)
import itertools as it
import pywt as wv
from scipy.fftpack import fft
from copy import deepcopy
from pprint import pprint as pp 
from pandas.plotting import register_matplotlib_converters
from RGCPD import RGCPD
from RGCPD import BivariateMI
import core_pp
import plot_signal_decomp
import plot_coeffs
from visualize_cwt import *
register_matplotlib_converters()
# np.random.seed(12345)
plt.style.use('seaborn')
current_analysis_path = os.path.join(main_dir, 'Jier_analysis')

# rows, col = 1000, 1
# data = np.random.rand(rows, col)
# t_idx = pd.date_range('1980-01-01', periods=rows, freq='MS')
# df = pd.DataFrame(data=data, columns=['value'], index=t_idx)
# df.plot()
# testing.N, testing.K = rows, col 
# df = testing.makeTimeDataFrame(freq='MS')
families = ['haar', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7']


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
        # XXX Min entropy!
        return - np.log2(np.max(X))
    elif np.isclose(alpha, 0):
        # XXX Max entropy!
        return np.log2(len(X))
    elif np.isclose(alpha, 1):
        # XXX Shannon entropy!
        return entropy(X)
    else:
        counts = Counter(X).most_common()
        probs = np.array([float(count[1]) / len(X) for count in counts])
        return (1.0 / (1.0 - alpha)) * np.log2(np.sum(probs ** alpha))

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
            ratio = e_ap/ entr if entr else 0.0
            r_ren = e_ap/ ren if ren else 0.0
            bit = rennies[0] - rennies[i-1] if len(rennies) > 2 else 0.0
            if debug == True:
                print('index', i,'wave ', fam, 'energy', e_ap, 'entropy sh', entr, 'renyi', ren,  'ratio shanny', ratio, "ratio renny", r_ren,'Renyi bit of info', round(bit) ) 
            info[fam]['energy'].append(e_ap)
            info[fam]['shanon'].append(entr)
            info[fam]['renyi'].append(ren)
            info[fam]['r_shanon'].append(ratio)
            info[fam]['r_renyi'].append(r_ren)
            info[fam]['bit'].append(round(bit))
    plot_choice_wavelet_signal(data=info, columns=tests)

def plot_choice_wavelet_signal(data, columns):
    # TODO FIX THIS PLOT
    df = pd.DataFrame.from_dict(data=data, orient='index').stack().to_frame()
    df = pd.DataFrame(df[0].values.tolist(), index=df.index) 

    # fig, ax = plt.subplots(len(families),len(columns), figsize=(19, 8)) 
    # fig.suptitle('Choice wavelets for given data', fontsize=18)
    df.plot(subplots=True,  layout=(len(families), len(df.columns)), figsize=(19, 8))
    # plt.tight_layout()
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
    precursor_list = [rg_data['prec'+str(i)].values for i in range(1, len(cols))]
    target = rg_data[cols[0]].values
    wave  = wv.Wavelet(wave)
    mode=modes 
    return (rg_data, rg_index), (precursor_list, target), (wave, mode)

def plot_discr_wave_decomp(data, wave):
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
    plt.show()

def create_low_freq_components(data, wave='db4', mode=wv.Modes.periodic, debug=False):
    assert isinstance(data, pd.Series) , f"Expect pandas Series, {type(data)} given"
    s = data
    cA = []
    lvl_decomp = wv.dwt_max_level(len(data), wave.dec_len)
    lvl_decomp = 6 if lvl_decomp > 6 else lvl_decomp
    for i in range(lvl_decomp): # Using recursion to overwrite signal to go level deepeer
        s, _ =  wv.dwt(s, wave , mode=mode)
        cA.append(s)
    
    if debug == True:
        print('Inspecting approximations length of low freq')
        for i, c in enumerate(cA):
            print('Level: ', i,'Size: ', len(c))
    return cA

def create_mci_coeff(cA, cA_t, rg_index, debug=False):

    obj_rgcpd = []
    for i in range(0,len(cA)):    
        idx_lvl_t = pd.DatetimeIndex(pd.date_range(rg_index[0] ,end=rg_index[-1], periods=len(cA_t[i]) ).strftime('%Y-%m-%d') )
        idx_prec = pd.DatetimeIndex(pd.date_range(rg_index[0], rg_index[-1], periods=len(cA[i]) ).strftime('%Y-%m-%d') )
        dates = core_pp.get_subdates(dates=idx_lvl_t, start_end_date=('06-15', '08-20'), start_end_year=None, lpyr=False)
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

def plot_mci_pred_relation(cA, prec_lag, savefig=False):
    x_as = np.arange(1, len(cA)+1)
    x_as = np.exp2(x_as)
    plt.figure(figsize=(19,8), dpi=120)
    plt.plot(x_as, prec_lag, label='precrursor ')
    plt.title('Relation MCI on scale wavelet on lag 0')
    plt.xlabel('Scales in daily means')
    plt.ylabel('MCI')
    plt.legend(loc=0)
    if savefig ==True:
        plt.savefig('Wavelet/Mci/Mci_prec_lag0.pdf', dpi=120)
        plt.savefig('Wavelet/Mci/Mci_prec_lag0.png', dpi=120)
    plt.show()