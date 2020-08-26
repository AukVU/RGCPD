import os, sys, inspect, warnings
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
sub_dir = os.path.join(main_dir, 'RGCPD/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(sub_dir)
import numpy as np 
import statsmodels as st 
import statsmodels.api as sm 
import pandas as pd 
import matplotlib.pyplot as plt 
import itertools as it
import  statsmodels.stats.api as stats
from statsmodels.tsa.stattools import adfuller, kpss 
from statsmodels.tsa.arima_process import  arma_generate_sample, ArmaProcess
from statsmodels.regression.linear_model import yule_walker
from statsmodels.tsa.tsatools import detrend
from pprint import pprint as pp 
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
# np.random.seed(12345)
plt.style.use('seaborn')
from tqdm import tqdm
from sklearn.preprocessing import PowerTransformer 
from sklearn.linear_model import LinearRegression 
from scipy.stats import normaltest
from scipy import signal
import multiprocessing as mp

from statsmodels.tools.sm_exceptions import ConvergenceWarning, InterpolationWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', InterpolationWarning)

# TODO Look if simulation of more than one realization is needed and if so why, comment it. 
# Above is only needed when we want to test ground truth data scenarios against our target data. 

# TODO EVALUATE HOW TO HANDLE EVALUATION OF DATA IN AR OR ARMA SENSE, WHEN CALLED WITH ONE TS CSV WITH ONE COLUMN

synthetic = dict()
synthetic['ARrange'] = np.array([.75, -.25])
synthetic['MArange'] = np.array([.65, .35])
synthetic['arparams'] = np.r_[1, -synthetic['ARrange']]
synthetic['maparams'] = np.r_[1, -synthetic['MArange']]
synthetic['nobs'] = 1000
synthetic['startyear'] = '1980m1'

def generate_synthetic_data():
    return arma_generate_sample(synthetic['arparams'],synthetic['maparams'], synthetic['nobs'])

def generate_date_range():
    return sm.tsa.datetools.dates_from_range(synthetic['startyear'], length=synthetic['nobs'])

def generate_combination_lags(end_range=2):
        return list(it.product(range(1,end_range), repeat=2))

def extract_lags_aic_bic_info_synth(y:pd.Series, combos:list):
    print('[INFO] Starting manually fitting ARMA process..')
    summaries = []
    aic, bic = 10000000000, 10000000000
    ar_ma_params, aci_info, arma_res, arma_mod = [] , [], None, None

    for i, j in combos:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            try:
                arma_mod = sm.tsa.ARMA(y, order=(i, j))
                arma_res = arma_mod.fit(trend='c', disp=-1)
            except:
                continue
        if arma_res.aic < aic and arma_res.bic < bic: 
            aic = arma_res.aic
            bic = arma_res.bic
            ar_ma_params.append((arma_res.arparams, arma_res.maparams,arma_res.sigma2,(i,j)))
            aci_info.append((aic, bic, (i, j),(arma_res.arroots, arma_res.maroots)))
            summaries.append(arma_res.summary())
    print('[INFO] Done manually fitting')
    return ar_ma_params, aci_info, summaries

def retrieve_aci_info(aci_info:list):
    print("AIC, BIC, (i, j), ARroots, MAroots", sep='\t')
    pp(aci_info)

def retrieve_ar_ma_info(ar_ma_params:list):
    print("ARparams, MAparams, Sigmas")
    pp(ar_ma_params)

def retrieve_summaries(summaries:list):
    pp(summaries)

def fit_manually_data(y, arparams:np.array, maparams:np.array, nobs:int, startyear:'', end_range:int, synthetics:bool=True):
    combos = generate_combination_lags(end_range=end_range)
    ar_ma_params, aci_info, summaries = None, None, None
    if synthetics:
        synthetic['ARrange'] = arparams
        synthetic['MArange'] = maparams
        synthetic['nobs'] = nobs
        synthetic['startyear'] = startyear
        y_ = generate_synthetic_data() 
        dates  = generate_date_range()
        y_ = pd.Series(y_, index=dates)
        ar_ma_params, aci_info, summaries = extract_lags_aic_bic_info_synth(y=y_, combos=combos)
    else:
        assert isinstance(y, pd.Series), f"Expected pandas Series, {type(y)} given"
        serie = y
        ar_ma_params, aci_info, summaries = extract_lags_aic_bic_info_synth(y=serie, combos=combos)

    return ar_ma_params, aci_info, summaries
    
def fit_automatically_data(y:pd.Series, pmax:int, qmax:int, ic:list=['aic','bic']):
    try:
        print('[INFO] Starting automatic fitting ARMA...')
        aut_model, aut_order_aic , aut_order_bic= None, [], []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            order_model =  sm.tsa.stattools.arma_order_select_ic(y, ic=ic, trend='c', max_ar=pmax, max_ma=qmax)
            if 'aic' in ic:
                aut_model = sm.tsa.ARMA(y, order=order_model.aic_min_order).fit(trend='c', disp=0)
                print('[DEBUG] AUTO AIC minimum order ',order_model.aic_min_order )
                aut_order_aic.append((aut_model.aic, aut_model.bic, order_model.aic_min_order))
          
            if 'bic' in ic:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    aut_model = sm.tsa.ARMA(y, order=order_model.bic_min_order).fit(trend='c', disp=0)
                print('[DEBUG] AUTO BIC minimum order ',order_model.bic_min_order )
                aut_order_bic.append((aut_model.aic, aut_model.bic, order_model.bic_min_order))

    except ValueError as err:
        print("[ERROR] Occured errors: ", err)
    print("[INFO] Done Automatic fitting\n")
    if (order_model.aic_min_order >= order_model.bic_min_order) == True:
        return aut_order_bic
    else:
        return aut_order_aic

def determine_order(man_model:list, aut_model_order:list, check_info_score:list=['aic']):

    print("[INFO] Starting determining order....")
    temp = [man_model[i][:3] for i in range(len(man_model))]
    man_aic = [temp[i][0] for i in range(len(temp))] 
    man_bic = [temp[i][1] for i in range(len(temp))]
    aut_aic, aut_bic, aut_order = aut_model_order[0]

    idx_aic, idx_bic = None, None
    aut_aic_check, aut_bic_check = False, False
    aic_check, bic_check = False, False

    if len(check_info_score) > 0:
        if 'aic' in check_info_score:
            if min(man_aic) <= aut_aic:
                idx_aic = next( (i for i , v in enumerate(temp) if v[0] == min(man_aic)),-1 )
                print("[DEBUG] Small AIC mannually chosen ", temp[idx_aic])
                aic_check = True
            elif aut_aic < min(man_aic):
                print("[DEBUG] Small AIC automatically chosen ", aut_aic, aut_order)
                aut_aic_check = True
        if 'bic' in check_info_score:
            if min(man_bic) <= aut_bic:
                idx_bic = next( (i for i , v in enumerate(temp) if v[0] == min(man_bic)),-1 )
                print("[DEBUG] Small BIC mannually chosen ", temp[idx_bic])
                bic_check = True
            elif aut_bic < min(man_bic):
                print("[DEBUG] Small BIC automatically chosen ", aut_bic, aut_order)
                aut_bic_check = True

    print("[INFO] Done determining order.\n")

    if aut_bic_check == True or aut_aic_check == True:
        return aut_order
    
    if aic_check == True and bic_check == True:
        if (temp[idx_aic][-1] >= temp[idx_bic][-1]) == True:
            return temp[idx_bic]
    elif aic_check == True:
        return temp[idx_aic]

def check_stationarity_invertibility(aci_info:list, ar_ma_params:list, check_info_score:list=[],nobs:int=synthetic['nobs'], get_params:bool=False, aic_check:bool=False):
    arma_process = []
    aic_decision, bic_decision = None, None
    print('[INFO] Checking stationarity...')
    for i in range(len(aci_info)):
        arma_process.append((ar_ma_params[i], ArmaProcess(ar=ar_ma_params[i][:2][0], ma=ar_ma_params[i][:2][1], nobs=nobs)))
        print("-------------------------------------------------------------------")
        print("[DBEUG] Statioarity ", arma_process[i][-1].isstationary, "Invertibility ", arma_process[i][-1].isinvertible, ar_ma_params[i][-1])
        print('[DBEUG] AR: ', ar_ma_params[i][0], 'MA: ', ar_ma_params[i][1], 'sigma2: ', ar_ma_params[i][2])
        print("-------------------------------------------------------------------")
        print("\n")

    if len(check_info_score) > 0:
        aic_check, bic_check = aic_check,  aic_check 
        idx_pr = None
        temp = [aci_info[i][:3] for i in range(len(aci_info))] # construct temp list of aic, bic info
        if 'aic' in check_info_score:
            man_aic  =[temp[i][0] for i in range(len(temp))]
            idx_aic = next( (i for i , v in enumerate(temp) if v[0] == min(man_aic)),-1 )
            idx_pr = next((i for i , v in enumerate(arma_process) if v[0][-1] == temp[idx_aic][-1]), -1)
            print('[DEBUG] AIC stationarity and Invertibility ', arma_process[idx_pr][:-1][0])
            if get_params == True and aic_check == True:
                aic_decision= arma_process[idx_pr][:-1][0]

        if 'bic' in check_info_score:
            man_bic = [temp[i][1] for i in range(len(temp))]
            idx_bic = next( (i for i , v in enumerate(temp) if v[0] == min(man_bic)),-1 )
            idx_pr = next((i for i , v in enumerate(arma_process) if v[0][-1] == temp[idx_bic][-1]), -1)
            print('[DBEUG] BIC stationarity and Invertibility ', arma_process[idx_pr][:-1][0])

            if get_params == True and bic_check == True:
                bic_decision = arma_process[idx_pr][:-1][0] 
        if (aic_decision[-1] == bic_decision[-1]) == True:
            return bic_decision
        else:
            return aic_decision

def display_info_ts(y:pd.Series, figsize=(16, 8), title="TS", lags=20, save_fig:bool=False):
    assert isinstance(y, pd.Series), f"Expect pandas Series, {type(y)} given"
    serie = y 
    fig = plt.figure(figsize=figsize)
    serie.plot(ax = fig.add_subplot(3, 1, 1), title="$Time \ Series \ "+ title +"$", legend=False)
    sm.graphics.tsa.plot_acf(serie, lags=lags, zero=False, ax = fig.add_subplot(3, 2, 3))
    plt.xticks(np.arange(1, lags + 1, 1.0))

    sm.graphics.tsa.plot_pacf(serie, lags=lags, zero=False, ax = fig.add_subplot(3, 2, 4))
    plt.xticks(np.arange(1, lags + 1, 1.0))

    sm.qqplot(serie, line='s', ax = fig.add_subplot(3, 2, 5))
    fig.add_subplot(326).hist(serie, bins= 40, density=True)
    plt.tight_layout()
    if save_fig == True:
        plt.savefig('Fitted/AR/Plots/times_serie_'+title+'_stats.pdf', dpi=120)
        plt.savefig('Fitted/AR/Plots/time_serie_'+title+'stats.png', dpi=120)
    plt.show()

def display_pierce_LJbox(y:pd.Series, dates:pd.DatetimeIndex, figsize=(16, 8), title="", lags=20, save_fig:bool=False, debug:bool=False):
    
    assert isinstance(y, pd.Series), f"Expect pandas Series, {type(y)} given"
    y.index = dates + pd.Timedelta(1, unit=dates.freqstr)
    acor_ljungbox = list(sm.stats.diagnostic.acorr_ljungbox(y, lags=lags, boxpierce=True))

    if debug == True:
        print('[DEBUG]')
        column_ind = ["Ljung-Box: X-squared", 'Ljung-Box: p-value', 'Box-Pierce: X-squared', 'Box-Pierce: p-value']
        df =pd.DataFrame(acor_ljungbox, index=column_ind, columns= range(1, len(acor_ljungbox[0]) + 1))
        df.plot(subplots=True, layout=(len(column_ind), -1), figsize=figsize, label="Lag ", title='Quick inspection of Ljung-Box' )
        plt.show()

    plt.plot(range(1, len(acor_ljungbox[0]) + 1), acor_ljungbox[1], 'bo', label= 'Ljung-Box values')
    plt.plot(range(1, len(acor_ljungbox[0]) + 1), acor_ljungbox[3], 'go', label= 'Box-Pierce values')
    plt.xticks(np.arange(1, len(acor_ljungbox[0]) + 1, 1.0))
    plt.axhline(y = 0.05, color = 'red', label= "$5 \%  critical value$")
    plt.title("$Time\ Serie\ " + title +"$")
    plt.legend(loc=0)

    if save_fig == True:
        plt.savefig('Fitted/AR/Plots/time_serie__pierce_LJbox.pdf', dpi=120)
        plt.savefig('Fitted/AR/Plots/time_serie__pierce_LJbox.png', dpi=120)
    plt.show()
   
def display_poly_data_arma(simul_data:np.array, ar:list, ma:list, signal:pd.Series, order:tuple, save_fig:bool=False):
    # TODO FIX IF ORDER [0] DIFFERS FROM ORDER[1]
    _dates = pd.DatetimeIndex(signal.index.levels[1], freq='infer')
    l = ''
    if order[0] >= order[1]:
        for i in range(order[0]):
            try:
                l +=str(round(ar[i], 2))+'Y_{t-'+f'{i}'+'}' + str(round(ma[i], 2)) +'\\epsilon_{t-' +f'{i}'+'}'
            except:
                l +=str(round(ar[i], 2))+'Y_{t-'+f'{i}'+'}' + str(round(ma[i-1], 2)) +'\\epsilon_{t-'+f'{i}'+'}'
    else:
        for i in range(order[1]):
            try:
                l +=str(round(ar[i-1], 2))+'Y_{t-' +f'{i}'+'}' + str(round(ma[i], 2)) +'\\epsilon_{t-' +f'{i}'+'}'
            except:
                pass 
    l +='+\\epsilon_{t}'
    plt.figure(figsize=(16, 8))
    plt.plot(_dates, simul_data, label='ARMA='+'$'+l+'$')
    plt.plot(_dates, signal, label='Precursor data')
    plt.title('Chosen fitted model params with original data')
    plt.legend()
    if save_fig == True:
        plt.savefig('Fitted/ARMA/Synthetic_data_.pdf', dpi=120)
        plt.savefig('Fitted/ARMA/Synthetic_data_.png', dpi=120)
    plt.show()

def display_poly_data_ar(simul_data:np.array, ar:list, signal:pd.Series, title:str=' ', save_fig:bool=False, dep:bool=False):

    _dates = pd.DatetimeIndex(signal.index.levels[1], freq='infer')
    __dates = None 
    if len(_dates) > len(simul_data):
        print(f'[INFO] Due to differencing  is polynomial ts shorter than its original ts')
        __dates = pd.DatetimeIndex(signal.index.levels[1][1:], freq='infer')
    l = ''
    if dep == True:
        for i, idx in zip(range(len(ar)), ['-1', '-2', '-1']):
            l +=str(round(ar[i], 2))+'Y_{t'+f'{idx}'+'}+' 
    else:
        for i in range(len(ar[:2])):
            l +=str(round(ar[i], 2))+'Y_{t-'+f'{i+1}'+'}+' 
        

    l +='\\epsilon_{t}'
    plt.figure(figsize=(16, 8), dpi=90)

    ci_low, ci_high  = stats.DescrStatsW(simul_data).tconfint_mean()
    if len(_dates) > len(simul_data):
        plt.fill_between(__dates, simul_data-ci_low, simul_data + ci_high, color='r', alpha=0.3, label=r'95 % confidence interval')
        plt.plot(__dates, simul_data, '-b', label='AR(2)= '+'$'+l+'$', alpha=0.5)
    else:
        plt.fill_between(_dates, simul_data-ci_low, simul_data + ci_high, color='r', alpha=0.3, label=r'95 % confidence interval')
        plt.plot(_dates, simul_data, '-b', label='AR(2)= '+'$'+l+'$', alpha=0.5)

    plt.plot(_dates, signal, '-k', label='Precursor data', alpha=0.5)
    plt.title('Ar2 fit on ' + title)
    plt.xlabel('Dates')
    plt.ylabel('Variance in temperature Celsius')
    plt.legend()
    if save_fig == True:
        plt.savefig('Fitted/AR/Plots/'+title+'.pdf', dpi=120)
        plt.savefig('Fitted/AR/Plots/'+title+'.png', dpi=120)

    # plt.show()

def evaluate_synthetic_data(display:bool=False):
    # Synthetic data
    dates = generate_date_range()
    dates = pd.DatetimeIndex(dates, freq='infer')
    y = generate_synthetic_data()
    y = pd.Series(y, index=dates)

    ar_ma_params, aci_info, summaries = fit_manually_data(y=None, arparams=np.array([.75, -.25]), maparams=np.array([.65, .35]), nobs=synthetic['nobs'], startyear=synthetic['startyear'], end_range=4)
    aut_model, aut_order = fit_automatically_data(y=y, pmax=4, qmax=4, ic=['bic'])

    order = determine_order(aci_info, aut_order)
    check_stationarity_invertibility(aci_info, ar_ma_params)
    model = sm.tsa.ARMA(y, order=order[-1]).fit(trend='c', disp=-1)

    if display == True:
        display_info_ts(pd.Series(y), title='Synthetic generated')
        display_pierce_LJbox(model.resid, dates=dates, title='Synthetic generated')

def evaluate_data_ar(data, col,  ic='aic', method='mle', disp=0, title='Fitted  AR on precursor', debug=True, display=False, save_fig= False):
    
    assert isinstance(data, pd.Series), f"Expect pandas Series, {type(data)} given"
    if debug == True:
        print('[DEBUG] Start fitting AR process ')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        temp_dates = pd.DatetimeIndex(data.index.levels[1], freq='infer')
        temp_m = pd.DataFrame(data=data.values, index=temp_dates, columns=[col])
        ar_model = st.tsa.ar_model.AR(temp_m).fit(ic=ic, method=method, maxiter=len(data), disp=disp)

    if debug == True:
        print('[DEBUG] Done fitting AR process')

    if display == True:
        display_info_ts(ar_model.resid, title=title, save_fig=save_fig)
        display_pierce_LJbox(ar_model.resid, dates=temp_dates, title=title, debug= False, save_fig=save_fig)
        display_info_ts(data, title='Original Precursor', save_fig=save_fig)
        display_pierce_LJbox(data, dates=temp_dates, title='Original Precursor', debug=False, save_fig=save_fig)

    return ar_model.params[0], ar_model.params[1:] 

def evaluate_data_yule_walker(data, col, order=2, method='mle', debug=False, display=False, save_fig=False):
    print(f'\n[INFO] Start applying Yule-Walker algorithm for AR process serie {col}')
    rho, sigma = yule_walker(data[col], order=order, method=method)
    if debug == True:
        print(f'[DEBUG] rho : {rho}, sigma: {sigma}', sep='\n')
    if display == True:
        temp_dates = pd.DatetimeIndex(data.index.levels[1], freq='infer')
        display_info_ts(data, title='Original Precursor', save_fig=save_fig)
        display_pierce_LJbox(data, dates=temp_dates, title='Original Precursor', debug=False, save_fig=save_fig)
        print('[INFO] Evaluation done.')
    return rho, sigma

def evaluate_data_arma(signal, display:bool=False, title='Fitted ARMA precursor', aic_check:bool=False, debug:bool=False):

    assert isinstance(signal, pd.Series), f"Expect pandas Series, {type(signal)} given"
    ar_ma_params, aci_info, summaries = fit_manually_data(signal['values'], arparams=np.array([.75, -.25]), maparams=np.array([.65, .35]), nobs=synthetic['nobs'], startyear=signal.index[0][1], end_range=5,synthetics=False)
    ar, ma, sigma, order = check_stationarity_invertibility(aci_info, ar_ma_params,check_info_score=['aic', 'bic'] ,nobs=len(signal), get_params=True,aic_check=aic_check)
    aut_order = fit_automatically_data(y=signal, pmax=4, qmax=4, ic=['aic', 'bic'])
    determine_order(aci_info, aut_model_order=aut_order,check_info_score=['aic', 'bic'])

    if debug == True:
        retrieve_summaries(summaries)
        retrieve_ar_ma_info(ar_ma_params)
        retrieve_aci_info(aci_info)

    if display == True:
        temp_dates = pd.DatetimeIndex(signal.index.levels[1], freq='infer')
        temp_m = pd.DataFrame(data=signal.values, index=temp_dates)
        model = sm.tsa.ARMA(temp_m, order=order[-1]).fit(trend='c', disp=-1)
        display_info_ts(model.resid, title=title)
        display_pierce_LJbox(model.resid, dates=temp_dates, title=title)
        display_info_ts(signal, title='Original Precursor')
        display_pierce_LJbox(signal, dates=temp_dates, title='Original Precursor')

    return ar, ma, sigma, order

def create_polynomial_fit_arma(ar:list, ma:list, sigma:float, data:pd.Series):
    N =  len(data)
    epsilon = np.random.normal(loc=0, scale=sigma, size=N)
    simul_data = np.zeros(N)
    simul_data[0] = epsilon[0]
    simul_data[1] =  ar[0] * simul_data[0] + epsilon[1] + ma[0] * epsilon[0]
    for i in range(2, N):
        if len(ar) > 1 and len(ma) > 1:
            simul_data[i] = ar[0] * simul_data[ i - 1] + ar[1] * simul_data[i  - 2] + epsilon[i] + ma[0] * epsilon[i - 1] + ma[1] * epsilon[i - 2]
        elif len(ar) == 1:
            simul_data[i] = ar[0] * simul_data[ i - 1] + epsilon[i] + ma[0] * epsilon[i - 1] + ma[1] * epsilon[i - 2]
        elif len(ma) == 1:
            simul_data[i] = ar[0] * simul_data[ i - 1] + ar[1] * simul_data[i  - 2] + epsilon[i] + ma[0] * epsilon[i - 1] 
    
    return simul_data

def create_polynomial_fit_ar(ar:list, sigma:float, data:pd.Series, const:int, dependance:bool=False, yule_walker:bool=False, theta:float=0.1, nu:float=0.1,  gamma:float=0.1,  x1:np.array=np.zeros(100)):
    print('\n[INFO] Start running polynomial fit...')
    N =  len(data)
    epsilon = np.random.normal(loc=0, scale= sigma, size=N)
    simul_data = np.zeros(N)
    simul_data[0] = const + epsilon[0]
    ar_0, ar_1 = ar[0], ar[1]

    if dependance == True:
        simul_data[1] =  const + ar_0 * simul_data[0] + epsilon[1] + gamma * x1[0]
    else:
        simul_data[1] =  const + ar_0 * simul_data[0] + epsilon[1]
    for i in range(2, N):
        if dependance == True:
            simul_data[i] = const + ( ar_0 * simul_data[i -1]) + ( ar_1 * simul_data[i - 2])+ epsilon[i] + (gamma * x1[i  -1])
        else:    
            simul_data[i] = const + (ar_0 * simul_data[i -1]) + (ar_1 * simul_data[i - 2]) + epsilon[i]
    print('[INFO] Polynomial fit done.')
    if yule_walker == True:
        print('[INFO] Yule walker standardisation')
        simul = np.array([(i - np.mean(simul_data))/(np.std(simul_data)) for i in simul_data])    
        return simul 
    else:
        return simul_data

def create_polynomial_fit_ar_turbulance(ar:list, sigma:float, data:pd.Series, const:int, yule_walker:bool=False, theta:float=0.1, nu:float=0.1):
    print('\n[INFO] Start running polynomial fit turbulance...')
    N =  len(data)
    epsilon = np.random.normal(loc=0, scale=sigma, size=N)
    simul_data = np.zeros(N)
    simul_data[0] = const + epsilon[0]
    ar_0, ar_1 = ar[0], ar[1]

    simul_data[1] =  const + (nu * ar_0 * simul_data[0] ) + epsilon[1]
    for i in range(2, N):

        simul_data[i] = const + (nu * ar_0 * simul_data[i -1]) + (theta * ar_1 * simul_data[i - 2])+ epsilon[i] 

    print('[INFO] Polynomial turbulance fit done.')
    if yule_walker == True:
        print('[INFO] Yule walker standardisation')
        simul = np.array([(i - np.mean(simul_data))/(np.std(simul_data)) for i in simul_data])    
        return simul 
    else:
        return simul_data

def create_polynomial_fit_ar_depence(x0:np.array, x1:np.array, gamma:float, data:pd.Series):
    print('[INFO] Start running polynomial fit dependance...')
    N =  len(data)
    simul_data = np.zeros(N)
    simul_data = x0 + gamma * x1
    print('[INFO] Polynomial fit dependance done.')
    return simul_data

def stationarity_test(serie, regression='c', debug=False):
    
    adf = None
    try:
        # ADF Test
        results = adfuller(serie, autolag='AIC')
        if debug == True:
            print(f'[DBEUG] ADF Statistic: {results[0]}')
            print(f'[DEBUG] p-value: {results[1]}')
            print('[DEBUG] Critial Values:')
            for key, value in results[4].items():
                print(f'   {key}, {value}')
        print(f'[INFO] Result ADF: The serie is {"NOT " if results[1] > 0.05 else ""}stationary')
        adf = False if results[1] > 0.05 else True
    except ValueError as err:
        print(f'[ERROR] {err}. Using polynomial root testing to evaluate')
        ts = np.polynomial.polynomial.Polynomial(serie)
        print(f'[INFO] Polynomial Time Serie {ts} and its roots {ts.roots()}')
        adf = np.all([True if i.real > 1.0 else False  for i in  ts.roots() ])
        print(f'[INFO] ADF with roots testing stationarity is  {adf}')


    # KPSS Test
    result = None
    if regression == 'c':
        result = kpss(serie, regression='c', lags='auto')
        print(f'[INFO] Result KPSS: The mean of the serie is {"NOT " if result[1] < 0.05 else ""}stationary')
    else:
        result = kpss(serie, regression=regression, lags='auto')
        print(f'[INFO] Result KPSS: The trend of the serie is {"NOT " if result[1] < 0.05 else ""}stationary')
    if debug == True:
        print(f'\n [DEBUG] KPSS Statistic:  {result[0]} ')
        print(f'[DEBUG] p-value: {result[1]} ')
        print('[DEBUG] Critial Values:')
        for key, value in result[3].items():
            print(f'   {key}, {value}')
    kps = False if result[1] < 0.05 else True
    return (adf , kps)

def preprocess_ts(serie, col, threshold=0.05, debug=False):

    print(f'\n[INFO] Preprocessing Time serie \'{col}\' with first standardising and checking for gaussian like p-value ')

    serie = serie.apply(lambda x : (x - x.mean())/ x.std(), axis=0)
    index = serie.index
    p_value = normaltest(serie)[1][0]
    print(f'[INFO] Time serie p_value  {p_value}  ')
    if p_value < threshold :
        print('[INFO] Time serie is gaussian like \n')
        return serie
    else:
        print('[INFO] Time serie is non-gaussian, tranforming the time serie to gaussian like approximation')
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        pt.fit(serie[col].values.reshape(-1, 1))
        trans_serie = pt.transform(serie[col].values.reshape(-1, 1))
        p_value = normaltest(trans_serie)[1][0]
        print(f'[INFO] Gaussian approximation with p-value {p_value}')
        trans_df = pd.DataFrame(data=np.ravel(trans_serie), index=index, columns=[col])
        stat_bool = stationarity_test(serie=trans_df[col], debug=debug)
        if stat_bool[0] == True:
            print(f'[INFO] Transformed time serie is stationair ADF {stat_bool[0]},  KPSS {stat_bool[1]}\n')
            return trans_df

def postprocess_ts(serie, regression, col, debug=False):
    print(f'\n[INFO] Postprocess Polynome of \'{col}\' to examine stationarity')
    stat_bool = stationarity_test(serie, regression=regression)
    if (stat_bool[0]== False) or (stat_bool[1] ==False):
        serie_ = detrend_poly(serie) 
        print(f'[INFO] Detrend polynome of {col} to force trend stationarity')
        stats_bool = stationarity_test(serie_, regression=regression)
        if debug == True:
            print(f'[ERROR] Polynome not stationair, applying first order differencing \n{serie} ADF {stat_bool[0]} KPSS {stat_bool[1]}')
            print(f'[DEBUG] DIFF AR {serie_}' )
            print(f'[DEBUG] ADF: {stats_bool[0]} KPSS: {stats_bool[1]}')
        
        if (stats_bool[0] ==True) and (stats_bool[1] == True):
            if debug == True:
                print(f'[PASS] Polynome stationarity and trend stationarity test passed, postprocess done\n {serie_} ADF {stat_bool[0]} KPSS {stat_bool[1]}\n')
            print('[INFO] Differencing succes, ADF and KPSS Stationarity passed.\n')
            return stats_bool[0], serie_
        print('[WARNING] Still not trend stationarity solved')
    print('[INFO] No differencing, Postprocess done.\n')
    return stat_bool[0], serie


def detrend_poly(dataset, method='scipy'):
    if method == 'stats':
        return detrend(dataset, order=1)
    if method == 'lr':
        def linear_regression(dataset):
            # TODO really need this one?
            X = np.zeros(len(dataset))
            X = np.reshape(X, (len(X), 1))
            y= dataset
            model = LinearRegression()
            model.fit(X, y)
            trend = model.predict(X)
            return np.array([y[i] - trend[i] for i in range(len(dataset))])
    if method == 'scipy':
        return signal.detrend(dataset)

if __name__ == "__main__":
    pass 
    # current_analysis_path = os.path.join(main_dir, 'Jier_analysis/Data/sst/')
    # # ar_data_path = os.path.join(main_dir, 'Jier_analysis/Fitted/AR/AR_Data')

    # # # DEBUG  CREATING POLYNOMIAL
    # target_sst = pd.read_csv(os.path.join(current_analysis_path, 'sst_3ts_1.csv'), engine='python', index_col=[0, 1])
    # target_sst = preprocess_ts(serie=target_sst, col='3ts')


    # first_sst = pd.read_csv(os.path.join(current_analysis_path, 'sst_prec1_1.csv'), engine='python', index_col=[0, 1])
    # first_sst = preprocess_ts(serie=first_sst, col='prec1')

    # second_sst = pd.read_csv(os.path.join(current_analysis_path, 'sst_prec2_1.csv'), engine='python', index_col=[0, 1])
    # second_sst = preprocess_ts(serie=second_sst, col='prec2')

 

    # pt = PowerTransformer(method='yeo-johnson')
    # pt.fit(second_sst['values'].values.reshape(-1, 1))
    # transformed_s2 = pt.transform(second_sst['values'].values.reshape(-1, 1))
    # print(transformed_s2.shape, np.ravel(transformed_s2))
    # s2 = pd.DataFrame(data=np.ravel(transformed_s2), index=target_sst.index, columns=['values'])
    # print(s2)
    # p = mp.Pool(mp.cpu_count()-1)
    # answer  = p.starmap_async(evaluate_data_ar,zip([target_sst['3ts'], first_sst['prec1'], second_sst['prec2']] ,['3ts', 'prec1', 'prec2']))
    # result_3ts, result_prec1, result_prec2 =  answer.get()
    # p.close()
    # p.join()
    # const_ts, ar_ts = result_3ts
    # const_p1, ar_p1 = result_prec1
    # const_p2, ar_p2 = result_prec2
    # const_p1, ar_p1 = evaluate_data_ar(data=first_sst['prec1'],col='prec1', display=False, debug=True)
    # const_p2, ar_p2 = evaluate_data_ar(data=second_sst['prec2'],col='prec2',  display=False, debug=True)
    # const_ts, ar_ts = evaluate_data_ar(data=target_sst['3ts'], col='3ts', display=False, debug=True)

    # rho_ts, sigma_ts = evaluate_data_yule_walker(target_sst, col='3ts', order=2, method='mle', debug=True)
    # rho_1, sigma_1 = evaluate_data_yule_walker(first_sst, col='prec1', order=2, method='mle', debug=True)
    # rho_2, sigma_2 = evaluate_data_yule_walker(second_sst, col='prec2', order=2, method='mle', debug=True)

    
    # ar = [val for _,val in enumerate(ar_p2)]
  
    # pp(second_sst.values)
    # pp(transformed_ar)
    # print(f'P-value sklearn transform {normaltest(transformed_ar)[1][0]}')
    # fig, ax = plt.subplots(2,  1, figsize=(16, 8), dpi=90)
    # second_sst['values'].plot(ax=ax[0])
    # ax[1].plot(transformed_ar)
    # plt.show()
   
    # poly_p1= create_polynomial_fit_ar(ar=rho_1, sigma=first_sst.std(), data=first_sst, const=sigma_1, yule_walker=True, dependance=False)
    # _, poly_p1 = postprocess_ts(poly_p1, regression='ct', col='prec1')
    # poly_p1_d = create_polynomial_fit_ar_turbulance(ar=rho_1, sigma=first_sst.std(), data=first_sst, const=sigma_1, yule_walker=True, theta=1, nu=0.01)
    # _, poly_p1_d = postprocess_ts(poly_p1_d, regression='ct', col='prec1')
    # poly_p2= create_polynomial_fit_ar(ar=rho_2, sigma=second_sst.std(), data=second_sst, const=sigma_2, yule_walker=True, dependance=False)
    # _, poly_p2 = postprocess_ts(poly_p2, regression='ct', col='prec2')
    # poly_ts = create_polynomial_fit_ar(ar=rho_ts, sigma=target_sst.std(), data=target_sst, const=sigma_ts, yule_walker=True, dependance=False )
    # _, poly_ts = postprocess_ts(poly_ts, regression='ct', col='3ts')
    # # poly_dep_ = create_polynomial_fit_ar_depence(x0=poly_ts, x1=poly_p1, gamma=0.1, data=target_sst)
    # poly_dep_ = create_polynomial_fit_ar(ar=rho_ts, sigma=target_sst.std(), data=target_sst, const=sigma_ts, gamma=0.1, yule_walker=True, dependance=True, x1=poly_p1)
    # _, poly_dep_ = postprocess_ts(poly_dep_, regression='ct', col='dep')

    # display_poly_data_ar(simul_data=poly_p1,  ar=rho_1,  signal=first_sst, dep=False)
    # display_poly_data_ar(simul_data=poly_p1_d,  ar=rho_1,  signal=first_sst, dep=False)
    # display_poly_data_ar(simul_data=poly_p2,  ar=rho_2,  signal=second_sst, dep=False)
    # display_poly_data_ar(simul_data=poly_ts, ar=rho_ts,  signal=target_sst, dep=False)
    # display_poly_data_ar(simul_data=poly_dep_, ar=[rho_ts[0], rho_ts[1], 0.1], signal=target_sst, dep=True )
    # plt.show()

  
    # ar_t = np.load(os.path.join(ar_data_path, 'ar_sst_3ts_c.npz'))
    # ar_p1 = np.load(os.path.join(ar_data_path, 'ar_sst_prec1_c.npz'))
    # ar_p2 = np.load(os.path.join(ar_data_path, 'ar_sst_prec2_c.npz'))

    # const_sst_t, ar_sst_t = ar_t['const'] , ar_t['ar']
    # const_sst_p1, ar_sst_p1 = ar_p1['const'], ar_p1['ar']
    # cosnt_sst_p2, ar_sst_p2= ar_p2['const'] ,ar_p2['ar']

  
    # pp(ar_sst_p1)
    # pp(ar_sst_t)
    # print('---------------------')
    # poly_p1= create_polynomial_fit_ar(ar=ar_sst_p1, sigma=np.var(first_sst.values), data=first_sst, const=const_sst_p1, dependance=False)
    # poly_ts = create_polynomial_fit_ar(ar=ar_sst_t, sigma=np.var(target_sst.values), data=target_sst, const=const_sst_t, dependance=False )
    # poly_dep = create_polynomial_fit_ar_depence(x0=poly_ts, x1=poly_p1, gamma=0.1, data=target_sst)
    # poly_dep = create_polynomial_fit_ar(ar=ar_sst_t, sigma=np.var(target_sst.values), data=target_sst, const=const_sst_t,gamma=0.1, dependance=True, x1=poly_p1)
    # print('-------------------')
    # pp(poly_p1)
    # pp(poly_ts)
    # pp(poly_dep)
    # poly1 = create_polynomial_fit_ar(ar=arr, sigma=var, data=first_sst, const=const)
    # polyt = create_polynomial_fit_ar(ar=arrt, sigma=np.var(target['values'].values), data=target, const=constt)
    # polydep = create_polynomial_fit_ar_depence(x0 = poly1, x1=polyt, alpha=arr, beta=arrt, data=target)
    # pp(poly1)
    # pp(polyt)
    # pp(polydep)
    # display_poly_data_ar(simul_data=poly_dep, ar=[ar_sst_t[0], ar_sst_p1[0]], signal=target_sst)

        
        
    # DEBUG Synthetic data is to play and understand sampling process
    # evaluate_synthetic_data()

 
  
    # DBEUG TO VISUALY INVESTIGATE TS
    # fig, ax = plt.subplots(3,  1, figsize=(16, 8), dpi=90)
    # with pd.plotting.plot_params.use('x_compat', True):
    #     target.plot(title='target',  ax=ax[0])
    #     t_scale.plot(title='Scale ts', ax=ax[1])
    #     t_minmax.plot(title='Minmax', ax=ax[2])
    #     # first_sst.plot(title='prec1',  ax=ax[3])
    #     # f_scale.plot(title='Scale p1', ax=ax[4])
    #     # second_sst.plot(title='prec2', ax=ax[5])
    #     # s_scale.plot(title='Scale p2', ax=ax[6])
    #     plt.tight_layout(h_pad=1.5)
    #     plt.show()
    
    # DEBUG FOR STANDARDISATION VS NORMALISATION OF DATA
    # target = pd.read_csv(os.path.join(current_analysis_path, 'target.csv'), engine='python', index_col=[0,1])
    # t_scale = target.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    # t_minmax = target.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)  
    # first_sst = pd.read_csv(os.path.join(current_analysis_path, 'first_sst_prec.csv'), engine='python', index_col=[0, 1])
    # f_scale  = first_sst.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    # f_minmax = first_sst.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0) 
    # second_sst = pd.read_csv(os.path.join(current_analysis_path, 'second_sst_prec.csv'), engine='python', index_col=[0, 1])
    # s_scale = second_sst.apply(lambda x: (x-x.mean())/ x.std(), axis=0)
    # var_1 = np.var(first_sst['values'].values)
    # var_2 = np.var(second_sst['values'].values)
    # var_ts = np.var(target['values'].values)

    # var_1_s = np.var(f_scale['values'].values)
    # var_2_s = np.var(s_scale['values'].values)
    # var_ts_s = np.var(t_scale['values'].values)

    # var_1_mm = np.var(f_minmax['values'].values)
    # # stationarity_test(target['values'].values)
    # # stationarity_test(t_scale['values'])
    # # stationarity_test(t_minmax['values'])

    # p1 = stationarity_test(first_sst['values'].values)
    # ts = stationarity_test(target['values'].values)
    # p2 = stationarity_test(second_sst['values'].values)


    # const_p1, ar_p1 = evaluate_data_ar(data=first_sst, display=False, debug=True)
    # poly1 = create_polynomial_fit_ar(ar=ar_p1, sigma=var_1, data=first_sst, const=const_p1)
    # const_p1_s, ar_p1_s = evaluate_data_ar(data=f_scale, display=False, debug=True)
    # poly1_s = create_polynomial_fit_ar(ar=ar_p1_s, sigma=var_1_s, data=f_scale, const=const_p1_s)
    # const_p1_mm, ar_p1_mm = evaluate_data_ar(data=f_minmax, display=False, debug=True)
    # poly1_mm = create_polynomial_fit_ar(ar=ar_p1_mm, sigma=var_1_mm, data=f_minmax, const=const_p1_mm)
    # display_poly_data_ar(simul_data=poly1, signal=first_sst['values'], ar=ar_p1, title='Ar2 fit on original prec')
    # display_poly_data_ar(simul_data=poly1_s, signal=f_scale['values'], ar=ar_p1_s, title='Ar2 fit on standardise prec')
    # display_poly_data_ar(simul_data=poly1_mm, signal=f_minmax['values'], ar=ar_p1_mm, title='Ar2 fit on normalised prec')
    # plt.show()

    # DEBUG TO INVESTIGATE ARMA2AR VS PURE AR
    # if (p1 and p2 and ts) == True:
        
    #     # CHANGING FROM ARMA TO AR IS CREATE DISSAPEARING AR PROCESS THE SAME AS WHITENOISE, ONLY USE ARMA TO CHECK VARIABILITY OF PROCESS
    #     ar, ma, sigma, order = evaluate_data_arma(signal=first_sst, display=False, aic_check=True, debug=False)
        
    #     # UNNECESSARY
    #     # arr = st.tsa.arima_process.arma2ar(ma, ar, lags=len(first_sst))
    #     # const = (1 + arr.sum())/len(arr)
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings("ignore")
    #         marr = st.tsa.ar_model.AR(first_sst['values']).fit(ic='aic' ,method='mle', maxiter=len(first_sst), disp=0)
    #         const = marr.params[0]
    #         arr = marr.params[1:]
    #         arr_ = st.tsa.arima_process.arma2ar(ma, ar, lags=len(first_sst))
    #         const_ = (1 + arr_.sum())/len(arr_)
    #         # print(const, const_, arr, arr_, sep='\n')
    #         arr_s = create_polynomial_fit_ar(ar=arr, sigma=var_1, data=first_sst, const=const)
    #         arr_ss = create_polynomial_fit_ar(ar=arr_, sigma=var_1, data=first_sst, const=const_)
    #         plt.plot(arr_s, '-k', label='AR')
    #         plt.plot(arr_ss, 'r', label='ARMA2AR')
    #         plt.legend(loc=0)
    #         display_poly_data_ar(simul_data=arr_s, ar=arr, signal=first_sst, title='AR2 fit on prec original')
    #         display_poly_data_ar(simul_data=arr_ss, ar=arr_, signal=first_sst, title='AR2 fit from ARMA2AR')
    #         plt.show()
            
            # DEBUG investigate fittedvalues  from ARMA/AR process 

            # fig, ax = plt.subplots(3, 1, figsize=(19,8))
            # first_sst.plot(label='original', ax=ax[0])
            # marr.fittedvalues.plot(label='Fitted', ax=ax[1])
            # plt.legend(loc=0)
            # plt.show()
            # display_poly_data_arma(simul_data=polyARMA, ar=ar, ma=ma, order=order, signal=second_sst)
            # display_poly_data_ar(simul_data=poly1,signal=first_sst, ar=arr)
            # display_poly_data_ar(simul_data=polyt, signal=target, ar=arrt)
            # display_poly_data_ar(simul_data=poly1, ar=arr, signal=first_sst)
            # display_poly_data_ar(simul_data=polydep,signal=target, ar=[arr[0], arrt[0]])
            # plt.show()
        


    
    



