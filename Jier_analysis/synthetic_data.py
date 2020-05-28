import os, sys, inspect, warnings
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
sub_dir = os.path.join(main_dir, 'RGCPD/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(sub_dir)
import numpy as np 
import statsmodels.api as sm 
import pandas as pd 
import matplotlib.pyplot as plt 
import itertools as it
from statsmodels.tsa.arima_process import  arma_generate_sample, ArmaProcess
from pprint import pprint as pp 

np.random.seed(12345)

# TODO Look if simulation of more than one realization is needed and if so why, comment it. 
# Above is only needed when we want to test ground truth data scenarios against our target data. 
# TODO Visualise and modularise steps
# TODO Fit actual data to arma process

synthetic = dict()
synthetic['ARrange'] = np.array([.75, -.25])
synthetic['MArange'] = np.array([.65, .35])
synthetic['arparams'] = np.r_[1, -synthetic['ARrange']]
synthetic['maparams'] = np.r_[1, -synthetic['MArange']]
synthetic['nobs'] = 500
synthetic['startyear'] = '1980m1'

def generate_synthetic_data():
    return arma_generate_sample(synthetic['arparams'],synthetic['maparams'], synthetic['nobs'])

def generate_date_range():
    return sm.tsa.datetools.dates_from_range(synthetic['startyear'], length=synthetic['nobs'])

def generate_combination_lags(end_range=4):
        return list(it.product(range(1,end_range), repeat=2))

def extract_lags_aic_bic_info_synth(y:pd.Series, combos:list):
    print('Starting manually..')
    summaries = []
    aic, bic = 10000000000, 10000000000
    ar_ma_params, aci_info, arma_res = [] , [], None
    for i, j in combos:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            arma_mod = sm.tsa.ARIMA(y, order=(i,0,j))
            arma_res = arma_mod.fit(trend='c', disp=-1)
        if arma_res.aic < aic and arma_res.bic < bic: 
            aic = arma_res.aic
            bic = arma_res.bic
            ar_ma_params.append((arma_res.arparams, arma_res.maparams,arma_res.sigma2,(i,j)))
            aci_info.append((aic, bic, (i, j),(arma_res.arroots, arma_res.maroots)))
            summaries.append(arma_res.summary())
    print('Done manually')
    return ar_ma_params, aci_info, summaries

def retrieve_aci_info(aci_info:list):
    print("AIC, BIC, (i, j), ARroots, MAroots", sep='\t')
    pp(aci_info)

def retrieve_ar_ma_info(ar_ma_params:list):
    print("ARparams, MAparams, Sigmas")
    pp(ar_ma_params)

def retrieve_summaries(summaries:list):
    pp(summaries)

def check_stationarity_invertibility(aci_info:list, ar_ma_params:list, check_info_score:list=[], get_params:bool=False):
    arma_process = []
    for i in range(len(aci_info)):
        arma_process.append((ar_ma_params[i], ArmaProcess(ar=ar_ma_params[i][:2][0], ma=ar_ma_params[i][:2][1], nobs=synthetic['nobs'])))
        print("Statioarity ", arma_process[i][-1].isstationary, "Invertibility ", arma_process[i][-1].isinvertible, ar_ma_params[i][-1])

    if len(check_info_score) > 0:
        aic_check, bic_check = False, False 
        idx_pr = None
        temp = [aci_info[i][:3] for i in range(len(aci_info))]
        if 'aic' in check_info_score:
            man_aic  =[temp[i][0] for i in range(len(temp))]
            idx_aic = next( (i for i , v in enumerate(temp) if v[0] == min(man_aic)),-1 )
            idx_pr = next((i for i , v in enumerate(arma_process) if v[0][-1] == temp[idx_aic][-1]), -1)
            print('AIC stationarity and Invertibility ', arma_process[idx_pr][:-1][0])
            aic_check = True

        if 'bic' in check_info_score:
            man_bic = [temp[i][1] for i in range(len(temp))]
            idx_bic = next( (i for i , v in enumerate(temp) if v[0] == min(man_bic)),-1 )
            idx_pr = next((i for i , v in enumerate(arma_process) if v[0][-1] == temp[idx_bic][-1]), -1)
            print('BIC stationarity and Invertibility ', arma_process[idx_pr][:-1][0])
            bic_check = True

        if get_params == True:
            return arma_process[idx_pr][:-1][0]


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
        assert isinstance(y, pd.Series), "Expect pandas Series"
        serie = y
        ar_ma_params, aci_info, summaries = extract_lags_aic_bic_info_synth(y=serie, combos=combos)

    return ar_ma_params, aci_info, summaries

    
def fit_automatically_data(y:pd.Series, pmax:int, qmax:int, ic:list=['aic','bic']):
    try:
        print('Starting automatic...')
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            order_model =  sm.tsa.stattools.arma_order_select_ic(y, ic=ic, trend='c', max_ar=pmax, max_ma=qmax)

        aut_model, aut_order = None, []
        if 'aic' in ic:
            aut_model = sm.tsa.ARMA(y, order=order_model.aic_min_order).fit(trend='c', disp=-1)
            aut_order.append((aut_model.aic, aut_model.bic, order_model.aic_min_order))
        if 'bic' in ic:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                aut_model = sm.tsa.ARMA(y, order=order_model.bic_min_order).fit(trend='c', disp=-1)
            aut_order.append((aut_model.aic, aut_model.bic, order_model.bic_min_order))

    except ValueError as err:
        print("Occured errors: ", err)
    print("Done Automatic")
    return aut_model, aut_order

def determine_order(man_model:list, aut_model:list, check_info_score:list=['aic']):

    temp = [man_model[i][:3] for i in range(len(man_model))]
    man_aic = [temp[i][0] for i in range(len(temp))] 
    man_bic = [temp[i][1] for i in range(len(temp))]
    aut_aic, aut_bic, aut_order = aut_model[0]

    idx_aic, idx_bic = None, None
    aut_aic_check, aut_bic_check = False, False
    aic_check, bic_check = False, False

    if len(check_info_score) > 0:
        if 'aic' in check_info_score:
            if min(man_aic) <= aut_aic:
                idx_aic = next( (i for i , v in enumerate(temp) if v[0] == min(man_aic)),-1 )
                print("Small AIC mannually ", temp[idx_aic])
                aic_check = True
            elif aut_aic < min(man_aic):
                print("Small AIC automatically ", aut_aic, aut_order)
                aut_aic_check = True
        if 'bic' in check_info_score:
            if min(man_bic) <= aut_bic:
                idx_bic = next( (i for i , v in enumerate(temp) if v[0] == min(man_bic)),-1 )
                print("Small BIC mannually ", temp[idx_bic])
                bic_check = True
            elif aut_bic < min(man_bic):
                print("Small BIC automatically ", aut_bic, aut_order)
                aut_bic_check = True

    if aut_bic_check == True or aut_aic_check == True:
        return aut_model[0]
    if aic_check == True: 
        return temp[idx_aic]
    elif bic_check == True:
        return temp[idx_bic]
    

def display_info_ts(y:pd.Series, figsize=(14, 8), title="", lags=20):
    assert isinstance(y, pd.Series), " Expect pandas Series"
    serie = y 
    fig = plt.figure(figsize=figsize)
    serie.plot(ax = fig.add_subplot(3, 1, 1), title="$Time \ Series \ "+ title + "$", legend=False)
    sm.graphics.tsa.plot_acf(serie, lags=lags, zero=False, ax = fig.add_subplot(3, 2, 3))
    plt.xticks(np.arange(1, lags + 1, 1.0))

    sm.graphics.tsa.plot_pacf(serie, lags=lags, zero=False, ax = fig.add_subplot(3, 2, 4))
    plt.xticks(np.arange(1, lags + 1, 1.0))

    sm.qqplot(serie, line='s', ax = fig.add_subplot(3, 2, 5))

    fig.add_subplot(326).hist(serie, bins= 40, density=True)
    plt.tight_layout()
    plt.show()


def display_pierce_LJbox(y:pd.Series, dates:pd.DatetimeIndex, figsize=(14, 8), title="", lags=20):
    assert isinstance(y, pd.Series), " Expect pandas Series"

    y.index = dates + pd.Timedelta(1, unit=dates.freqstr)
    acor_ljungbox = list(sm.stats.diagnostic.acorr_ljungbox(y, lags=lags, boxpierce=True))

    plt.plot(range(1, len(acor_ljungbox[0]) + 1), acor_ljungbox[1], 'bo', label= 'Ljung-Box values')
    plt.plot(range(1, len(acor_ljungbox[0]) + 1), acor_ljungbox[3], 'go', label= 'Box-Pierce values')
    plt.xticks(np.arange(1, len(acor_ljungbox[0]) + 1, 1.0))
    plt.axhline(y = 0.05, color = 'red', label= "$5 \%  critical value$")
    plt.title("$Time\ Serie\ " + title + " $")
    plt.legend()
    plt.show()
    column_ind = ["Ljung-Box: X-squared", 'Ljung-Box: p-value', 'Box-Pierce: X-squared', 'Box-Pierce: p-value']
    return pd.DataFrame(acor_ljungbox, index=column_ind, columns= range(1, len(acor_ljungbox[0]) + 1))

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

def evaluate_data(current_analysis_path, display:bool=False):
    # TODO INDUCE STATIONARITY IN ORIGINAL DATA TIME SERIES 

    # target = pd.read_csv(os.path.join(current_analysis_path, 'target.csv'), engine='python', index_col=[0,1])
    # second_sst = pd.read_csv(os.path.join(current_analysis_path, 'second_sst_prec.csv'), engine='python', index_col=[0, 1])
    # print(type(target.index[0][1]))
    # ar_ma_params, aci_info, summaries = fit_manually_data(target['values'], arparams=np.array([.75, -.25]), maparams=np.array([.65, .35]), nobs=synthetic['nobs'], startyear=target.index[0][1], end_range=4,synthetics=False)
    # ar_ma_params, aci_info, summaries = fit_manually_data(second_sst['values'], arparams=np.array([.75, -.25]), maparams=np.array([.65, .35]), nobs=synthetic['nobs'], startyear=second_sst.index[0][1], end_range=4,synthetics=False)

    first_sst = pd.read_csv(os.path.join(current_analysis_path, 'first_sst_prec.csv'), engine='python', index_col=[0, 1])
    ar_ma_params, aci_info, summaries = fit_manually_data(first_sst['values'], arparams=np.array([.75, -.25]), maparams=np.array([.65, .35]), nobs=synthetic['nobs'], startyear=first_sst.index[0][1], end_range=4,synthetics=False)
    _, aut_order = fit_automatically_data(y=first_sst['values'], pmax=4, qmax=4, ic=['bic'])

    # pp(summaries)
    # # # retrieve_aci_info(aci_info)
    # # # pp(aut_order)
    # # # print(aci_info[0], len(aci_info), len(aci_info[0]))

    _ = determine_order(aci_info, aut_order,check_info_score=['aic'])
    ar, ma, sigma, _ = check_stationarity_invertibility(aci_info, ar_ma_params,check_info_score=['aic'],get_params=True)

    if display == True:
        pass
        # TODO FIND WAY TO CHECK FOR MULTIPLE FOLDS OF MULTIINDEX AND CREATE MUTLIPLE ARIMA MODELS
        # temp_dates = pd.DatetimeIndex(first_sst.index.levels[1], freq='infer')
        # temp_m = pd.DataFrame(data=first_sst['values'].values, index=temp_dates)
        # model = sm.tsa.ARMA(temp_m, order=order[-1]).fit(trend='c', disp=-1)
        # display_info_ts(model.resid, title='Fitted  RGCPD precursor')
        # display_pierce_LJbox(model.resid, dates=temp_dates, title='Fitted RGCPD precursor')
        # display_info_ts(first_sst['values'], title='OriginalPrecursor')
        # display_pierce_LJbox(first_sst['values'], dates=temp_dates, title='OriginalPrecursor')

    return ar, ma, sigma, first_sst
if __name__ == "__main__":
    current_analysis_path = os.path.join(main_dir, 'Jier_analysis')
    # evaluate_synthetic_data()
    ar, ma, sigma, first_sst = evaluate_data(current_analysis_path=current_analysis_path)

    N =  synthetic['nobs']
    epsilon = np.random.normal(loc=0, scale=sigma, size=N)
    simul_data = np.array(epsilon[0], ndmin= 1)
    simul_data = np.append(simul_data, ar[0] * simul_data + epsilon[1] + ma[0] * epsilon[0])
    for i in range(2, N):
        temp = ar[0] * simul_data[ i - 1] + ar[1] * simul_data[i  - 2] + epsilon[i] + ma[0] * epsilon[i - 1] + ma[1] * epsilon[i - 2]
        simul_data= np.append(simul_data, temp)
    
    plt.figure(figsize=(14, 8))
    plt.plot(simul_data, label='$Y_t = 1.74Y_{t-1} -0.74Y_{t-2} -0.76 \\epsilon_{t -1} - 0.11 \\epsilon_{t-2}$')
    plt.plot(first_sst['values'].values, label='Precursor data')
    plt.xlim(0.01, N)
    plt.ylim(-0.5, 0.5)
    plt.title('Chosen model params with original data')
    plt.legend()
    plt.show()
    



