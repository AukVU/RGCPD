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
            summaries.append(((i, j),arma_res.summary()))
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

def check_stationarity_invertibility(aci_info:list, ar_ma_params:list, check_info_score:list=[]):
    arma_process = []
    for i in range(len(aci_info)):
        arma_process.append((ArmaProcess(ar=ar_ma_params[i][:2][0], ma=ar_ma_params[i][:2][1], nobs=synthetic['nobs']),ar_ma_params[i][-1]))
        print("Statioarity ", arma_process[i][0].isstationary, "Invertibility ", arma_process[i][0].isinvertible, ar_ma_params[i][-1])

    if len(check_info_score) > 0:
        temp = [aci_info[i][:3] for i in range(len(aci_info[0]))]
        if 'aic' in check_info_score:
            man_aic  =[temp[i][0] for i in range(len(temp))]
            idx_aic = next( (i for i , v in enumerate(temp) if v[0] == min(man_aic)),-1 )
            idx_pr = next((i for i , v in enumerate(arma_process) if v[1] == temp[idx_aic][-1]), -1)
            print(arma_process[idx_pr])
        elif 'bic' in check_info_score:
            man_bic = [temp[i][1] for i in range(len(temp))]
            idx_bic = next( (i for i , v in enumerate(temp) if v[0] == min(man_bic)),-1 )
            idx_pr = next((i for i , v in enumerate(arma_process) if v[1] == temp[idx_bic][-1]), -1)
            print(arma_process[idx_pr])


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

    temp = [man_model[i][:3] for i in range(len(man_model[0]))]
    man_aic = [temp[i][0] for i in range(len(temp))] 
    man_bic = [temp[i][1] for i in range(len(temp))]
    aut_aic, aut_bic, aut_order = aut_model[0]

    if len(check_info_score) > 0:
        if 'aic' in check_info_score:
            if min(man_aic) < aut_aic:
                idx = next( (i for i , v in enumerate(temp) if v[0] == min(man_aic)),-1 )
                print("Small AIC mannually ", temp[idx])
                return temp[idx]
            elif aut_aic < min(man_aic):
                print("Small AIC automatically ", aut_aic, aut_order)
                return (aut_aic, aut_bic, aut_order)
        elif 'bic' in check_info_score:
            if min(man_bic) < aut_bic:
                idx = next( (i for i , v in enumerate(temp) if v[0] == min(man_bic)),-1 )
                print("Small AIC mannually ", temp[idx])
                return temp[idx]
            elif aut_bic < min(man_bic):
                print("Small AIC automatically ", aut_bic, aut_order)
                return (aut_aic, aut_bic, aut_order) 

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

    fig.add_subplot(326).hist(serie, bins= 40, normed= 1)
    plt.tight_layout()
    plt.show()


def display_pierce_LJbox(y, dates, figsize=(14, 8), title="", lags=20):
    serie = pd.Series(y)
    serie.index = dates + pd.Timedelta(1, unit=dates.freqstr)
    acor_ljungbox = list(sm.stats.diagnostic.acorr_ljungbox(serie, lags=lags, boxpierce=True))

    plt.plot(range(1, len(acor_ljungbox[0]) + 1), acor_ljungbox[1], 'bo', label= 'Ljung-Box values')
    plt.plot(range(1, len(acor_ljungbox[0]) + 1), acor_ljungbox[3], 'go', label= 'Box-Pierce values')
    plt.xticks(np.arange(1, len(acor_ljungbox[0]) + 1, 1.0))
    plt.axhline(y = 0.05, color = 'red', label= "$5 \%  critical value$")
    plt.title("$Time\ Serie\ " + title + " $")
    plt.legend()
    plt.show()
    column_ind = ["Ljung-Box: X-squared", 'Ljung-Box: p-value', 'Box-Pierce: X-squared', 'Box-Pierce: p-value']
    return pd.DataFrame(serie, index=column_ind, columns= range(1, len(acor_ljungbox[0]) + 1))



if __name__ == "__main__":
    current_analysis_path = os.path.join(main_dir, 'Jier_analysis')
    # dates = generate_date_range()
    # dates = pd.DatetimeIndex(dates, freq='infer')
    # y = generate_synthetic_data()
    # y = pd.Series(y, index=dates)

    # ar_ma_params, aci_info, summaries = fit_manually_data(y=None, arparams=np.array([.75, -.25]), maparams=np.array([.65, .35]), nobs=synthetic['nobs'], startyear=synthetic['startyear'], end_range=4)
    # aut_model, aut_order = fit_automatically_data(y=y, pmax=4, qmax=4, ic=['bic'])

    # order = determine_order(aci_info, aut_order)
    # check_stationarity_invertibility(aci_info, ar_ma_params)
    # model = sm.tsa.ARMA(y, order=order[-1]).fit(trend='c', disp=-1)

 
    # # # display_info_ts(pd.Series(y), title='Synthetic generated')
    # display_pierce_LJbox(model.resid, dates=dates, title='Synthetic generated')

    target = pd.read_csv(os.path.join(current_analysis_path, 'target.csv'), engine='python', index_col=[0])
    print(target.head(2))
