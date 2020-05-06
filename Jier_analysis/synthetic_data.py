import os, sys, inspect
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
sub_dir = os.path.join(main_dir, 'RGCPD/')
if main_dir not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(sub_dir)
import numpy as np 
import statsmodels.api as sm 
import pandas as pd 
from statsmodels.tsa.arima_process import  arma_generate_sample, arma_pacf, arma_periodogram 
np.random.seed(12345)

path_data = os.path.join(main_dir, 'data')
target= 1
list_of_name_path = [(target, os.path.join(path_data, 'tf5_nc5_dendo_80d77.nc')),
                    ('sst', os.path.join(path_data,'sst_1979-2018_2.5deg_Pacific.nc'))]


arparams = np.array([.75, -.25])
maparams = np.array([.65, .35])

arparams = np.r_[1, -arparams]
maparams = np.r_[1, maparams]
nobs = 500
y = arma_generate_sample(arparams, maparams, nobs)

dates = sm.tsa.datetools.dates_from_range('1980m1', length=nobs)
y = pd.Series(y, index=dates)
arma_mod = sm.tsa.ARMA(y, order=(2,2))
arma_res = arma_mod.fit(trend='nc', disp=-1)

print(arma_res.summary())