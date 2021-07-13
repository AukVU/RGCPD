# -*- coding: utf-8 -*-

#%%

import os, inspect, sys
import matplotlib.pyplot as plt

#curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
#main_dir = '/'.join(curr_dir.split('/')[:-1])
main_dir ='C:/Users/Auk/Documents/GitHub/RGCPD'
print(main_dir)

SPI_func_dir = 'C:/Users/Auk/Documents/GitHub/RGCPD/auk_code/calc_SPI_Africa.py'
RGCPD_func = os.path.join(main_dir, 'RGCPD')
if RGCPD_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(SPI_func_dir)
    # sys.path.append(RGCPD_func)

from RGCPD import core_pp, functions_pp
import func_SPI

path_raw = os.path.join(main_dir, 'data')
p_filepath = 'C:/Users/Auk/Documents/GitHub/RGCPD/data/tp_1950-2020_1_12_monthly_1.0deg.nc'
# USBox = (240, 250, 40, 45)
ds = core_pp.import_ds_lazy(p_filepath, auto_detect_mask=False)

aggr = 12
output = func_SPI.calc_SPI_from_monthly(ds, aggr)
output = core_pp.detect_mask(output)
output.name = f'SPI{aggr}'
output.to_netcdf(functions_pp.get_download_path() + f'/Own_SPI_{aggr}.nc')

# filepath = 'C:/Users/Auk/Documents/GitHub/RGCPD/data/tp_world_SPI_3.nc'
# output = core_pp.import_ds_lazy(filepath)

# SMI_package_filepath = os.path.join(path_raw, '/Users/semvijverberg/surfdrive/ERA5/SM_spi_gamma_01_1950-2019_1_12_monthly_1.0deg.nc')
# SMI_package = core_pp.import_ds_lazy(SMI_package_filepath)
latitude = 7.5 ; longitude =227.5 
ts_raw = ds.sel(latitude=latitude, longitude=longitude)
ts_stn = output.sel(latitude=latitude, longitude=longitude)
# ts_pack = SMI_package.sel(latitude=40, longitude=240)

fig = plt.figure(figsize=(20,10) )

# plot observed versus corresponding Gamma probability
ax1 = plt.subplot(1, 2, 1)
vals = ax1.plot(ts_raw, '.k');
# ax1.set_ylabel('Cumulative probability (from Gamma distribution)')
# ax1.set_xlabel('Aggregated Precipitation [mm] - Original values')
# plot transformed standard normal from Gamma probability
ax1 = plt.subplot(1, 2, 2)
vals = ax1.plot(ts_stn, '.k');
# ax1.plot(ts_pack, '.r')
# ax1.set_ylabel('Cumulative probability (from Gamma distribution)')
ax1.set_xlabel('SPI - Gamma prob. transformed to standard normal ')

