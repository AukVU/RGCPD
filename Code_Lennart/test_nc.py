from netCDF4 import Dataset
import netCDF4 as nc4
import xarray as xr

import numpy as np

lon = np.arange(45,101,2)
lat = np.arange(-30,25,2.5)
z = np.arange(0,200,10)
x = np.random.randint(10,25, size=(len(lon), len(lat), len(z)))
noise = np.random.rand(len(lon), len(lat), len(z))
temp_data = x+noise
filename = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD/Code_Lennart/NC/sample.nc'

f = nc4.Dataset(filename,'w', format='NETCDF4') #'w' stands for write

# tempgrp = f.createGroup('TEST')
# tempgrp.createDimension('lon', len(lon))
# tempgrp.createDimension('lat', len(lat))
# tempgrp.createDimension('z', len(z))
# tempgrp.createDimension('time', None)

f.createDimension('lon', len(lon))
f.createDimension('lat', len(lat))
f.createDimension('z', len(z))
f.createDimension('time', None)

# longitude = tempgrp.createVariable('Longitude', 'f4', 'lon')
# latitude = tempgrp.createVariable('Latitude', 'f4', 'lat')  
# levels = tempgrp.createVariable('Levels', 'i4', 'z')
# temp = tempgrp.createVariable('Temperature', 'f4', ('time', 'lon', 'lat', 'z'))
# time = tempgrp.createVariable('Time', 'i4', 'time')

longitude = f.createVariable('longitude', 'f4', 'lon')
latitude = f.createVariable('latitude', 'f4', 'lat')  
levels = f.createVariable('kevels', 'i4', 'z')
temp = f.createVariable('temperature', 'f4', ('time', 'lon', 'lat', 'z'))
time = f.createVariable('time', 'i4', 'time')

time.units = 'hours since 1900-01-01 00:00:00.0'
time.calendar = 'gregorian'

longitude[:] = lon #The "[:]" at the end of the variable instance is necessary
latitude[:] = lat
levels[:] = z
temp[0,:,:,:] = temp_data

#get time in days since Jan 01,01
from datetime import datetime
today = datetime.today()
time_num = today.toordinal()
time[0] = time_num









f.close()

ds = xr.open_dataset(filename, decode_cf=False, decode_coords=True, decode_times=False)

print('hoi')