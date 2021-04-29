# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 12:01:31 2021

@author: Auk
"""

import os, inspect, sys
# main_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
main_dir ='C:/Users/Auk/Documents/GitHub/RGCPD'

print(main_dir)                           # script directory
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering')
if RGCPD_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
  
import clustering_spatial as cl
from RGCPD import RGCPD
from RGCPD import core_pp
import cartopy.crs as ccrs

import plot_maps
list_of_name_path = [('fake', None),('tp_SPI','C:/Users/Auk/Documents/GitHub/RGCPD/data/tp_world_SPI_12.nc')]
rg = RGCPD(list_of_name_path=list_of_name_path)



rg.pp_precursors()

var_filename = rg.list_precur_pp[0][1]
var_filename = 'C:/Users/Auk/Documents/GitHub/RGCPD/data/sst_1950-2020_1_12_monthly_1.0deg.nc' 
ds = core_pp.import_ds_lazy(var_filename)
print(ds.latitude)



print(var_filename)
# mask = (0,360, -20.5, 40.5)
# q = [0.66, 0.75, 0.85]
# n_clusters = [10,15,20]
# xrclustered, results = cl.dendogram_clustering(var_filename,mask=mask, kwrgs_clust={'q':q, 'n_clusters':n_clusters})

# print(xrclustered.latitude)

# fig = plot_maps.plot_labels(xrclustered, {'col_dim':'n_clusters', 'map_proj' : ccrs.LambertCylindrical(central_longitude='cen_lon')} )
import xarray as xr
xrclustered  = xr.DataArray()
import clustering_spatial as cl

var_filename ='C:/Users/Auk/Documents/GitHub/RGCPD/data/tp_world_SPI_12_clustered.nc'
c = 20; q = 85
ds = cl.spatial_mean_clusters(var_filename,xrclustered.sel(q=q, n_clusters=c))
f_name = 'q{}_nc{}'.format(int(q), int(c)) 
ds['xrclusteredall'] = xrclustered
filepath = os.path.join(rg.path_outmain, f_name)
cl.store_netcdf(ds, filepath=filepath, append_hash='dendo_'+xrclustered.attrs['hash'])
TVpath = filepath + '_' + 'dendo_'+xrclustered.attrs['hash'] + '.nc'

cl.store_netcdf(xrclustered, filepath='C:/Users/Auk/Documents/GitHub/RGCPD/data/tp_world_SPI_12_clustered.nc')