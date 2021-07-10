# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 14:33:46 2021

@author: Auk
"""

import os, inspect, sys
# main_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir ='C:/Users/Auk/Documents/GitHub/RGCPD'
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering')
if RGCPD_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    
    
import clustering_spatial as cl
from RGCPD import RGCPD
import plot_maps
import core_pp 
 

# define input
list_of_name_path = [('fake', None),('SPI3','C:/Users/Auk/Documents/GitHub/RGCPD/data/tp_world_SPI_3.nc')]
# use RGCPD to pre-process data
rg = RGCPD(list_of_name_path=list_of_name_path)

rg.pp_precursors()
# list_precur_pp contains name and filepath to pre-processed data 
rg.list_precur_pp

var_filename = rg.list_precur_pp[0][1]

mask = [0,360, -20.5, 40.5]
q = [85]
n_clusters = [20]

xrclustered, results = cl.dendogram_clustering(var_filename, mask=mask, kwrgs_clust={'q':q, 'n_clusters':n_clusters})
fig = plot_maps.plot_labels(xrclustered, {'col_dim':'n_clusters', 'title':'Hierarchical Clustering'})

# var_filename = rg.list_precur_pp[0][1]
# mask = [0,360, -20.5, 40.5]

# xrclustered, results = cl.sklearn_clustering(var_filename, mask=mask, clustermethodkey='DBSCAN', kwrgs_clust={'eps':[90, 100]})
# xrclustered += 1
# fig = plot_maps.plot_labels(xrclustered, {'col_dim':'eps', 'title':'Density-Based Spatial Clustering of Applications with Noise'})

# var_filename = rg.list_precur_pp[0][1]
# var_filename = 'C:/Users/Auk/Documents/GitHub/RGCPD/data/tp_world_SPI_3.nc' 
# ds = core_pp.import_ds_lazy(var_filename)


xrclustered.dims

# select parameter(s)
n_clusters = 20
q=85
final_cluster_output = xrclustered.sel(n_clusters=n_clusters).copy()
final_cluster_output = final_cluster_output.sel(q=q,).copy()
# spatial_mean_clusters will calculate an area-weighted spatial mean timeseries on the timescale and the dates 
# of the input dataset, i.e. in this case on daily timescale and year-round. 
ds = cl.spatial_mean_clusters(var_filename,
                              final_cluster_output,
                              selbox=None)
f_name = 'q{}'.format(q)
ds['xrclusteredall'] = xrclustered
filepath = os.path.join(rg.path_outmain, f_name)
# To store, uncomment line below
cl.store_netcdf(ds, filepath=filepath, append_hash='dendo_'+xrclustered.attrs['hash']) 
ds 
