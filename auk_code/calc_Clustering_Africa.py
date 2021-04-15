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
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
  
import clustering_spatial as cl
from RGCPD import RGCPD
import plot_maps

rg = RGCPD() 

rg.pp_precursors()

var_filename = rg.list_precur_pp[0][1]
print(var_filename)
mask = [0.0, 360.0, -45.0, 50.0]
q = 0.85
n_clusters = [2, 3]
xrclustered, results = cl.dendogram_clustering(var_filename, mask=mask, kwrgs_clust={'q':q, 'n_clusters':n_clusters})
fig = plot_maps.plot_labels(xrclustered, {'col_dim':'n_clusters'})