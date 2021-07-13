# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 15:10:58 2021

@author: Auk
"""

c = 11 ; q = 65
ds = cl.spatial_mean_clusters(var_filename,
                              xrclustered.sel(q=q, n_clusters=c),
                              selbox=selbox)
f_name = 'q{}_nc{}'.format(int(q), int(c)) 
ds['xrclusteredall'] = xrclustered
filepath = os.path.join(rg.path_outmain, f_name)
cl.store_netcdf(ds, filepath=filepath, append_hash='dendo_'+xrclustered.attrs['hash'])
TVpath = filepath + '_' + 'dendo_'+xrclustered.attrs['hash'] + '.nc'