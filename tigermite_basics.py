import sys
import warnings
import numpy as np 
import sklearn
import matplotlib.pyplot as plt 
plt.style.use('ggplot')

import tigramite
from tigramite import data_processing as pp 
from tigramite import plotting as tp 
from tigramite.pcmci import PCMCI 
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb  
# if not sys.warnoptions:
#     import os
#     warnings.simplefilter("default")
#     os.environ["PYTHONWARNINGS"] = "default" 
# Inspired from https://github.com/jakobrunge/tigramite/blob/master/tutorials/tigramite_tutorial_basics.ipynb

np.random.seed(42)     # Fix random seed
# links_coeffs = {0: [((0, -1), 0.7), ((1, -1), -0.8)],
#                 1: [((1, -1), 0.8), ((3, -1), 0.8)],
#                 2: [((2, -1), 0.5), ((1, -2), 0.5), ((3, -3), 0.6)],
#                 3: [((3, -1), 0.4)],
#                 }
# T = 1000     # time series length
# data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)
# T, N = data.shape

# # Initialize dataframe object, specify time axis and variable names
var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']
# dataframe = pp.DataFrame(data, 
#                          datatime = np.arange(len(data)), 
#                          var_names=var_names)

# tp.plot_timeseries(dataframe)

parcorr = ParCorr(significance='analytic')
# pcmci = PCMCI(
#     dataframe=dataframe, 
#     cond_ind_test=parcorr,
#     verbosity=1)

# correlations = pcmci.get_lagged_dependencies(tau_max=20)
# lag_func_matrix = tp.plot_lagfuncs(val_matrix=correlations, setup_args={'var_names':var_names, 
#                                     'x_base':5, 'y_base':.5})
# '''
# Since the dependencies decay beyond a maximum lag of around 8, we choose tau_max=8 for PCMCI. 
# The other main parameter is pc_alpha which sets the significance level in the condition-selection step. 
# Here we let PCMCI choose the optimal value by setting it to pc_alpha=None. 
# Then PCMCI will optimize this parameter in the ParCorr case by the Akaike Information criterion among a reasonable default list of values 
# (e.g., pc_alpha = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]).
# '''
# pcmci.verbosity = 1
# results = pcmci.run_pcmci(tau_max=8, pc_alpha=None) 
# '''
# As you can see from the output, PCMCI selected different pc_alpha for each variable. The result of run_pcmci is a dictionary containing the matrix of p-values, 
# the matrix of test statistic values (here MCI partial correlations), and optionally its confidence bounds (can be specified upon initializing ParCorr). 
# p_matrix and val_matrix are of shape (N, N, tau_max+1) with entry (i, j, \tau) denoting the test for the link $X^i_{t-\tau} \to X^j_t$. 
# The MCI values for $\tau=0$ do not exclude other contemporaneous effects, only past variables are conditioned upon.
# '''

# print("p-values")
# print (results['p_matrix'].round(3))
# print("MCI partial correlations")
# print (results['val_matrix'].round(2))

# #  Test result matrix to correct p-values by for example using False Discovery Rate test
# q_matrix = pcmci.get_corrected_pvalues(p_matrix=results['p_matrix'], fdr_method='fdr_bh')
# pcmci.print_significant_links(
#         p_matrix = results['p_matrix'], 
#         q_matrix = q_matrix,
#         val_matrix = results['val_matrix'],
#         alpha_level = 0.01)

# # Visualisation 

# link_matrix = pcmci.return_significant_parents(pq_matrix=q_matrix,
#                         val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']

# # # Process graph
# tp.plot_graph(
#     val_matrix=results['val_matrix'],
#     link_matrix=link_matrix,
#     var_names=var_names,
#     link_colorbar_label='cross-MCI',
#     node_colorbar_label='auto-MCI',
#     )


# # Time series graph with visible spatio-temporal structure form
# tp.plot_time_series_graph(
#     val_matrix=results['val_matrix'],
#     link_matrix=link_matrix,
#     var_names=var_names,
#     link_colorbar_label='MCI',
#     )

# Non linear conditional independence tests

new_data = np.random.randn(500, 3)
for t in range(1, 500):
    new_data[t, 0] += 0.4 * new_data[t - 1, 1]**2
    new_data[t, 2] += 0.3 * new_data[t  - 2, 1]**2
new_df = pp.DataFrame(new_data, var_names=var_names)
# tp.plot_timeseries(new_df)
# plt.show()

new_pcmi_parcorr = PCMCI(dataframe=new_df, cond_ind_test=parcorr, verbosity=0)
new_results = new_pcmi_parcorr.run_pcmci(tau_max=2, pc_alpha=0.2)
# new_pcmi_parcorr.print_significant_links(p_matrix=new_results['p_matrix'], val_matrix=new_results['val_matrix'], alpha_level= 0.01)
'''
Tigramite covers nonlinear additive dependencies with a test based on Gaussian process regression and a distance correlation (GPDC) on the residuals. For GPDC no analytical null distribution of the distance correlation (DC) is available. For significance testing, Tigramite with the parameter significance = 'analytic' pre-computes the distribution for each sample size (stored in memory), thereby avoiding computationally expensive permutation tests for each conditional independence test (significance = 'shuffle_test'). If needed, the null distribution can be pre-computed for different anticipated sample sizes with generate_and_save_nulldists and stored to disk (could be run overnight:). Then significance='analytic' loads this file. GP regression is performed with sklearn default parameters, except for the kernel which here defaults to the radial basis function + a white kernel (both hyperparameters are internally optimized) and the assumed noise level alpha which is set to zero since we added a white kernel. These and other parameters can be set via the gp_params dictionary. See the documentation in sklearn for further discussion

'''

gpdc = GPDC(significance='analytic', gp_params=None)
# # gpdc.generate_and_save_nulldists(sample_sizes=range(495, 501),
# #     null_dist_filename='dc_nulldists.npz')
# # gpdc.null_dist_filename ='dc_nulldists.npz'

# pcmci_gpdc = PCMCI(dataframe=new_df, cond_ind_test=gpdc,verbosity=0)
# gpdc_results = pcmci_gpdc.run_pcmci(tau_max= 2, pc_alpha= 0.1)
# # pcmci_gpdc.print_significant_links(p_matrix= gpdc_results['p_matrix'], val_matrix= gpdc_results['val_matrix'], alpha_level= 0.01)
# array, dymmy, dummy = gpdc._get_array(X=[(0, -1)], Y=[(2, 0)], Z=[(1, -2)], tau_max=2)
# x, meanx = gpdc._get_single_residuals(array, target_var=0, return_means=True)
# y, meany = gpdc._get_single_residuals(array, target_var=1, return_means=True)

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,3))
# axes[0].scatter(array[2], array[0], color='grey')
# axes[0].scatter(array[2], meanx, color='black')
# axes[0].set_title("GP of %s on %s" % (var_names[0], var_names[1]) )
# axes[0].set_xlabel(var_names[1]); axes[0].set_ylabel(var_names[0])
# axes[1].scatter(array[2], array[1], color='grey')
# axes[1].scatter(array[2], meany, color='black')
# axes[1].set_title("GP of %s on %s" % (var_names[2], var_names[1]) )
# axes[1].set_xlabel(var_names[1]); axes[1].set_ylabel(var_names[2])
# axes[2].scatter(x, y, color='red')
# axes[2].set_title("DC of residuals:" "\n val=%.3f / p-val=%.3f" % (gpdc.run_test(
#             X=[(0, -1)], Y=[(2, 0)], Z=[(1, -2)], tau_max=2)) )
# axes[2].set_xlabel("resid. "+var_names[0]); axes[2].set_ylabel("resid. "+var_names[2])
# plt.tight_layout()
# plt.show()

# Multiplicative noise

mp_data = np.random.randn(500, 3)
for t in range(1, 500):
    mp_data[t, 0] *= 0.2 * mp_data[t - 1, 1]
    mp_data[t, 2] *= 0.3 * mp_data[t - 2, 1]
mp_df = pp.DataFrame(mp_data, var_names=var_names)
# tp.plot_timeseries(mp_df)
# plt.show()
pcmci_mp_gdpc = PCMCI(dataframe=mp_df, cond_ind_test=gpdc)
mp_results = pcmci_mp_gdpc.run_pcmci(tau_max= 2, pc_alpha= 0.1)
# pcmci_mp_gdpc.print_significant_links(p_matrix=mp_results['p_matrix'], val_matrix=mp_results['val_matrix'], alpha_level= 0.01)

# array, dymmy, dummy = gpdc._get_array(X=[(0, -1)], Y=[(2, 0)], Z=[(1, -2)], tau_max=2)
# x, meanx = gpdc._get_single_residuals(array, target_var=0, return_means=True)
# y, meany = gpdc._get_single_residuals(array, target_var=1, return_means=True)

# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(8,3))
# axes[0].scatter(array[2], array[0], color='grey')
# axes[0].scatter(array[2], meanx, color='black')
# axes[0].set_title("GP of %s on %s" % (var_names[0], var_names[1]) )
# axes[0].set_xlabel(var_names[1]); axes[0].set_ylabel(var_names[0])
# axes[1].scatter(array[2], array[1], color='grey')
# axes[1].scatter(array[2], meany, color='black')
# axes[1].set_title("GP of %s on %s" % (var_names[2], var_names[1]) )
# axes[1].set_xlabel(var_names[1]); axes[1].set_ylabel(var_names[2])
# axes[2].scatter(x, y, color='red', alpha=0.3)
# axes[2].set_title("DC of residuals:" "\n val=%.3f / p-val=%.3f" % (gpdc.run_test(
#             X=[(0, -1)], Y=[(2, 0)], Z=[(1, -2)], tau_max=2)) )
# axes[2].set_xlabel("resid. "+var_names[0]); axes[2].set_ylabel("resid. "+var_names[2])
# plt.tight_layout()
# plt.show()

'''
The most general conditional independence test implemented in Tigramite is CMIknn based on conditional mutual information estimated with a k-nearest neighbor estimator. This test is described in the paper

Runge, Jakob. 2018. “Conditional Independence Testing Based on a Nearest-Neighbor Estimator of Conditional Mutual Information.” In Proceedings of the 21st International Conference on Artificial Intelligence and Statistics.

CMIknn involves no assumptions about the dependencies. The parameter knn determines the size of hypercubes, ie., the (data-adaptive) local length-scale. Now we cannot even pre-compute the null distribution because CMIknn is not residual-based like GPDC and the nulldistribution depends on many more factors. We, therefore, use significance='shuffle_test' to generate it in each individual test. The shuffle test for testing $I(X;Y|Z)=0$ shuffles $X$ values locally: Each sample point $i$’s $x$-value is mapped randomly to one of its nearest neigbors (shuffle_neighbors parameter) in subspace $Z$. Another free parameter is transform which specifies whether data is transformed before CMI estimation. The new default is transform=ranks which works better than the old transform=standardize. The following cell may take some minutes.
'''

# cmi_knn = CMIknn(significance='shuffle_test', knn= 0.1, shuffle_neighbors= 5, transform='ranks')
# pcmci_cmi_knn = PCMCI(dataframe=mp_df, cond_ind_test=cmi_knn, verbosity=2)
# cmi_results = pcmci_cmi_knn.run_pcmci(tau_max= 2, pc_alpha= 0.05)
# # pcmci_cmi_knn.print_significant_links(p_matrix=cmi_results['p_matrix'], val_matrix=cmi_results['val_matrix'], alpha_level= 0.01)

# cmi_knn_link_matrix = pcmci_cmi_knn.return_significant_parents(pq_matrix=cmi_results['p_matrix'], val_matrix=cmi_results['val_matrix'], alpha_level=0.01)['link_matrix']

# tp.plot_graph(val_matrix=cmi_results['val_matrix'], link_matrix=cmi_knn_link_matrix, var_names=var_names, 
#             link_colorbar_label='cross-MCI',
#             node_colorbar_label='auto-MCI',
#             vmin_edges=0.,
#             vmax_edges = 0.3,
#             edge_ticks=0.05,
#             cmap_edges='OrRd',
#             vmin_nodes=0,
#             vmax_nodes=.5,
#             node_ticks=.1,
#             cmap_nodes='OrRd',
#             )
# plt.show()