import numpy as np
import matplotlib
from matplotlib import pyplot as plt
# %matplotlib inline     
## use `%matplotlib notebook` for interactive figures
plt.style.use('ggplot')
import sklearn

import tigramite
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests import ParCorr, GPDC, CMIknn, CMIsymb
from tigramite.models import LinearMediation, Prediction

np.random.seed(42)     # Fix random seed
links_coeffs = {0: [((0, -1), 0.9), ((1, -1), -0.25)],
                1: [((1, -1), 0.95), ((3, -1), 0.3)],
                2: [((2, -1), 0.85), ((1, -2), 0.3), ((3, -3), 0.3)],
                3: [((3, -1), 0.8)],
                }
T = 100     # time series length
N = len(links_coeffs)
tau_max = 5
realizations = 100
alpha_level = 0.05

var_names = [r'$Z$', r'$X$', r'$Y$', r'$W$']
# # Define whole past
# whole_past = {}
# for j in range(N):
#     whole_past[j] = [(var, -lag)
#                          for var in range(N)
#                          for lag in range(1, tau_max + 1)
#                     ]
def get_sig_links():
    p_matrices = {'PCMCI':np.ones((realizations, N, N, tau_max+1)),
                  'FullCI':np.ones((realizations, N, N, tau_max+1))}
    val_matrices = {'PCMCI':np.zeros((realizations, N, N, tau_max+1)),
                  'FullCI':np.zeros((realizations, N, N, tau_max+1))}  
    for i in range(realizations):
        data, true_parents_neighbors = pp.var_process(links_coeffs, T=T)
        dataframe = pp.DataFrame(data)
        
        # PCMCI
        pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr())
        results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=0.2)
        p_matrices['PCMCI'][i] = results['p_matrix']
        val_matrices['PCMCI'][i] = results['val_matrix']

        # Condition on whole past
        results = pcmci.run_fullci(tau_max=tau_max)
        p_matrices['FullCI'][i] = results['p_matrix']
        val_matrices['FullCI'][i] = results['val_matrix']

    # Get true positive rate (=power) and false positive rate 
    sig_links = {'PCMCI':(p_matrices['PCMCI'] <= alpha_level).mean(axis=0),
                  'FullCI':(p_matrices['FullCI'] <= alpha_level).mean(axis=0),}
    ave_val_matrices = {'PCMCI':val_matrices['PCMCI'].mean(axis=0),
                  'FullCI':val_matrices['FullCI'].mean(axis=0),}
    return sig_links, ave_val_matrices

sig_links, ave_val_matrices = get_sig_links()
# We estimate how often a link was detected at the given alpha_level and plot this fraction as the width of the links while the average effect size for each link is given as the color:
# Showing detection power as width of links
# min_sig = 0.2
# vminmax = 0.4
# link_matrix = (sig_links['PCMCI'] > min_sig)
# tp.plot_graph(val_matrix=ave_val_matrices['PCMCI'],
#               link_matrix=link_matrix, var_names=var_names,
#               link_width=sig_links['PCMCI'],
#              arrow_linewidth=70.,
#               vmin_edges=-vminmax,
#               vmax_edges=vminmax,

# )
# link_matrix = (sig_links['FullCI'] > min_sig)
# tp.plot_graph(val_matrix=ave_val_matrices['FullCI'],
#               link_matrix=link_matrix, var_names=var_names,
#               link_width=sig_links['FullCI'], 
#               link_colorbar_label='FullCI',
#               node_colorbar_label='auto-FullCI',
#              arrow_linewidth=70.,
#               vmin_edges=-vminmax,
#               vmax_edges=vminmax,
# )
# plt.show()