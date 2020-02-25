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

#Faithfulness

np.random.seed(42)
data = np.random.randn(500, 3)
for  t in range(1, 500):
    data[t, 1] += 0.6 * data[t - 1, 0]
    data[t, 2] += 0.6 * data[t - 1, 1] - 0.36 * data[t - 2, 0]
var_names = [r'$x^0$', r'$x^1$', r'$x^2$', r'$x^3$']
df = pp.DataFrame(data, var_names=var_names)
# tp.plot_timeseries(df)
# plt.show()

par_corr = ParCorr()
pcmci_parcorr = PCMCI(dataframe=df, cond_ind_test=par_corr, verbosity=1)
all_parents = pcmci_parcorr.run_pc_stable(tau_max=2, pc_alpha=0.2)
print('\n-----------Revealing true underlying graph relations------------------------\n')

results = pcmci_parcorr.run_pcmci(tau_max=2, pc_alpha=2)
pcmci_parcorr.print_significant_links(p_matrix=results['p_matrix'], val_matrix=results['val_matrix'], alpha_level=0.01)

# Deterministic dependencies
print('\n---------------------Deterministic dependencies----------------------------\n')

dep_data = np.random.randn(500, 3)
for t in range(1, 500):
    dep_data[t, 0] += .4* dep_data[t - 1, 1]
    dep_data[t, 2] += .3* dep_data[t - 2, 1] + .7* dep_data[t - 1, 0]
df_dep = pp.DataFrame(dep_data, var_names=var_names)
# tp.plot_timeseries(df_dep)
# plt.show()

det_parcorr = ParCorr()
pcmci_parcorr_det = PCMCI(dataframe=df_dep, cond_ind_test=det_parcorr, verbosity=2)
det_results = pcmci_parcorr_det.run_pcmci(tau_max=2, pc_alpha=0.2)
pcmci_parcorr_det.print_significant_links(p_matrix=det_results['p_matrix'], val_matrix=det_results['val_matrix'], alpha_level=0.01)
link_matrix_det = pcmci_parcorr_det.return_significant_parents(pq_matrix=det_results['p_matrix'], val_matrix=det_results['val_matrix'],alpha_level=0.01)['link_matrix']

# tp.plot_time_series_graph(figsize=(6, 3), val_matrix=det_results['val_matrix'], link_matrix=link_matrix_det, var_names=var_names, link_colorbar_label='MCI',)
# plt.show()

print('\n---------------- Start Causal sufficiency----------------------\n')
#Causal Sufficiency

c_data = np.random.randn(10000, 5)
a = .8
var_names_ = [r'$x^0$', r'$x^1$', r'$x^2$', r'$x^3$', r'$x^4$']
for t in range(5, 10000):
    c_data[t, 0] += a* c_data[t - 1, 0]
    c_data[t, 1] += a* c_data[t - 1, 1] + .5 *c_data[t  - 1,  0]
    c_data[t, 2] += a* c_data[t - 1, 2] + .5 *c_data[t - 1, 1] + .5* c_data[t - 1, 4]
    c_data[t, 3] += a* c_data[t - 1 , 3] + .5 *c_data[t - 2, 4]
    c_data[t, 4] += a* c_data[t - 1, 4]
c_df = pp.DataFrame(c_data, var_names=var_names_)
# tp.plot_timeseries(c_df)
# plt.show()
obs_data = c_data[:, [0, 1, 2, 3]]
var_names_lat = ['W', 'Y', 'X', 'Z', 'U']

# Visualize causal data and observational data to see collider
# for data_here in [c_data, obs_data]:
#     dataframe = pp.DataFrame(data_here)
#     parcorr = ParCorr()
#     pcmci_parcorr = PCMCI(
#         dataframe=dataframe, 
#         cond_ind_test=parcorr,
#         verbosity=0)
#     results = pcmci_parcorr.run_pcmci(tau_max=5, pc_alpha=0.1)
#     pcmci_parcorr.print_significant_links(
#             p_matrix = results['p_matrix'], 
#             val_matrix = results['val_matrix'],
#             alpha_level = 0.01)
#     link_matrix = pcmci_parcorr.return_significant_parents(pq_matrix=results['p_matrix'],
#                             val_matrix=results['val_matrix'], alpha_level=0.001)['link_matrix']
#     tp.plot_graph(
#         val_matrix=results['val_matrix'],
#         link_matrix=link_matrix,
#         var_names=var_names_lat,
#         link_colorbar_label='cross-MCI',
#         node_colorbar_label='auto-MCI',
#         )
# plt.show()
print('\n------------------Solar forcing-------------------------\n')
# Solar forcing

s_data = np.random.randn(1000, 4)
#Simple sun
s_data[:, 3] = np.sin(np.arange(1000)*20 / np.pi)
var_names[3] = 'Sun'
c = .8
# More realistic data by adding autodependencies on past values
for t in range(1, 1000):
    s_data[t, 0] += .4 * s_data[t - 1, 0] + .4 * s_data[t - 1, 1] + c * s_data[t - 1, 3]
    s_data[t, 1] += .5 * s_data[t - 1, 1] + c * s_data[t - 1, 3] 
    s_data[t, 2] += .6 * s_data[t - 1, 2] + .3 * s_data[t - 2, 1] + c * s_data[t - 1, 3]
s_df  = pp.DataFrame(s_data, var_names=var_names)
# tp.plot_timeseries(s_df)
# plt.show()

# Not accounting for solar forcing, spurious links may emerge
dataframe_nosun = pp.DataFrame(s_data[:,[0,1,2]], var_names=var_names)
pcmci_parcorr = PCMCI(
    selected_variables = [0,1,2],
    dataframe=dataframe_nosun, 
    cond_ind_test=par_corr,
    verbosity=0)
results = pcmci_parcorr.run_pcmci(tau_max=2, pc_alpha=0.2)
pcmci_parcorr.print_significant_links(
        p_matrix = results['p_matrix'], 
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)