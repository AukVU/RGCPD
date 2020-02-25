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
# Include forcing variable but not interested in driver for forcing variable
pcmci_parcorr = PCMCI(
    selected_variables = [0,1,2],
    dataframe=s_df, 
    cond_ind_test=par_corr,
    verbosity=0)
results = pcmci_parcorr.run_pcmci(tau_max=2, pc_alpha=0.2)
pcmci_parcorr.print_significant_links(
        p_matrix = results['p_matrix'], 
        val_matrix = results['val_matrix'],
        alpha_level = 0.01)
# link_matrix = pcmci_parcorr.return_significant_parents(pq_matrix=results['p_matrix'],
#                         val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']
# # Plot time series graph
# tp.plot_time_series_graph(
#     figsize=(6, 3),
#     val_matrix=results['val_matrix'],
#     link_matrix=link_matrix,
#     var_names=var_names,
#     link_colorbar_label='MCI',
#     )
# plt.show()

print('\n-------------------------Time sub-sampling----------------------\n')
# Time sub-sampling
t_data = np.random.randn(1000, 3)
for t  in range(1, 1000):
    t_data[t, 0] += 0.*t_data[t-1, 0] + 0.6*t_data[t-1,2]
    t_data[t, 1] += 0.*t_data[t-1, 1] + 0.6*t_data[t-1,0]
    t_data[t, 2] += 0.*t_data[t-1, 2] + 0.6*t_data[t-1,1]
t_df = pp.DataFrame(t_data, var_names=var_names)
# tp.plot_timeseries(t_df)
# plt.show()

pcmci_parcorr_t = PCMCI(dataframe=t_df, cond_ind_test=ParCorr())
results = pcmci_parcorr_t.run_pcmci(tau_min=0, tau_max=2, pc_alpha=0.2)
pcmci_parcorr_t.print_significant_links(p_matrix=results['p_matrix'], val_matrix=results['val_matrix'], alpha_level=0.01)

link_matrix_t = pcmci_parcorr_t.return_significant_parents(pq_matrix=results['p_matrix'], val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']

# Plot 

# tp.plot_time_series_graph(figsize=(6, 3), val_matrix=results['val_matrix'], link_matrix=link_matrix_t, var_names=var_names, link_colorbar_label='MCI')

# Sub-sample data causes loop detected in the wrong direction(s)

sampled_data = t_data[::2]
pcmci_parcorr = PCMCI(dataframe=pp.DataFrame(sampled_data, var_names=var_names), 
                      cond_ind_test=ParCorr(), verbosity=0)
results = pcmci_parcorr.run_pcmci(tau_min=0, tau_max=2, pc_alpha=0.2)
link_matrix = pcmci_parcorr.return_significant_parents(pq_matrix=results['p_matrix'],
                        val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']
# Plot time series graph
# tp.plot_time_series_graph(
#     figsize=(6, 3),
#     val_matrix=results['val_matrix'],
#     link_matrix=link_matrix,
#     var_names=var_names,
#     link_colorbar_label='MCI'
#     )
# plt.show()
print('\n-----------------Causal Markov Condition-------------------------\n')
# Causal Markov condition

T= 10000
# Generate 1/f noise by averaging AR1-process with wide range of coeffs 
# (http://www.scholarpedia.org/article/1/f_noise)
def one_over_f_noise(T, n_ar=20):
    whitenoise = np.random.randn(T, n_ar)
    ar_coeffs = np.linspace(0.1, 0.9, n_ar)
    for t in range(T):
        whitenoise[t] += ar_coeffs*whitenoise[t-1] 
    return whitenoise.sum(axis=1)      
m_data = np.random.randn(T, 3)
m_data[:, 0] += one_over_f_noise(T)
m_data[:, 1] += one_over_f_noise(T)
m_data[:, 2] += one_over_f_noise(T)

for t in range(1, T):
    m_data[t, 0] += .4 * m_data[t - 1, 1]
    m_data[t, 2] += .3 * m_data[t - 2, 1]
m_df = pp.DataFrame(m_data, var_names=var_names)
# tp.plot_timeseries(m_df)
# plt.show()
# Due to the violation of Markov condition spurious links , especially auto-dependencies will be detected, since process has long memory and the present state is not independent of further past given some set of parents. 
pcmci_parcorr_m = PCMCI(dataframe=m_df, cond_ind_test=ParCorr())
results = pcmci_parcorr_m.run_pcmci(tau_max=5, pc_alpha=0.2)
pcmci_parcorr_m.print_significant_links(p_matrix=results['p_matrix'], val_matrix=results['val_matrix'],alpha_level=0.01)

print('\n-----------------------------Time aggregation------------------------\n')
# Time aggregation 
# An important choice is how to aggregate measured time series. For example, climate time series might have been measured daily, but one might be interested in a less noisy time-scale and analyze monthly aggregates.
a_data = np.random.randn(1000, 3)
for t in range(1, 1000):
    a_data[t, 0] += .7*a_data[t -1, 0]
    a_data[t, 1] += .6*a_data[t -1, 1] + .6* a_data[t -1, 0]
    a_data[t, 2] += .5*a_data[t -1, 2] + .6* a_data[t -1, 1]
a_df = pp.DataFrame(a_data, var_names=var_names)
tp.plot_timeseries(a_df)
plt.show()

pcmci_parcorr_ag = PCMCI(dataframe=a_df, cond_ind_test=ParCorr())
results = pcmci_parcorr_ag.run_pcmci(tau_min=0, tau_max=2, pc_alpha=.2)
pcmci_parcorr_ag.print_significant_links(p_matrix=results['p_matrix'], val_matrix=results['val_matrix'], alpha_level=0.01)

link_matrix_ag = pcmci_parcorr_ag.return_significant_parents(pq_matrix=results['p_matrix'], val_matrix=results['val_matrix'], alpha_level=0.01)['link_matrix']

# Plot
tp.plot_time_series_graph(figsize=(6, 3), val_matrix=results['val_matrix'], link_matrix=link_matrix_ag, var_names=var_names, link_colorbar_label='MCI')

print('\n------------------------END---------------------------------\n')
