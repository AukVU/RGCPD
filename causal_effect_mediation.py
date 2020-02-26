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

np.random.seed(42)
# links_coeffs = {0: [],
#                 1: [((0, -1), 0.5)],
#                 2: [((1, -1), 0.5)],
#                 }
links_coeffs = {0: [((0,-1), 0.8)],
                1: [((1,-1), 0.8), ((0, -1), 0.5)],
                2: [((2,-1), 0.8), ((1, -1), 0.5)],
                }
var_names = [r"$X$", r"$Y$", r"$Z$"]
    
data, true_parents = pp.var_process(links_coeffs, T=1000)

# Initialize dataframe object, specify time axis and variable names
dataframe = pp.DataFrame(data, 
                         var_names=var_names)
med = LinearMediation(dataframe=dataframe)
med.fit_model(all_parents=true_parents, tau_max=4)

print ("Link coefficient (0, -2) --> 2: ", med.get_coeff(i=0, tau=-2, j=2))

print ("Causal effect (0, -2) --> 2: ", med.get_ce(i=0, tau=-2, j=2))

print ("Mediated Causal effect (0, -2) --> 2 through 1: ", med.get_mce(i=0, tau=-2, j=2, k=1))

# Plot
# i=0; tau=4; j=2
# graph_data = med.get_mediation_graph_data(i=i, tau=tau, j=j)
# tp.plot_mediation_time_series_graph(
#     var_names=var_names,
#     path_node_array=graph_data['path_node_array'],
#     tsg_path_val_matrix=graph_data['tsg_path_val_matrix']
#     )
# tp.plot_mediation_graph(
#                     var_names=var_names,
#                     path_val_matrix=graph_data['path_val_matrix'], 
#                     path_node_array=graph_data['path_node_array'],
#                     )
# plt.show()
print ("Average Causal Effect X=%.2f, Y=%.2f, Z=%.2f " % tuple(med.get_all_ace()))
print ("Average Causal Susceptibility X=%.2f, Y=%.2f, Z=%.2f " % tuple(med.get_all_acs()))
print ("Average Mediated Causal Effect X=%.2f, Y=%.2f, Z=%.2f " % tuple(med.get_all_amce()))
print("Self-loops are excluded per default")