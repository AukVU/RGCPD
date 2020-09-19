import matplotlib.pyplot as plt
params = {'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large'}
plt.rcParams.update(params)
import seaborn as sns
sns.set()

import numpy as np
import pandas as pd
import os
import sys


user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'


def to_name(old_name):
    params = [None,None]
    color = 'black'
    if old_name == 'corr_map':
        name = 'Correlation'
        color = 'blue'
    elif old_name == 'pcmci':
        name = 'PCMCI directly on time series'
    elif old_name == 'parcorr_map':
        name = 'Partial correlation'
    elif old_name == 'N epsilon_corr_map':
        name = 'Correlation'
    elif old_name[:-3] == 'pcaaa_corr':
        name ='skip'
    elif old_name[:7] == 'parcorr':
        name = old_name.split(sep='-')
        if name[2] == 'True':
            target = 'target'
            if name[1] == '5':
                color = 'darkgreen'
            elif name[1] == '2':
                color = 'green'
            elif name[1] == '1':
                color = 'lightgreen'
        else:
            target = 'precur'
            if name[1] == '5':
                color = 'darkred'
            elif name[1] == '2':
                color = 'red'
            elif name[1] == '1':
                color = 'orange'
        params = [name[1], target]
        name = f"Partial correlation with lag {name[1]} on {target}"
        # print(params)
    else:
        name = old_name
    return name, params, color

def to_test(old_test):
    if old_test == 'sign':
        test = 'Signal'
    elif old_test == 'mode':
        test = 'Mode'
    elif old_test == 'time':
        test = 'Years'
    elif old_test == 'DBSC':
        test = 'DBSCAN distance_eps'
    elif old_test == 'spat':
        test = 'Spatial covariance'
    elif old_test == 'mday':
        test = '$x$ day mean'
    elif old_test == 'acor':
        test = 'Autocorrelation'
    else:
        test = old_test
    return test

def plot_scores(settings, path=None, target=False, precur=True, lags=[1,2,5], correlation=True):
    local_base_path = user_dir
    output = settings['filename']
    if path == None:
        path = local_base_path + f'/Code_Lennart/results/scores/{output}'
    plt.figure(figsize=(20,5))
    for subdir, dirs, files in os.walk(path, topdown=True):
        print(dirs)
        dirs[:] = [d for d in dirs if d not in ['plot']]
        correlations = [s for s in files if 'corr_map.csv' in s]
        parr_corr_targets = [s for s in files if 'True-False' in s]
        parr_corr_precurs = [s for s in files if 'False-True' in s]
        pcmcis = [s for s in files if 'pcmci' in s]
        files = correlations + parr_corr_targets + parr_corr_precurs + pcmcis

        
        for file in files:
            method, params, color = to_name(file[5:-4])
            if method == 'skip':
                continue
            if params[0] is not None:
                if int(params[0]) not in lags:
                    continue
                if (params[1] == 'target') and not target:
                    continue
                if (params[1] == 'precur') and not precur:
                    continue
            if (method == 'Correlation') and not correlation:
                continue


                
            test = file[:4]
            file = os.path.join(subdir, file)
            df = pd.read_csv(file)
            # print(df)
            # df = df.replace(0, np.NaN).T
            df = df.T
            r, c = len(df[0]), len(list(df))
            ndata = []
            for col in df.columns:
                ndata += list(df[col])
            r2 = len(ndata) - r
            t = r2*c
            dfnan = pd.DataFrame(np.reshape([np.nan]*t, (r2,c)), columns=list(df))
            df = df.append(dfnan)
            df['Score'] = ndata
            df['Mode'] = df.index
            repeats = int(r2/r)
            repeats = list(df['Mode'][:r]) * (repeats + 1)
            try:
                repeats = list(map(int, repeats))
            except ValueError:
                repeats = list(map(float, repeats))
            df['Mode'][:] = repeats
            sns.lineplot(x='Mode', y='Score', ci=95, data=df, label=method, color=color)
            axes = plt.gca()
    axes.set_ylim([-0.03,1.03])
    # axes.set_xlim([197,603])
    axes.set_xticks(repeats)
    axes.set_xlabel(to_test(test), fontsize=24)
    axes.set_ylabel('Score', fontsize=24)
            # plt.show()
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    leg = plt.legend(prop={'size': 24},loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=2)
    for legobj in leg.legendHandles:
        legobj.set_linewidth(3.0)
    for label in axes.get_xmajorticklabels():
        label.set_rotation(45)
        # label.set_horizontalalignment("right")
    
    filepath = path + '/plot'
    if os.path.isdir(filepath) != True : os.makedirs(filepath)
    filename = filepath  + f'/{to_test(test)}.pdf'
    if os.path.isfile(filename):
        os.remove(filename)
    plt.savefig(filename, format='pdf',bbox_inches='tight')
    plt.show()




settings = {}
settings['N'] = 5
settings['nx'], settings['ny'], settings['T'] = 30, settings['N'] * 30, 5114
settings['spatial_covariance'] = 0.3
settings['random_modes'] = False
settings['noise_use_mean'] = True
settings['noise_level'] = 0
settings['transient'] = 200
settings['spatial_factor'] = 0.1

settings['user_dir'] = user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'
settings['extra_dir'] = 'Code_Lennart'
settings['filename'] = 'multiple_test'


settings['random_causal_map'] = True
settings['area_size'] = None


## If any of the following settings is set to True, the results folder with {filename} will be removed!
## Also when 'plot_points' is not None
settings['netcdf4'] = True
settings['save_time_series'] = True
settings['do_pcmci'] = True
settings['save_matrices'] = True
settings['plot_points'] = 500
links_coeffs = 'model3'

settings['alpha'] = 0.01
settings['measure'] = 'average'
settings['val_measure'] = 'average'

# test = 'NEW_MODEL/NOISE_S0.05'
# test = 'AAA_NO_LAG_AUTOCOR/MODES'

# test = 'REAL_MODEL/target_ac=0.6/NOISE_CUSTOM'
test = 'REAL_MODEL/random/SIGNAL'

user_dir = settings['user_dir']
path = user_dir + f'/Results_Lennart/scores/' + test
# path = user_dir + f'/Code_Lennart/results/scores/multiple_test/test'
plot_scores(settings, path=path, lags=[1,2,5], target=True, precur=True, correlation=True)