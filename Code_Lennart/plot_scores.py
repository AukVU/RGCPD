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
    if old_name == 'corr_map':
        name = 'Correlation'
    elif old_name == 'pcmci':
        name = 'PCMCI directly on time series'
    elif old_name == 'parcorr_map':
        name = 'Partial correlation'
    elif old_name == 'N epsilon_corr_map':
        name = 'Correlation'
    else:
        name = old_name
    return name

def to_test(old_test):
    if old_test == 'sign':
        test = 'Signal'
    elif old_test == 'mode':
        test = 'Mode'
    elif old_test == 'time':
        test = 'Years'
    elif old_test == 'DBSC':
        test = 'DBSCAN distance_eps'
    else:
        test = old_test
    return test

def plot_scores(settings, path=None):
    local_base_path = user_dir
    output = settings['filename']
    if path == None:
        path = local_base_path + f'/Code_Lennart/results/scores/{output}'
    plt.figure(figsize=(20,5))
    for subdir, dirs, files in os.walk(path, topdown=True):
        print(dirs)
        dirs[:] = [d for d in dirs if d not in ['plot']]
        
        for file in files:
            method = to_name(file[5:-4])
            test = file[:4]
            file = os.path.join(subdir, file)
            df = pd.read_csv(file)
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
            sns.lineplot(x='Mode', y='Score', ci=95, data=df, label=method)
            axes = plt.gca()
    axes.set_ylim([0.2,1.03])
    # axes.set_xlim([197,603])
    axes.set_xticks(repeats)
    axes.set_xlabel(to_test(test), fontsize=24)
    axes.set_ylabel('Score', fontsize=24)
            # plt.show()
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)
    plt.legend(prop={'size': 24},loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=2)
    
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

test = 'TIME'

user_dir = settings['user_dir']
path = user_dir + f'/Results_Lennart/scores/' + test
plot_scores(settings, path=path)