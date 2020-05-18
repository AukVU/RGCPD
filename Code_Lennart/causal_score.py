import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import sys

def calculate_p_value_score(found, real, measure='', alpha=0.001):
    measure = measure.capitalize()
    real_matrix = np.load(real)
    found_matrix = np.load(found)

    found_matrix = found_matrix < alpha
    real_matrix = real_matrix < alpha

    accuracy = 0.0
    if measure == 'Average':
        accuracy = (found_matrix == real_matrix).mean()
    return accuracy, real_matrix

def calculate_val_score(found, real, mask, measure=''):
    measure = measure.capitalize()
    real_matrix = np.load(real)
    found_matrix = np.load(found)

    found_masked = ma.masked_array(found_matrix, mask=np.logical_not(mask))
    max_found = np.max(found_masked)
    real_masked = ma.masked_array(real_matrix, mask=np.logical_not(mask))
    max_real = np.max(real_masked)
    found_masked = found_masked / max_found * max_real

    accuracy = 0.0
    if measure == 'Average':
        # print(found_masked)
        accuracy = (found_masked == real_masked).mean()
    return accuracy

def calculate_causal_score(settings):
    general_path = settings['user_dir'] + '/' + settings['extra_dir'] + '/results/' + settings['filename']
    general_path = general_path + '/matrices'
    all_files=[]
    i = 0
    for subdir, dirs, files in os.walk(general_path):
        if i == 0:
            tests = dirs
        files_paths = [os.path.join(subdir, file) for file in files]
        all_files.append(files_paths)
        i += 1
    all_files = all_files[1:]
    tests = [test.replace('AAA_','') for test in tests]
    number_of_modes = int(len(all_files[0]) / 2)
    
    real = all_files[0]

    results = {}
    for test in tests:
        results[f'{test} p_value'] = []
        results[f'{test} value'] = []
    for mode in range(number_of_modes):
        for i, test in enumerate(tests):
            p_score, real_mask = calculate_p_value_score(all_files[i][mode], real[mode], measure=settings['measure'], alpha=settings['alpha'])
            results[f'{test} p_value'].append(p_score)
            val_score = calculate_val_score(all_files[i][number_of_modes + mode], real[number_of_modes + mode], real_mask,
                                            measure=settings['val_measure'])
            results[f'{test} value'].append(val_score)
        
    
    results = pd.DataFrame(data=results)
    print(results)












settings = {}
settings['user_dir'] = user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'
settings['extra_dir'] = 'Code_Lennart'
settings['filename'] = 'very_small'

settings['alpha'] = 0.01
settings['measure'] = 'average'
settings['val_measure'] = 'average'

calculate_causal_score(settings)