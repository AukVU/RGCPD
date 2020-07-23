import numpy as np
import numpy.ma as ma
import pandas as pd
import os
import sys

def calculate_p_value_score(found, real, correlated, measure='', alpha=0.001, target_only=True):
    measure = measure.capitalize()
    real_matrix_orig = np.load(real)
    found_matrix_orig = np.load(found)

    found_matrix_orig = found_matrix_orig < alpha
    real_matrix_orig = real_matrix_orig < alpha
    real_matrix = real_matrix_orig[correlated]
    found_matrix = found_matrix_orig
    if target_only:
        real_matrix = real_matrix_orig[0]
        found_matrix = found_matrix_orig[0]

    accuracy = 0.0
    if measure == 'Average':
        accuracy = (found_matrix == real_matrix).mean()
    return accuracy, real_matrix

def calculate_val_score(found, real, mask, correlated, measure='', target_only=True):
    measure = measure.capitalize()
    real_matrix_orig = np.load(real)
    found_matrix_orig = np.load(found)
    real_matrix = real_matrix_orig[correlated]
    found_matrix = found_matrix_orig
    if target_only:
        real_matrix = real_matrix_orig[0]
        found_matrix = found_matrix_orig[0]

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

def calculate_causal_score(settings, val=False, verbose=False, locs=None, target_only=True):
    general_path = settings['user_dir'] + '/' + settings['extra_dir'] + '/results/' + settings['filename']
    general_path = general_path + '/matrices'

    if locs == None:
        locs = list(range(settings['N']))

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
    
    
    real = all_files[0]
    real_links = np.load(general_path + '/AAA_real/ZZZ_real_links.npy')

    results = {}
    for test in tests:
        results[f'{test} p_value'] = []
        # results[f'{test} value'] = []
    for i, test in enumerate(tests):
        number_of_modes = int((len(all_files[i]) - 2) / 2)
        test2 = test
        if test == 'real':
            test2 = 'AAA_real'
        correlated = np.load(general_path + '/' + test2 + '/ZZZ_correlated.npy')
        found_links = np.ones(len(real_links))
        found_links[correlated] = 0
        scores = np.multiply(real_links, found_links)
        for score in scores:
            results[f'{test} p_value'].append(score)
            # results[f'{test} value'].append(score)
        for j, mode in enumerate(correlated):
            p_score, real_mask = calculate_p_value_score(all_files[i][j], real[j], correlated, measure=settings['measure'], alpha=settings['alpha'], target_only=target_only)
            results[f'{test} p_value'][mode] = p_score
            # val_score = calculate_val_score(all_files[i][number_of_modes + j], real[number_of_modes + j], real_mask, correlated, measure=settings['val_measure'], target_only=target_only)
            # results[f'{test} value'][mode] = val_score
        
    
    results = pd.DataFrame(data=results)
    results.loc['mean'] = results.mean()
    if not val:
        results = results.filter(like='p_value')
    if verbose:
        print(results)
    
    return results.tail(1)












# settings = {}
# settings['user_dir'] = user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'
# settings['extra_dir'] = 'Code_Lennart'
# settings['filename'] = 'test_met_savar7'
# settings['N'] = 5

# settings['alpha'] = 0.01
# settings['measure'] = 'average'
# settings['val_measure'] = 'average'

# score = calculate_causal_score(settings, val=False, verbose=True, target_only=True)
# print(score)







# path = settings['user_dir'] + '/' + settings['extra_dir'] + '/results/'\
#             + settings['filename'] + '/scores'# + key.split(' ', 1)[0]

# pcmci_df = pd.DataFrame(columns=['0'])
# corr_df = pd.DataFrame(columns=['0'])
# parcorr_df = pd.DataFrame(columns=['0'])

# if os.path.isdir(path) != True : os.makedirs(path)
# for i, key in enumerate(score.columns):
#     key = key.split(' ', 1)[0]
#     if key == 'pcmci_test':
#         pcmci_df = pcmci_df.append({'0': score.values[0][i]}, ignore_index=True)
#     if key == 'parcorr_map':
#         parcorr_df = parcorr_df.append({'0': score.values[0][i]}, ignore_index=True)
#     elif key != 'real':
#         corr_df = corr_df.append({'0': score.values[0][i]}, ignore_index=True)
# pcmci_df.to_csv(path + '/pcmci.csv', index=False)
# corr_df.to_csv(path + '/corr.csv', index=False)
# parcorr_df.to_csv(path + '/parcorr.csv', index=False)


    
