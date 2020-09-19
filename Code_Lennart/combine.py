import pandas as pd
import numpy as np
import os
import sys

user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'

test = 'REAL_MODEL/target_ac=0.6/YEARS'

path1 = user_dir + f'/Results_Lennart/scores/' + test
path2 = user_dir + f'/Results_Lennart/scores/' + test + '2'
pathF = user_dir + f'/Results_Lennart/scores/' + test + '_FULL'

print('\n', path1, '\n', path2, '\n', pathF)

for subdir, dirs, files in os.walk(path1, topdown=True):
    dirs[:] = [d for d in dirs if d not in ['plot']]
    files1 = files

for subdir, dirs, files in os.walk(path2, topdown=True):
    dirs[:] = [d for d in dirs if d not in ['plot']]
    files2 = files

for i, file in enumerate(files1):
    splitted = file.split(sep='_')
    if splitted[1] == 'pcaaa':
        continue
    # print(f"Read from file:\n{path1 + '/' + file}\n")
    df1 = pd.read_csv(path1 + '/' + file)
    # print(f"Read from file:\n{path2 + '/' + files2[i]}\n")
    df2 = pd.read_csv(path2 + '/' + files2[i])


    # print(df1.columns, df2.columns)
    to_concat = [df1, df2]
    concat = pd.concat(to_concat, axis=1)
    print(concat.columns)
    concat.to_csv(pathF + '/' + file, index=False)
