import sys, subprocess, os  

script = os.path.join(os.path.abspath('.'), 'experiment_sst.py')

# experiments = ['experiment_sst.py -it 100 -v nu', 'experiment_sst.py -it 1000 -v nu']
experiments = [str(script)+' -it 100 -v nu',str(script) +' -it 1000 -v nu']
for i, exp in enumerate(experiments):
    print(f'[INFO] Start run on experiment {str(i)} \n')
    print(exp)
    subprocess.Popen('python '+exp, shell=True).wait()
print(f'\n[INFO] Done running experiments\n')