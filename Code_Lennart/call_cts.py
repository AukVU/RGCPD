# Setting up the settings and running above code
import os, inspect, sys

curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
main_dir = '/'.join(curr_dir.split('/')[:-1])
RGCPD_func = os.path.join(main_dir, 'RGCPD')
cluster_func = os.path.join(main_dir, 'clustering/') 
df_ana_func = os.path.join(main_dir, 'df_analysis/df_analysis') 
if cluster_func not in sys.path:
    sys.path.append(main_dir)
    sys.path.append(RGCPD_func)
    sys.path.append(cluster_func)
    sys.path.append(df_ana_func)


import creating_time_series as cts

settings = {}


settings['random_modes'] = False
settings['noise_use_mean'] = False

settings['transient'] = 200
settings['spatial_factor'] = 0.05 #0.1

if len(sys.argv) > 1:
    settings['user_dir'] = sys.argv[1]
else:
    settings['user_dir'] = user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'
settings['extra_dir'] = 'Code_Lennart'
settings['filename'] = 'Xavier'

# sys.exit()
settings['random_causal_map'] = True
settings['area_size'] = None


## If any of the following settings is set to True, the results folder with {filename} will be removed!
## Also when 'plot_points' is not None
settings['netcdf4'] = True
settings['save_time_series'] = True
settings['do_pcmci'] = True
settings['save_matrices'] = True


settings['N'] = 5
settings['nx'], settings['ny'], settings['T'] = 30, settings['N'] * 30, 731 #3287 #1826 #5114 731
settings['plot_points'] = settings['T']

settings['signal'] = 0.15#0.5
settings['noise_level'] = 10
settings['custom_noise_level'] = 0
settings['spatial_covariance'] = 500


links_coeffs = 'Jier'

settings['model'] = 'some'
settings['autocor'] = 0.96
settings['autocor_target'] = 0.6
settings['timefreq'] = 1

print('Start generating')
cts.create_time_series(settings, links_coeffs,  verbose=True,
                                            plot_modes=False,
                                            plot_timeseries=True,
                                            draw_network=True,
                                            custom_noise=False,
                                            cluster=False)