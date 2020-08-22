# Setting up the settings and running above code
import sys


import creating_time_series as cts

settings = {}


settings['random_modes'] = False
settings['noise_use_mean'] = False

settings['transient'] = 200
settings['spatial_factor'] = 0.1

if len(sys.argv) > 1:
    settings['user_dir'] = sys.argv[1]
else:
    settings['user_dir'] = user_dir = '/mnt/c/Users/lenna/Documents/Studie/2019-2020/Scriptie/RGCPD'
settings['extra_dir'] = 'Code_Lennart'
settings['filename'] = 'test_met_savar11'

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
settings['nx'], settings['ny'], settings['T'] = 30, settings['N'] * 30,  3287 #1826 #5114 731
settings['plot_points'] = settings['T']

settings['signal'] = 0#0.5
settings['noise_level'] = 5
settings['spatial_covariance'] = 2000


links_coeffs = 'model3'

settings['model'] = 'one'

print('Start generating')
cts.create_time_series(settings, links_coeffs,  verbose=True,
                                            plot_modes=False,
                                            plot_timeseries=True,
                                            draw_network=True,
                                            cluster=False)