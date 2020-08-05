# common
import sys
import pandas as pd
import numpy as np
import xarray as xr
import os
import os.path as op
import datetime

import warnings
warnings.filterwarnings("ignore")

# custom
import spectra_functions as specfun

p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))

# -------------- EDIT THIS PART --------------------------------------------- #
name = 'SAF' # used name in the SWAN section
# --------------------------------------------------------------------------- #

print('Extracting CSIRO information...')

partitions = pd.read_pickle(op.join(p_data, 'reconstructed',
                                    'reconstructed_partitioned_'+name+'.pkl'))[::50]
print(partitions.info())

## -------------- RUN THIS PART FOR DAILY HOURS ----------------------------- #
#dates = np.arange(datetime.datetime(1979,1,1), datetime.datetime(2020,2,29), 
#                  datetime.timedelta(hours=1))
#daylight_dates = []
#for date in dates:
#    if int(str(date)[5:7])<4 or int(str(date)[5:7])>9:
#        if int(str(date)[11:13])>8 and int(str(date)[11:13])<18:
#            daylight_dates.append(date)
#    else:
#        if int(str(date)[11:13])>7 and int(str(date)[11:13])<20:
#            daylight_dates.append(date)
#partitions = partitions.loc[daylight_dates][::3]
## -------------------------------------------------------------------------- #

# If example data wants to be tested, please use this dataframe below!!!

# ---------------------- TEST WAVE CONDITIONS ------------------------------- #
test = pd.DataFrame({'Hsea':      [1.2,    1.5,    1.7,    2.6],
                     'Tpsea':     [8.1,    8.8,    18,     18],
                     'Dirsea':    [70.3,   90,    345,    70],
                     'Sprsea':    [25.5,   30,     15,     30],
                     'Hswell1':   [2.0,    2.5,    np.nan, np.nan],
                     'Tpswell1':  [15.82,  18,     np.nan, np.nan],
                     'Dirswell1': [350.6,  355,    np.nan, np.nan],
                     'Sprswell1': [10.1,   10,     np.nan, np.nan],
                     'Hswell2':   [0.8,    0.8,    np.nan, np.nan],
                     'Tpswell2':  [19.52,  18,     np.nan, np.nan],
                     'Dirswell2': [240.3,  180,    np.nan, np.nan],
                     'Sprswell2': [6.56,   12,     np.nan, np.nan],
                     'Hswell3':   [np.nan, np.nan, np.nan, np.nan],
                     'Tpswell3':  [np.nan, np.nan, np.nan, np.nan],
                     'Dirswell3': [np.nan, np.nan, np.nan, np.nan],
                     'Sprswell3': [np.nan, np.nan, np.nan, np.nan],
                     'Tm_02':     [12,     13,     16,     14]})
gamma_values = [[3, 3, 3, 3],
                [10, 10, 10, 10],
                [10, 10, 10, 10],
                [10, 10, 10, 10]]
print(test.info())
# --------------------------------------------------------------------------- #

print('Creating spectrums and elevations...')
print('--------------------------------------------')

# A maximum of 400 data can be spectrally analyzed at the same time
timesteps = 10
timestep  = int(len(partitions)/timesteps)
for ts in range(timesteps):
    print('Group number {}...'.format(ts+1))
    print('--------------------------------------------')
    ts_spec = specfun.spectra(partitions.iloc[ts*timestep:(ts+1)*timestep],
                              gamma_values=False)
    ts_spec = ts_spec.sum(dim='partition')
    if ts==0:
        specs = ts_spec
    else:
        specs = xr.concat([specs, ts_spec], dim='time')
    print('{} joined...'.format((ts+1)*timestep))
    print('--------------------------------------------')

# Plot the obtained spectra now
specs_plotted = specfun.plot_spectrum(specs, time_plots=[10,150,160,34])

sys.exit()

# Select 1 so just one image will be plotted (recommended)
time_waves = 5

# A GIF is automatically generated for the more than 1 case
gif_path = op.join(p_data, '..', 'images', 'elevations')

for surf in range(len(specs_plotted)):
    print('--------------------------------------------')
    print('{} surface plotted...'.format(surf))
    print('--------------------------------------------')
    spec_to_elev = specs.sel(time=specs_plotted[surf])
    for tw in range(time_waves):
        elev = specfun.surface(spec_to_elev, t=tw)
        if tw==0:
            elevs = elev
        else:
            elevs = xr.concat([elevs, elev], dim='time')
    new_path = op.join(gif_path, str(surf))
    os.mkdir(new_path)
    specfun.plot_surface(elevs, new_path, specs_plotted[surf])
    
print('Well done dude!')