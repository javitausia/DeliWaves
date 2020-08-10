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
test = pd.DataFrame({'Hsea':      [1.5,     np.nan,    1.7,    2.6],
                     'Tpsea':     [7.2,     np.nan,    18,     18],
                     'Dirsea':    [70,      np.nan,    345,    70],
                     'Sprsea':    [24,      np.nan,     15,     30],
                     'Hswell1':   [2.0,     2.0,    np.nan, np.nan],
                     'Tpswell1':  [16,      16,     np.nan, np.nan],
                     'Dirswell1': [350,     250,    np.nan, np.nan],
                     'Sprswell1': [12,      12,     np.nan, np.nan],
                     'Hswell2':   [np.nan,  0.8,    np.nan, np.nan],
                     'Tpswell2':  [np.nan,  18,     np.nan, np.nan],
                     'Dirswell2': [np.nan,  80,    np.nan, np.nan],
                     'Sprswell2': [np.nan,  12,     np.nan, np.nan],
                     'Hswell3':   [np.nan,  np.nan, np.nan, np.nan],
                     'Tpswell3':  [np.nan,  np.nan, np.nan, np.nan],
                     'Dirswell3': [np.nan,  np.nan, np.nan, np.nan],
                     'Sprswell3': [np.nan,  np.nan, np.nan, np.nan]})

heights = test.copy().fillna(0.0)
periods = test[['Tpsea', 'Tpswell1', 'Tpswell2', 'Tpswell3']].copy().fillna(np.inf)

test['Tm_02'] = np.sqrt(
        (heights['Hsea']**2 + heights['Hswell1']**2 +  
         heights['Hswell2']**2 + heights['Hswell1']**2) / (heights['Hsea']**2 / periods['Tpsea']**2 + 
                                                           heights['Hswell1']**2 / periods['Tpswell1']**2 +
                                                           heights['Hswell2']**2 / periods['Tpswell2']**2 +
                                                           heights['Hswell3']**2  /periods['Tpswell3']**2)
        )

gamma_values = [[3, 3, 3, 3],       # Sea1, Sea2, Sea3, Sea4
                [10, 10, 10, 10],   # Swell1, Swell2, Swell3, Swell4
                [10, 10, 10, 10],   # ...
                [10, 10, 10, 10]]   # ...

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