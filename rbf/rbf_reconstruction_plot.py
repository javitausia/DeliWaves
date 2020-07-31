# common
import sys
import os.path as op

# basic
import pandas as pd
import numpy as np
from datetime import timedelta as td
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

import warnings
warnings.filterwarnings("ignore")

# custom
import rbf_functions as rbff

p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))

# -------------- EDIT THIS PART --------------------------------------------- #
name = 'SAF' # used name in the SWAN section
# --------------------------------------------------------------------------- #

print('Extracting CSIRO information...')

csiro = pd.read_pickle(op.join(p_data, 'reconstructed',
                               'reconstructed_partitioned_'+name+'.pkl'))
Tm_02 = csiro['Tm_02'].mean(axis=1)
csiro = csiro.drop(columns=['Tm_02'])
csiro['Tm_02'] = Tm_02

# First copy to play with NaNs
agg = csiro.copy()
agg[['Tpsea', 'Tpswell1', 'Tpswell2', 'Tpswell3']] = \
  agg[['Tpsea', 'Tpswell1', 'Tpswell2', 'Tpswell3']].fillna(np.inf)
agg = agg.fillna(0.0)

# Bulk Hs
csiro['Hs_Agg'] = np.sqrt(
        agg['Hsea']**2 +
        agg['Hswell1']**2 +
        agg['Hswell2']**2 +
        agg['Hswell3']**2
        )

# Bulk Tp
csiro['Tp_Agg'] = np.sqrt(
        csiro['Hs_Agg']**2 / (agg['Hsea']**2/agg['Tpsea']**2 + 
                          agg['Hswell1']**2/agg['Tpswell1']**2 +
                          agg['Hswell2']**2/agg['Tpswell2']**2 +
                          agg['Hswell3']**2/agg['Tpswell3']**2)
        )
        
# Second copy to play with NaNs
agg = csiro.copy().fillna(0.0)

# Bulk Dir
csiro['Dir_Agg'] = np.arctan(
        (agg['Hsea']*agg['Tpsea']*np.sin(agg['Dirsea']*np.pi/180) +
         agg['Hswell1']*agg['Tpswell1']*np.sin(agg['Dirswell1']*np.pi/180) +
         agg['Hswell2']*agg['Tpswell2']*np.sin(agg['Dirswell2']*np.pi/180) +
         agg['Hswell3']*agg['Tpswell3']*np.sin(agg['Dirswell3']*np.pi/180)) /
        (agg['Hsea']*agg['Tpsea']*np.cos(agg['Dirsea']*np.pi/180) +
         agg['Hswell1']*agg['Tpswell1']*np.cos(agg['Dirswell1']*np.pi/180) +
         agg['Hswell2']*agg['Tpswell2']*np.cos(agg['Dirswell2']*np.pi/180) +
         agg['Hswell3']*agg['Tpswell3']*np.cos(agg['Dirswell3']*np.pi/180))
        )
csiro['Dir_Agg'] = csiro['Dir_Agg']*180/np.pi
csiro['Dir_Agg'] = csiro['Dir_Agg'].where(csiro['Dir_Agg']>0, 
                                          csiro['Dir_Agg']+360)

# Wind features
csiro_wind = pd.read_pickle(op.join(p_data, 'hindcast',
                            'csiro_dataframe_sat_corr.pkl'))
csiro = csiro.join(csiro_wind[['W', 'DirW']])

## In case coastal buoy exists
#print('Extracting Buoy information...')
#
#buoy = pd.read_pickle(op.join(p_data, 'buoy', 'COAST-BUOY FILE NAME'))
#buoy.index = buoy.index.round('H')
#buoy = buoy.drop_duplicates()
#buoy_index = sorted(buoy.index.values)

## In case spectral reconstruction has been performed
#print('Concatinating and plotting data...')
#
#csiro_spec = pd.read_csv(op.join(p_data, 'spectra',
#                                 'NAME_specsrecon.csv'))
#csiro_spec.index = csiro_spec[csiro_spec.columns.values[0]].values
#csiro_spec = csiro_spec.drop(columns=csiro_spec.columns.values[0]) 
#csiro = csiro.join(csiro_spec)

csiro.to_pickle(op.join(p_data, 'reconstructed', 
                        'reconstructed_'+name+'.pkl'))

print(csiro.info())

sys.exit()

# --------------------------------------------------------------------------- #
# ----------- RUN THIS PART IF BUOY AND SPECTRA ARE AVAILABE ---------------- #
# --------------------------------------------------------------------------- #

total = csiro.join(buoy, how='inner')

total_plot = total[['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy',
                    'Hs_Agg', 'Tp_Agg', 'Dir_Agg',
                    'Hs_Spec', 'Tp_Spec', 'Dir_Spec',
                    'Hsea', 'Tpsea', 'Dirsea',
                    'Hswell1', 'Tpswell1', 'Dirswell1']]
        
labels = ['Hs [m]', 'Tp [s]', 'Dir [º]']

validation_data = total_plot.copy()

register_matplotlib_converters()
year = 2005
ini = str(year)+'-01-01 00:00:00'
end = str(year)+'-12-31 23:00:00'
total_plot = total_plot.loc[ini:end]
fig, axs = plt.subplots(3, 1, figsize=(20,15), sharex=True)
fig.subplots_adjust(hspace=0.05, wspace=0.1)
fig.suptitle('Year: ' +str(year)+ ', ' +name+ ' buoy compared with propagated CSIRO',
             fontsize=24, y=0.94, fontweight='bold')
months = ['                        Jan', '                        Feb', '                        Mar', 
          '                        Apr', '                        May', '                        Jun', 
          '                        Jul', '                        Aug', '                        Sep', 
          '                        Oct', '                        Nov', '                        Dec']
i = 0
while i < 3:
    if i==2:
        axs[i].plot(total_plot[total_plot.columns.values[i]], '.', markersize=1, color='darkblue')
        axs[i].plot(total_plot[total_plot.columns.values[i+3]], '.', markersize=1, color='red')
        axs[i].plot(total_plot[total_plot.columns.values[i+6]], '.', markersize=1, color='darkgreen')
        #axs[i].plot(total_plot[total_plot.columns.values[i+9]], '.', markersize=1, color='orange')
        #axs[i].plot(total_plot[total_plot.columns.values[i+12]], '.', markersize=1, color='purple')
        axs[i].set_ylabel(labels[i], fontsize=12, fontweight='bold')
        axs[i].grid()
        axs[i].set_xlim(ini, end)
        axs[i].set_xticks(np.arange(pd.to_datetime(ini), pd.to_datetime(end), td(days=30.5)))
        axs[i].tick_params(direction='in')
        axs[i].set_xticklabels(months, fontsize=12, fontweight='bold')
    else:
        axs[i].plot(total_plot[total_plot.columns.values[i]], color='darkblue', linewidth=1)
        axs[i].plot(total_plot[total_plot.columns.values[i+3]], color='red', linewidth=1)
        axs[i].plot(total_plot[total_plot.columns.values[i+6]], color='darkgreen', linewidth=1)
        #axs[i].plot(total_plot[total_plot.columns.values[i+9]], color='orange', linewidth=1)
        #axs[i].plot(total_plot[total_plot.columns.values[i+12]], color='purple', linewidth=1)
        axs[i].set_ylabel(labels[i], fontsize=12, fontweight='bold')
        axs[i].grid()
        axs[i].tick_params(direction='in')
    fig.legend(['Buoy', 'CSIRO Agg', 'CSIRO Spec'], loc=(0.65, 0.04), ncol=3, fontsize=14)
    i += 1
    
rbff.buoy_validation(validation_data, 'BUOY NAME', 'agg')
rbff.buoy_validation(validation_data, 'BUOY NAME', 'spec')