import sys
import os.path as op
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mda_functions import MaxDiss_Simplified_NoThreshold as MDA

print('Reading, obtaining and plotting data...')

# VARIABLES
num_centroids = 300

# DATAFRAME
p_hind = op.abspath(op.join(op.dirname(__file__), '..', 'data', 'hindcast'))
dataframe_total = pd.read_pickle(op.join(p_hind, 'csiro_dataframe_sat_corr.pkl'))
sea = dataframe_total.copy()
swell = dataframe_total.copy()

print(dataframe_total.info())

# SEA DATAFRAME
xys_sea = ['Hsea', 'Tpsea', 'Dirsea', 'Sprsea', 'W', 'DirW']
sea = sea[xys_sea].dropna(axis=0, how='any')
sea['Sprsea'][sea['Sprsea']<5] = 5

# SWELL DATAFRAME
xys_swell = ['Tm_02',
             'Hswell1', 'Tpswell1', 'Dirswell1', 'Sprswell1',
             'Hswell2', 'Tpswell2', 'Dirswell2', 'Sprswell2',
             'Hswell3', 'Tpswell3', 'Dirswell3', 'Sprswell3']
swell = swell[xys_swell]
swell = pd.DataFrame({'Tm_02': np.repeat(swell['Tm_02'], 3),
                      'Hswell': np.concatenate((swell['Hswell1'].values,
                                                swell['Hswell2'].values,
                                                swell['Hswell3'].values),
                                               axis=0),
                      'Tpswell': np.concatenate((swell['Tpswell1'].values,
                                                 swell['Tpswell2'].values,
                                                 swell['Tpswell3'].values),
                                                axis=0),
                      'Dirswell': np.concatenate((swell['Dirswell1'].values,
                                                  swell['Dirswell2'].values,
                                                  swell['Dirswell3'].values),
                                                 axis=0),
                      'Sprswell': np.concatenate((swell['Sprswell1'].values,
                                                  swell['Sprswell2'].values,
                                                  swell['Sprswell3'].values),
                                                 axis=0)})
swell = swell.dropna(axis=0, how='any')
swell['Sprswell'][swell['Sprswell']<5] = 5
swell['Gamma'] = np.power(swell['Tpswell'].values/\
                          (swell['Tm_02'].values * 1.411), 
                          -12.5439)
swell['Gamma'][swell['Gamma']<1] = 1
swell['Gamma'][swell['Gamma']>50] = 50

# MDA
data = [sea, swell]
labels = [['Hsea', 'Tpsea', 'Dirsea', 'Sprsea', 'W', 'DirW'],
          ['Hswell', 'Tpswell', 'Dirswell', 'Sprswell', 'Gamma']]
names = [['$H_S$ [m]', '$T_P$ [s]', '$\u03B8_{m}$ [$\degree$]', 
          '$\sigma_{\u03B8}$ [$\degree$]', '$W_{speed}$ [m/s]', '$\u03B8_{W}$ [$\degree$]'],
         ['$H_S$ [m]', '$T_P$ [s]', '$\u03B8_{m}$ [$\degree$]', 
          '$\sigma_{\u03B8}$ [$\degree$]', '$\gamma$']]
title = ['sea', 'swell']
scalars = [[0,1,3,4], [0,1,3,4]]
directionals = [[2,5], [2]]

for wave in range(0,2):
    
    dataframe = data[wave]
    xys = labels[wave]
    xys_names = names[wave]
    scalar = scalars[wave]
    directional = directionals[wave]
    
    dataframe = dataframe[xys]
    dataarray = dataframe.to_numpy()

    # SUBSET
    subarray = MDA(dataarray, num_centroids, scalar, directional)
    subset = pd.DataFrame(subarray, columns=xys)

    # Scatter plot MDA centroids
    fig, axs = plt.subplots(len(xys)-1, len(xys)-1, figsize = (20,20))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.suptitle('Scatter plot for the different variables. \n ' + 
                 'All points (black) and MDA points (blue) \n ' +
                 title[wave].upper(), fontsize=16, fontweight='bold')

    for i, item in enumerate(xys):
        xys_copy = xys.copy()
        xys_names_copy = xys_names.copy()
        for p in range(i+1):
            xys_copy.pop(0)
            xys_names_copy.pop(0)
        for j, jtem in enumerate(xys_copy):
            axs[i,j+i].scatter(dataframe[jtem], dataframe[item], marker=".", 
                               s=1, color = 'lightblue')
            axs[i,j+i].scatter(subset[jtem], subset[item], marker=".", 
                               s=2,  color = 'k')
            axs[i,j+i].set_xlim(min(dataframe[jtem]), max(dataframe[jtem]))
            axs[i,j+i].set_ylim(min(dataframe[item]), max(dataframe[item]))
            if i==j+i:
                axs[i,j+i].set_xlabel(xys_names_copy[j], fontsize=12, 
                                      fontweight='bold')
                axs[i,j+i].set_ylabel(xys_names[i], fontsize=12, 
                                      fontweight='bold')
            else:
                axs[i,j+i].set_xticks([])
                axs[i,j+i].set_yticks([])
            len_xys = len(xys)-len(xys_copy)-1
            if len_xys != 0:
                for k in range(len_xys):
                    axs[i,k].axis('off')
                
    subset.to_pickle(op.join(p_hind, title[wave] + '_cases_' + str(num_centroids) + '.pkl'))