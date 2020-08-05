#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Apr 11 09:42:18 2020

@author: tausiaj
"""

# basic
import numpy as np

# plotting
from matplotlib import pyplot as plt
import cmocean
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# additional*
import scipy.stats as stats
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_squared_error


def buoy_validation(validation, T, validation_type='bulk'):
    """ Validate data with buoy
        ------------
        Parameters
        validation_type: Type of comparison to be done
                         (bulk, sea, swell)
        ------------
        Returns
        Different auto explicative plots
    """
        
    # Initialize the data to validate
    if validation_type=='agg':
        validation = validation[['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy',
                                 'Hs_Agg', 'Tp_Agg', 'Dir_Agg']]
        validation = validation.dropna(axis=0, how='any')
        title = 'Agg parameters'
    elif validation_type=='spec':
        validation = validation[['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy',
                                 'Hs_Spec', 'Tp_Spec', 'Dir_Spec']]
        validation = validation.dropna(axis=0, how='any')
        title = 'Spec parameters'
    elif validation_type=='sea':
        validation = validation[['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy',
                                 'Hsea', 'Tpsea', 'Dirsea']]
        validation = validation.dropna(axis=0, how='any')
        title = 'Sea parameters'
    elif (validation_type=='swell1' or validation_type=='swell2'):
        validation = validation[['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy',
                                 'H'+validation_type, 
                                 'Tp'+validation_type, 
                                 'Dir'+validation_type]]
        validation = validation.dropna(axis=0, how='any')
        title = 'Swell parameters'
    else:
        message = 'Not a valid value for validation_type'
        return message
        
    print('--------------------------------------------------------')
    print(validation_type.upper() + ' VALIDATION will be performed')
    print('-------------------------------------------------------- \n ')
    
    names = ['Hs_Buoy', 'Tp_Buoy', 'Dir_Buoy',
             'Hs', 'Tp', 'Dir']
    
    for n, name in enumerate(validation.columns.values):
        validation.rename(columns={name: names[n]}, inplace=True)
        
    print('Validating and plotting validated data... \n ')
    print('Length of data to validate: ' + str(len(validation)) + ' \n ')
        
    fig, axs = plt.subplots(2, 3, figsize=(20,20))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    fig.suptitle('Propagated hindcast: CSIRO' + 
                 ', ' + T + ' buoy validation \n ' +title, 
                 fontsize=24, y=0.98, fontweight='bold')
        
    for i in range(2):
        for j in range(3):
            if (i==j==0 or i==1 and j==0):
                if i==0:
                    x, y = validation['Hs_Buoy'], \
                           validation['Hs']
                    title = 'Hs [m]'
                else:
                    x, y = validation['Tp_Buoy'], \
                           validation['Tp']
                    title = 'Tp [s]'
                        
                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy) 
                idx = z.argsort()                                                  
                x2, y2, z = x[idx], y[idx], z[idx]
                axs[i,j].scatter(x2, y2, c=z, s=1, cmap=cmocean.cm.haline)
                axs[i,j].set_xlabel('Boya', fontsize=12, 
                                    fontweight='bold')
                axs[i,j].set_ylabel('Modelo', fontsize=12, 
                                    fontweight='bold')
                axs[i,j].set_title(title, fontsize=12, 
                                   fontweight='bold')
                maxt = np.ceil(max(max(x)+0.1, max(y)+0.1))
                axs[i,j].set_xlim(0, maxt)
                axs[i,j].set_ylim(0, maxt)
                axs[i,j].plot([0, maxt], [0, maxt], '-k', linewidth=0.6)
                axs[i,j].set_xticks(np.linspace(0, maxt, 5)) 
                axs[i,j].set_yticks(np.linspace(0, maxt, 5))
                axs[i,j].set_aspect('equal')
                xq = stats.probplot(x2, dist="norm")
                yq = stats.probplot(y2, dist="norm")
                axs[i,j].plot(xq[0][1], yq[0][1], "o", markersize=1, 
                              color='k', label='Q-Q plot')
                mse = mean_squared_error(x2, y2)
                rmse_e = rmse(x2, y2)
                BIAS = bias(x2, y2)
                SI = si(x2, y2)
                label = '\n'.join((
                        r'RMSE = %.2f' % (rmse_e, ),
                        r'mse =  %.2f' % (mse,  ),
                        r'BIAS = %.2f' % (BIAS,  ),
                        R'SI = %.2f' % (SI,  )))
                axs[i,j].text(0.7, 0.05, label, 
                              transform=axs[i,j].transAxes)
                    
            elif (i==0 and j==1 or i==0 and j==2):
                idx_buoy = validation['Tp_Buoy'].argsort()
                idx_hind = validation['Tp'].argsort()
                if j==1:
                    x, y = validation['Dir_Buoy'][idx_buoy], \
                           validation['Hs_Buoy'][idx_buoy]
                    index = 2
                    c = validation['Tp_Buoy'][idx_buoy]
                    title = 'Boya'
                else:
                    x, y = validation['Dir'][idx_hind], \
                           validation['Hs'][idx_hind]
                    index = 3
                    c = validation['Tp'][idx_hind]
                    title = 'Modelo'
                x = (x*np.pi)/180
                axs[i,j].axis('off')
                axs[i,j] = fig.add_subplot(2, 3, index, projection='polar')
                c = axs[i,j].scatter(x, y, c=c, s=5, cmap='magma_r', 
                                     alpha=0.75)
                cbar = plt.colorbar(c, pad=0.1)
                cbar.ax.set_ylabel('Tp [s]', fontsize=12, 
                                   fontweight='bold')
                axs[i,j].set_theta_zero_location('N', offset=0)
                axs[i,j].set_xticklabels(['N', 'NE', 'E','SE', 
                                          'S', 'SW', 'W', 'NW'])
                axs[i,j].set_theta_direction(-1)
                axs[i,j].set_xlabel('Dir [º]', fontsize=12, 
                                    fontweight='bold')
                axs[i,j].set_ylabel('Hs [m]', labelpad=20, fontsize=12, 
                                    fontweight='bold')
                axs[i,j].set_title(title, pad=15, fontsize=12, 
                                   fontweight='bold')
                    
            else:
                if j==1:
                    x, y = validation['Tp_Buoy'], \
                           validation['Hs_Buoy']
                    c = 'darkblue'
                    title = 'Boya'
                else:
                    x, y = validation['Tp'], \
                           validation['Hs']
                    c = 'red'
                    title = 'Modelo'
                xy = np.vstack([x, y])
                z = gaussian_kde(xy)(xy) 
                idx = z.argsort()                                                  
                x2, y2, z = x[idx], y[idx], z[idx]
                axs[i,j].scatter(x2, y2, c=z, s=3, cmap='Blues_r')
                axs[i,j].set_xlabel('Tp [s]', fontsize=12, 
                                    fontweight='bold')
                axs[i,j].set_ylabel('Hs [m]', fontsize=12, 
                                    fontweight='bold')
                axs[i,j].set_title(title, fontsize=12, 
                                   fontweight='bold')
                axs[i,j].set_xlim(0, 20)
                axs[i,j].set_ylim(0, 7.5)
                
#-----------------------------------------------------------------------------#
# Metric and accuracy functions
def rmse(pred, tar):
    return np.sqrt(((pred - tar) ** 2).mean())

def bias(pred, tar):
    return sum(pred - tar) / len(pred)

def si(pred, tar):
    S = pred.mean()
    O = tar.mean()
    return np.sqrt(sum(((pred-S) - (tar-O)) ** 2) / ((sum(tar ** 2))))

