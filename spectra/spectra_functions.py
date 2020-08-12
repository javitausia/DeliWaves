#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 5 09:42:18 2020

@author: tausiaj
"""

# basic
import os
import os.path as op

# common
from random import sample
import numpy as np
import xarray as xr
from scipy.special import gamma as gf
from time import time

# plotting
import matplotlib as mpl
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
import imageio


# SPECTRA AND FREE SURFACE
    
def spectra(data, gamma_values=False):
    # Constants
    a = 1.411
    b = -0.07972
    # Frequencys and directions to calculate
    freqs  = np.linspace(1/30, 1/3, 60) # more freqs: 120, 180...
    direcs = np.linspace(-np.pi, np.pi, 60) # more direcs: 120, 180...
    direcs_good_1 = np.linspace(270, 0, 40) # more direcs: 80, 120...
    direcs_good_2 = np.linspace(360, 270, 20) # more direcs: 40, 60...
    direcs_good = np.concatenate([direcs_good_1, direcs_good_2])
    # Labels to play
    labels = [['Hsea', 'Tpsea', 'Dirsea', 'Sprsea'],
              ['Hswell1', 'Tpswell1', 'Dirswell1', 'Sprswell1'],
              ['Hswell2', 'Tpswell2', 'Dirswell2', 'Sprswell2'],
              ['Hswell3', 'Tpswell3', 'Dirswell3', 'Sprswell3']]
    # Tm_02 values
    tm02 = data['Tm_02'].values
    # Grouped specs
    gspecs = []
    print('--------------------------------------------')
    print('{} sea states will be analyzed...'.format(len(data)))
    print('--------------------------------------------')
    # List of spectrums
    specs = [ [], [], [], [] ]
    for numpart, part in enumerate(labels):
        print('Calculating partition {}...'.format(numpart))
        # Values for the partition
        data_part = data[part]
        print('--------------------------------------------')
        t    =  data_part.index.values
        hs   =  data_part.iloc[:,0].values
        tp   =  data_part.iloc[:,1].values
        fp   =  1/tp
        dirr =  data_part.iloc[:,2].values
        spr  =  data_part.iloc[:,3].values
        spr  =  np.where(spr<5, 5, spr) # limit values for spr (sigma)
        spr  =  np.where(spr>80, 80, spr)
        spr  =  spr*np.pi/180
        s    =  (2/spr**2) - 1 # spectral shape parameter
        # Calculated values for the partition
        if gamma_values:
            gamma = gamma_values[numpart]
        else:
            gamma = np.exp((np.log(tp/(a*tm02)))/b)
            gamma = np.where(gamma<3, 3, gamma) # limit values for gamma
            gamma = np.where(gamma>50, 50, gamma)
        # Calculating the spectra
        for p in range(len(data_part)):
            if (dirr[p]>0 and dirr[p]<90):
                dirr[p] = (90-dirr[p])*np.pi/180
            elif (dirr[p]>90 and dirr[p]<180):
                dirr[p] = -(180-(dirr[p]-90))*np.pi/180
            elif (dirr[p]>180 and dirr[p]<270):
                dirr[p] = -(90-(dirr[p]-180))*np.pi/180
            else:
                dirr[p] = (90+(90-(dirr[p]-270)))*np.pi/180
            if (hs[p]>0 and tp[p]>0 and spr[p]>0):
                # Spectrum initializer
                spec = np.zeros((len(freqs), len(direcs)))
                S    = np.zeros(len(freqs))
                D    = np.zeros(len(direcs))
                for fi, f in enumerate(freqs):
                    if f < fp[p]:
                        sigma = 0.07
                    else:
                        sigma = 0.09
                    for di, d in enumerate(direcs):
                        alpha = 0.0624 / \
                                (0.230+0.0336*gamma[p]-0.185/(1.9+gamma[p])) * \
                                (1.094-0.01915*np.log(gamma[p]))
                        S[fi] = alpha * hs[p]**2 * tp[p]**(-4) * \
                                f**(-5) * np.exp(-1.25*(tp[p]*f)**(-4)) * \
                                gamma[p]**(np.exp(-(tp[p]*f-1)**2/(2*sigma**2)))
                        D[di] = ((2**(2*s[p]-1))/np.pi) * \
                                (gf(s[p]+1)**2/gf(2*s[p]+1)) * \
                                np.cos(np.abs(d-dirr[p])/2)**(2*s[p])
                        spec[fi, di]  =  S[fi] * D[di] * np.pi/180
                spec = np.where(spec>0, spec, 0.0)
                spec = spec.reshape(len(freqs), len(direcs), 1, 1)
                specs[numpart].append(xr.DataArray(spec, 
                                           coords={'freq': freqs, 
                                                   'dir': direcs_good,
                                                   'partition': [numpart],
                                                   'time': [t[p]]},
                                           dims=['freq', 
                                                 'dir', 
                                                 'partition',
                                                 'time'],
                                           name='efth'))
            else:
                specs[numpart].append(xr.DataArray(0.0, 
                                           coords={'freq': freqs, 
                                                   'dir': direcs_good,
                                                   'partition': [numpart],
                                                   'time': [t[p]]},
                                           dims=['freq', 
                                                 'dir', 
                                                 'partition',
                                                 'time'],
                                           name='efth'))
    # Concat DataArrays
    if len(data)>20:
        timesteps = int(len(data)/20)
    else:
        timesteps = 1
    timestep = int(len(data)/timesteps)
    print('Concatinating final spectrums in groups of {}...'.format(timestep))
    timestep_list = [ [], [], [], [] ]
    for ts in range(timesteps):
        for numpart in range(len(labels)):
            timestep_list[numpart].append(xr.concat(
                    specs[numpart][ts*timestep:(ts+1)*timestep], dim='time')
                    )
    for numpart in range(len(labels)):
        gspecs.append(xr.concat(timestep_list[numpart], dim='time'))
    return xr.concat(gspecs, dim='partition')

def plot_spectrum(specs, nplots=4, pcolor=True, time_plots=[0,1,2,3]):
    print('Plotting {} out of {} spectrums... \n'.format(nplots, len(specs.time)))
    if time_plots:
        print('Plotting specs: {}'.format(time_plots))
    else:
        time_plots = sample(list(specs.time.values), nplots)
    nfigs = int(np.sqrt(nplots))
    fig1, axs1 = plt.subplots(nfigs, nfigs, figsize=(15,15))
    fig1.suptitle('Spectrums for different sea states',
                  fontsize=16, fontweight='bold')
    fig1.subplots_adjust(hspace=0.5, wspace=0.2)
    #fig2, axs2 = plt.subplots(nfigs, nfigs, figsize=(15,15))
    #fig2.suptitle('Spectrums for different sea states in 3D',
    #              fontsize=16, fontweight='bold')
    #fig2.subplots_adjust(hspace=0.5, wspace=0.2)
    norm = LogNorm(vmin=0.0001, vmax=1)
    cm = mpl.cm.jet
    cm.set_bad(color='darkblue')
    for x, y in [(i, j) for i in range(nfigs) for j in range(nfigs)]:
        spec_plot = specs.sel(time=specs.time.values[time_plots[x+y*nfigs]])
        freqs  = spec_plot.freq.values
        direcs = spec_plot.dir.values*np.pi/180
        xx, yy = np.meshgrid(direcs, freqs)
        axs1[x,y].axis('off')
        #axs2[x,y].axis('off')
        axs1[x,y] = fig1.add_subplot(nfigs, nfigs, x+y*nfigs + 1, 
                                     projection='polar')
        #axs2[x,y] = fig2.add_subplot(nfigs, nfigs, x+y*nfigs +1,
        #                             projection='3d')
        spec_plot = np.where(spec_plot<0.0001, 0.0001, spec_plot)
        if pcolor==True:
            P  = axs1[x,y].pcolor(xx, yy, spec_plot, cmap=cm, norm=norm)
            #PD = axs2[x,y].plot_surface(xx, yy, spec_plot, cmap=cm,
            #                            antialiased=True, norm=norm)
        else:
            P = axs1[x,y].contourf(xx, yy, spec_plot, levels=100, cmap=cm)
        PC  = fig1.colorbar(P, ax=axs1[x,y])
        #PCD = fig2.colorbar(PD, ax=axs2[x,y])
        PC.set_label('Energy [m²/Hz·rad]', fontsize=10, fontweight='bold')
        #PCD.set_label('Energy [m²/Hz·rad]', fontsize=10, fontweight='bold')
        axs1[x,y].set_theta_zero_location('N', offset=0)
        axs1[x,y].set_theta_direction(-1)
        axs1[x,y].set_xticklabels(['N', 'NE', 'E','SE', 
                                   'S', 'SW', 'W', 'NW'])
        #axs2[x,y].set_xticklabels(['N', 'NE', 'E','SE', 
        #                           'S', 'SW', 'W', 'NW'])
        axs1[x,y].set_xlabel('$\u03B8$ [rad]', fontsize=12, 
                             fontweight='bold')
        axs1[x,y].set_ylabel('f [Hz]', labelpad=20, fontsize=12, 
                             fontweight='bold')
        #axs2[x,y].set_xlabel('Dir [rad]', fontsize=12, 
        #                     fontweight='bold')
        #axs2[x,y].set_ylabel('Freq [Hz]', labelpad=20, fontsize=12, 
        #                     fontweight='bold')
        axs1[x,y].tick_params(axis='y', colors='white')
        title = str(time_plots[x+y*nfigs]) + '\n' + \
                str(specs.time.values[time_plots[x+y*nfigs]])
        axs1[x,y].set_title(title, pad=25, fontsize=12, fontweight='bold')
        #axs2[x,y].set_title(time_plots[x+y*nfigs], pad=15, fontsize=12, 
        #                    fontweight='bold')
    return specs.time.values[time_plots]

def surface(data, t):
    # Constantes
    xlen    = 50               # Length x-field
    ylen    = 50               # Length y-field
    xmax    = 50               # Max x-field
    ymax    = 50               # Max y-field
    delta_d = np.pi/30         # delta-space directions axis
    delta_f = 0.005            # delta-space frequency axis
    # Elevs for all times
    televs = []
    # Time
    t0 = time()
    print('--------------------------------------------')
    print('Calculating the sea elevation for time {}...'.format(t))
    print('--------------------------------------------')
    # Values for the ampl
    ampl   = np.sqrt(data.values * 2 * delta_f * delta_d)
    print('Amplitude mean before selection: {}'.format(np.mean(ampl)))
    th     = np.quantile(ampl, 0.9)
    fr_ix  = np.unique(np.where(ampl>th)[0])
    dr_ix  = np.unique(np.where(ampl>th)[1])
    ampl   = ampl[fr_ix[0]:fr_ix[-1], dr_ix[0]:dr_ix[-1]]
    print('Amplitude mean after selection: {}'.format(np.mean(ampl)))
    print('--------------------------------------------')
    freqs  = data.freq.values[fr_ix]
    direcs = data.dir.values[dr_ix]*np.pi/180
    # k and w to calculate
    k      = (2*np.pi*freqs)**2 / 9.806
    w      = 2*np.pi*freqs
    # Matrix for the phases
    phi    = np.random.rand(len(freqs), len(direcs)) * 2 * np.pi
    # Matrix for elevation values 
    ele    = np.zeros((xlen, ylen))
    x      = np.linspace(0, xmax, xlen)
    y      = np.linspace(0, ymax, ylen)
    for xx in range(xlen):
        if xx%(xlen/10)==0:
            print('{}% completed...'.format(int((xx*100/xlen))))
            print('{} minutes transcurred...'.format(
                                                  round((time()-t0)/60, 2)
                                                  ))
            print('--------------------------------------------')
        for yy in range(ylen):
            for i in range(len(freqs)-1):
                for j in range(len(direcs)-1):
                    ele[xx, yy] += ampl[i, j] * \
                                       np.cos(w[i]*t - k[i] * \
                                       (x[xx]*np.cos(direcs[j]) + \
                                        y[yy]*np.sin(direcs[j])) + \
                                       + phi[i, j])
    ele = ele.reshape(xlen, ylen, 1)
    televs.append(xr.DataArray(ele, 
                               coords={'x': x, 
                                       'y': y,
                                       'time': [t]},
                               dims=['x', 
                                     'y', 
                                     'time']))
    return xr.concat(televs, dim='time')

def plot_surface(elevs, path, title):
    filenames = []
    x  = elevs.x.values
    y  = elevs.y.values 
    xx, yy = np.meshgrid(x, y)
    for nt in range(len(elevs.time)):
        fig, axs = plt.subplots(1, 1, figsize=(8,8))
        elev = elevs.sel(time=nt).values
        PS = axs.pcolor(xx, yy, elev, cmap='binary',
                        vmin=np.min(elevs.values.reshape(-1)),
                        vmax=np.max(elevs.values.reshape(-1)))
        cbar = fig.colorbar(PS)
        cbar.ax.set_ylabel('Elevation [m]', fontsize=18, fontweight='bold')
        fig.suptitle('Surface Reconstruction in {} s'.format(nt))
        axs.set_xlabel('X [m]', fontsize=16, fontweight='bold')
        axs.set_ylabel('Y [m]', fontsize=16, fontweight='bold')
        axs.axis('equal')
        fig.savefig(op.join(path, '{}.png'.format(nt)))
        filenames.append('{}.png'.format(nt))
        plt.clf()
    # GIF
    #path_images.sort()
    images = []
    os.chdir(path)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(str(title)+'.gif', images, duration = 0.5)    
            
    return 'Well done dude!!'
    
