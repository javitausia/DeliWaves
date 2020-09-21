# basic
import sys
import os
import os.path as op

# data libraries
import numpy as np
import pandas as pd
# import xarray as xr
import datetime
from datetime import timedelta as td

# plots
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
from pandas.plotting import register_matplotlib_converters
from termcolor import colored

# warnings
import warnings
warnings.filterwarnings('ignore')

# Slopes class
class Slopes(object):
    """
        This class Slopes analyze detailed aspects about the surfbreak.
        The historic dataframe is proportioned in the required beach and
        then, giving information about its location, its D50 and more
        characteristics, the propagation to shallower waters is performed
    """
    
    def __init__(self, reconstructed_data, tides, delta_angle, wf, name,
                 reconstructed_depth = 10):
        """
            The initializator is essential as it constructs the main
            object that we will later use for the visualization tools.
            both the SOM and usual plots
            ------------
            Parameters
            reconctructed_data: it is a dataframe with the historical data
                                and all the requested variables for each
                                individual beach
            tides: another dataframe with the sea level at each time
            delta_angle: the relative direction of the beach to the N, so
                         the posterior wave clustering analysis can be
                         performed correctly
            wf: sediment fall velocity, which is calculated from the D50
                at each beachbreak. A detailed explanation will be available
                in the notebook
            name: name of the surfbreak
            reconstructed_depth: depth where the reconstruction has been
                                 done, which is usually 10 for snell
            ------------
            Returns
            The initialized dataframe for the surfbreak with all the
            mentioned variables joined
        """
        
        # We first initialize the main variables
        self.data   =  reconstructed_data
        self.tides  =  tides
        self.angle  =  delta_angle
        self.wf     =  wf
        self.name   =  name
        
        # And now, we perform the pertinent actions
        
        # 1. Omega calculation
        data_mean = self.data[['Hs_Agg', 'Tp_Agg']].copy()
        data_mean = data_mean.rolling(window=31*24).mean()
        data_mean['Omega'] = data_mean['Hs_Agg'] / \
                               (self.wf*data_mean['Tp_Agg'])
        print('\n Rolling mean and $\u03A9$ calculated!! \n')
        
        # 2. Relative direction calculation and renaming
        # 2.1 Renaming for Agg o Spec
        msg  =  '\n For aggregated parameters (Agg) usage say True \n, '
        msg +=  'for Spectral parameters (Spec), say False (empty box): \n'
        ans  =  bool(input(msg))
        if ans:
            self.data = self.data[['Hs_Agg', 'Tp_Agg', 'Dir_Agg', 
                                   'Spr_Spec', 'W', 'DirW']].rename(
                          columns={'Hs_Agg': 'Hs', 'Tp_Agg': 'Tp', 
                                   'Dir_Agg': 'Dir',
                                   'Spr_Spec': 'Spr'}).copy()
        else:
            self.data = self.data[['Hs_Spec', 'Tp_Spec', 'Dir_Spec', 
                                   'Spr_Spec', 'W', 'DirW']].rename(
                          columns={'Hs_Spec': 'Hs', 'Tp_Spec': 'Tp', 
                                   'Dir_Spec': 'Dir', 
                                   'Spr_Spec': 'Spr'}).copy()
        # 2.2 Relative direction calculation
          # directions
        ddirss = self.data['Dir'] - self.angle
        ddirs = []
        for ddir in ddirss:
            if ddir>180:
                ddirs.append(ddir - 360)
            elif ddir>0:
                ddirs.append(ddir)
            elif ddir>(-180):
                ddirs.append(ddir)
            else:
                ddirs.append(360 + ddir)
        self.data['DDir'] = ddirs
        print('\n Mean wave direction: {} \n'.format(self.data.DDir.mean()))
        self.data['DDir'] = self.data['DDir'] * np.pi / 180
          # wind
        ddirssw = self.data['DirW'] - self.angle
        ddirsw = []
        for ddirw in ddirssw:
            if ddirw>180:
                ddirsw.append(ddirw - 360)
            elif ddirw>0:
                ddirsw.append(ddirw)
            elif ddirw>(-180):
                ddirsw.append(ddirw)
            else:
                ddirsw.append(360 + ddirw)
        self.data['DDirW'] = ddirsw
        print('\n Mean wind direction: {} \n'.format(self.data.DDirW.mean()))
        self.data['DDirW'] = self.data['DDirW'] * np.pi / 180
        
        # 3. Tides and omega joining
        self.data = self.data.join(np.abs(tides-tides.max()), how='inner')
        self.data = self.data.join(data_mean['Omega'], how='inner')
        self.data = self.data.dropna(how='any', axis=0)
        
        # 4. Tidal range value
        self.TR = float(input('\n Select the tidal range (TR): \n'))
        
        # 5. Wave height at break calculation, Hb
        # 5.1 Wave height
        break_th = float(input('\n Select the value for $\u03B3$: \n'))
        waves_br = []
        for hs, th in zip(self.data['Hs'], self.data['DDir']):
            hb_diff = 1
            hb_app  = hs
            for hb in np.linspace(hs, 3*hs, 50):
                eq = break_th * hb - hs * (reconstructed_depth/hb)**(1/4) * \
                     np.sqrt(np.cos(th) / np.cos(np.arcsin(np.sqrt(hb/reconstructed_depth) * \
                     np.sin(th))))
                if abs(eq) < hb_diff:
                    hb_diff = abs(eq)
                    hb_app = hb
            waves_br.append(hb_app)
        self.data['H_break'] = waves_br
        # 5.2 Wave direction, ¡Snell refraction!
        self.data['DDir_R'] = np.arcsin(np.sqrt(
                self.data['H_break']/reconstructed_depth) * \
                np.sin(self.data['DDir']))
        print('\n Heights asomerament difference: Hb / Hs : {} \n'.format(
                (self.data['H_break']/self.data['Hs']).mean()))
        
        print(colored('\n Slopes main object constructed!! \n', 'red', 
                      attrs=['blink']))
        
        
    def perform_propagation(self, profile_type='biparabolic'):
        """
            This function obtains the value for the slope of the beach just
            indicating the type of profie that wants to be used
        """
            
        if profile_type=='biparabolic':
            slopes = []
            for wave in range(len(self.data)):
                # Range to calculate the slope [m]
                #L = np.sqrt(9.8*waves.iloc[wave].H_break) * waves.iloc[wave].Tp
                #slope_range = L/2
                slope_range = 0.75 #en base a L/2
                if wave%10000==0:
                    print('{} waves analyzed...'.format(wave))
                # Empirical adjusted parameters
                A = 0.21 - 0.02 * self.data.iloc[wave].Omega
                B = 0.89 * np.exp(-1.24 * self.data.iloc[wave].Omega)
                C = 0.06 + 0.04 * self.data.iloc[wave].Omega
                D = 0.22 * np.exp(-0.83 * self.data.iloc[wave].Omega)
                # Asomerament height
                h = self.data.iloc[wave].H_break + \
                    self.data.iloc[wave].ocean_tide
                h_values = np.linspace(h, h+slope_range, 30)
                hr = 1.1 * self.data.iloc[wave].Hs + self.TR
                # Lines for the profile
                x  = []
                X  = []
                if h<hr:
                    xr = (h/A)**(3/2) + (B/(A**(3/2)))*h**3
                else:
                    xr = (h/C)**(3/2) + (D/(C**(3/2)))*h**3
                for h_value in h_values:
                    if h_value<hr:
                        x_max = 0
                        xapp = (h_value/A)**(3/2) + (B/(A**(3/2)))*h_value**3
                        x.append(xapp)
                        x_max = max(xapp, x_max)
                        # if xapp > (xr+slope_range):
                        #     h_diff = h_value-h
                        #     x_len  = xapp-xr
                        #     break
                    else:
                        Xapp = (h_value/C)**(3/2) + (D/(C**(3/2)))*h_value**3
                        if (h_value-hr)<0.1:
                            try:
                                x_diff = x_max - Xapp
                            except:
                                x_diff = 0
                        X.append(Xapp)
                        # if Xapp > (xr+slope_range):
                        #     h_diff = h_value-h
                        #     x_len  = Xapp-xr
                        #     break
                try:
                    x_tot = np.concatenate((np.array(x), 
                                            np.array(X)+x_diff))
                except:
                    if len(x)!=0:
                        x_tot = np.array(x)
                    else:
                        x_tot = np.array(X)
                        x_len = x_tot[-1]-x_tot[0]
                slopes.append(slope_range/x_len)
            self.data['Slope'] = slopes
            self.data['Iribarren'] = self.data['Slope'] / \
                                            np.sqrt(self.data['H_break'] / \
                                            ((9.8/2*np.pi)*self.data['Tp']**2))
                             
        # Print the constructed data
        print(colored('\n Slopes main object finally constructed!! \n', 
                      'red', attrs=['blink']))
        print(self.data.info())
            
            
    def plot_profile(self, year=2018):
        """
            Plots the profile features for the selected year
        """
        
        # Plot the profile
        register_matplotlib_converters()
        ini = str(year)+'-01-01 00:00:00'
        end = str(year)+'-12-31 23:00:00'
        waves_plot = self.data.loc[ini:end]
        fig, axs = plt.subplots(4, 1, figsize=(20,15), sharex=True)
        fig.subplots_adjust(hspace=0.05, wspace=0.1)
        fig.suptitle('Year: '+str(year)+', '+str(year)+' H$_{S,B}$, T$_P$, $\Omega$, Slope and Iribarren number', 
                     fontsize=24, y=0.94, fontweight='bold')
        months = ['                        Jan', '                        Feb', '                        Mar', 
                  '                        Apr', '                        May', '                        Jun', 
                  '                        Jul', '                        Aug', '                        Sep', 
                  '                        Oct', '                        Nov', '                        Dec']
        axs[0].plot(waves_plot['Hs'], color='k', linewidth=1, label='H$_S$ [m]')
        axs[0].plot(waves_plot['H_break'], color='grey', linewidth=1, label='H$_B$ [m]')
        axs[0].legend()
        axs[0].grid()
        axs[0].tick_params(direction='in')
        axs[1].plot(waves_plot['Tp'], color='darkblue', linewidth=1, label='T$_P$ [s]')
        axs[1].legend()
        axs[1].grid()
        axs[1].tick_params(direction='in')
        axs[2].plot(waves_plot['Slope'], color='darkgreen', lw=0.5, label='Actual beach slope')
        axs[2].legend()
        axs[2].grid()
        axs[2].tick_params(direction='in')
        axs[3].plot(waves_plot['Iribarren'], color='darkred', lw=0.5, label='nº Iribarren')
        axs[3].legend()
        axs[3].grid()
        axs[3].set_xlim(ini, end)
        axs[3].set_ylim(0, 3)
        axs[3].set_xticks(np.arange(pd.to_datetime(ini), pd.to_datetime(end), 
                          datetime.timedelta(days=30.5)))
        axs[3].tick_params(direction='in')
        axs[3].set_xticklabels(months, fontsize=12, fontweight='bold')
        
        
    def moving_profile(self, year=2018):
        """
            A function to see how the profile can variate along a year
        """
        # Plot the profile
        fig = plt.figure(figsize=(20,15), tight_layout=True)
        gs  = gridspec.GridSpec(6, 1)
        # Plot for hs, tp and slopes
        ini = str(year)+'-01-01 00:00:00'
        end = str(year)+'-12-31 23:00:00'
        slopes_list = self.data.loc[ini:end]
        months = ['                        Jan', '                        Feb', '                        Mar', 
                  '                        Apr', '                        May', '                        Jun', 
                  '                        Jul', '                        Aug', '                        Sep', 
                  '                        Oct', '                        Nov', '                        Dec']
        ax0 = fig.add_subplot(gs[0, :])
        ax0.plot(slopes_list['Hs'], color='k', linewidth=1, label='H$_S$ [m]')
        ax0.plot(slopes_list['H_break'], color='grey', linewidth=1, label='H$_B$ [m]')
        ax0.legend(fontsize=10)
        ax0.grid()
        ax0.set_xlim(ini, end)
        ax0.set_xticks(np.arange(pd.to_datetime(ini), pd.to_datetime(end), td(days=30.5)))
        ax0.tick_params(direction='in')
        ax0.set_xticklabels([])
        ax1 = fig.add_subplot(gs[1, :])
        ax1.plot(slopes_list['Tp'], color='darkblue', linewidth=1, label='T$_P$ [s]')
        ax1.legend(fontsize=10)
        ax1.grid()
        ax1.set_xlim(ini, end)
        ax1.set_xticks(np.arange(pd.to_datetime(ini), pd.to_datetime(end), td(days=30.5)))
        ax1.tick_params(direction='in')
        ax1.set_xticklabels([])
        ax2 = fig.add_subplot(gs[2, :])
        ax2.plot(slopes_list['Slope'], color='darkgreen', linewidth=1, label='Actual tide slope')
        ax2.legend(fontsize=10)
        ax2.grid()
        ax2.set_xlim(ini, end)
        ax2.set_xticks(np.arange(pd.to_datetime(ini), pd.to_datetime(end), td(days=30.5)))
        ax2.tick_params(direction='in')
        ax2.set_xticklabels([])
        ax3 = fig.add_subplot(gs[3, :])
        ax3.plot(slopes_list['Iribarren'], color='darkred', linewidth=1, label='nº Iribarren')
        ax3.legend(fontsize=10)
        ax3.grid()
        ax3.set_xlim(ini, end)
        ax3.set_xticks(np.arange(pd.to_datetime(ini), pd.to_datetime(end), td(days=30.5)))
        ax3.tick_params(direction='in')
        ax3.set_xticklabels(months, fontsize=12, fontweight='bold')
        # Plot for the profiles
        # Slopes_list to start with
        slopes_list = slopes_list[::100]
        ax3 = fig.add_subplot(gs[4:, :])
        ax3.set_ylim(-10, 2)
        ax3.set_xlim(0, 1000)
        # Colors to be used in the profiles
        colors = LinearSegmentedColormap.from_list('mycmap', ['black', 
                                                              'grey',
                                                              'lime',
                                                              'grey',
                                                              'black'])
        colors = colors(np.linspace(0, 1, len(slopes_list)))
        for s in range(len(slopes_list)):
            # Discontinuity point
            hr = 1.1 * slopes_list['Hs'][s] + self.TR
            # Legal point
            #ha = 3 * slopes_list['Hs'][s] + TR
            # Empirical adjusted parameters
            A = 0.21 - 0.02*slopes_list['Omega'][s]
            B = 0.89 * np.exp(-1.24*slopes_list['Omega'][s])
            C = 0.06 + 0.04*slopes_list['Omega'][s]
            D = 0.22 * np.exp(-0.83*slopes_list['Omega'][s])
            # Different values for the height
            h = np.linspace(0, 10, 150)
            # Important points for the profile
            #xr = (hr/A)**(3/2) + (B/(A**(3/2)))*hr**3
            #x0 = (hr/A)**(3/2) - (hr/C)**(3/2) + \
            #     (B/(A**(3/2)))*hr**3 - (D/(C**(3/2)))*hr**3
            # Lines for the profile
            x  = []
            X  = []
            for hs in h:
                if hs<hr:
                    x_max = 0
                    xapp = (hs/A)**(3/2) + (B/(A**(3/2)))*hs**3
                    x.append(xapp)
                    x_max = max(xapp, x_max)
                else:
                    Xapp = (hs/C)**(3/2) + (D/(C**(3/2)))*hs**3
                    if (hs-hr)<0.1:
                        x_diff = x_max - Xapp
                    X.append(Xapp)
                x_tot  = np.concatenate((np.array(x), 
                                         np.array(X)+x_diff))
            ax3.plot(x_tot, -h, color=colors[s], label=str(slopes_list.index[s]))
            #ax3.scatter(xr, -hr, s=10, c='red', label='Discontinuity point')
            #ax3.axhline(-ha, color='grey', ls='-.', label='Available region')
            #ax3.axhline(0, color='lightgrey', ls='--', label='HTL')
            #ax3.axhline(-TR[tr]/2, color='lightgrey', ls='--', label='MTL')
            #ax3.axhline(-TR[tr], color='lightgrey', ls='--', label='LTL')
            #ax3.axvline(x0, color='k', ls='--', label='Available region')
            ax3.legend(loc='upper right', fontsize=6, ncol=3)
            ax0.plot(slopes_list.index[s], slopes_list['Hs'][s], '.', 
                     color='red', markersize=5)
            ax0.plot(slopes_list.index[s], slopes_list['H_break'][s], '.', 
                     color='red', markersize=5)
            ax1.plot(slopes_list.index[s], slopes_list['Tp'][s], '.', 
                     color='red', markersize=5)
            # ax2.plot(slopes_list.index[s], slopes_list['LT_Slope'][s], '.', 
            #          color='red', markersize=5)
            # ax2.plot(slopes_list.index[s], slopes_list['MT_Slope'][s], '.', 
            #          color='red', markersize=5)
            # ax2.plot(slopes_list.index[s], slopes_list['HT_Slope'][s], '.', 
            #          color='red', markersize=5)

