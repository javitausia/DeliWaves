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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from pandas.plotting import register_matplotlib_converters
from termcolor import colored
from matplotlib.patches import RegularPolygon
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm

# clustering
from minisom import MiniSom

# warnings
import warnings
warnings.filterwarnings('ignore')


def consult(data, beaches, day, columns):
    """
    Consulting function for beaches, day and variable inputs
    
    Parameters
    ----------
    data : dataframe
        a dataframe with all the data, obtained in slopes_notebook.ipynb
    beaches : list of 3 items
        list of 3 beaches to consult the different variables
    day : str
        str with the day in format 'YYYY-MM-DD'
    columns : list of variables
        list with all the variables to consult the value

    Returns
    -------
    dataframe
        dataframe with the consulting
    """
    
    # append 'beach' column
    columns.append('beach')
    
    return data.where((
        data['beach']==beaches[0]) | (data['beach']==beaches[1]) | (data['beach']==beaches[2])
        ).dropna(how='all', axis=0).loc[day, columns]
    

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
            tides: another dataframe/nc with the sea level at each time
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
            mentioned and more variables joined
        """
        
        # We first initialize the main variables
        self.data   =  reconstructed_data
        self.tides  =  tides
        self.angle  =  delta_angle
        self.wf     =  wf
        self.name   =  name
        
        # And now, we perform the pertinent actions
        
        # 1. Omega shoaling calculation
        data_mean = self.data[['Hs_Agg', 'Tp_Agg']].copy()
        data_mean = data_mean.rolling(window=31*24).mean()
        data_mean['Omega'] = data_mean['Hs_Agg'] / \
                             (self.wf*data_mean['Tp_Agg'])
        print('\n Rolling mean and \u03A9 calculated!! \n')
        
        # 2. Relative direction calculation and renaming
        # 2.1 Renaming for Agg o Spec
        msg  =  '\n For aggregated parameters (Agg) usage say True \n, '
        msg +=  'for Spectral parameters (Spec), say False (empty box): \n'
        ans  =  True # bool(input(msg))
        if ans:
            self.data = self.data[['Hs_Agg', 'Tp_Agg', 'Dir_Agg', 
                                   'Spr_Spec', 'W', 'DirW']].rename(
                          columns={'Hs_Agg': 'Hs', 'Tp_Agg': 'Tp', 
                                   'Dir_Agg': 'Dir',
                                   'Spr_Spec': 'Spr'}).copy()
            self.data.interpolate(method='linear', 
                                  limit_direction='forward', 
                                  axis=0, inplace=True)
            self.data = self.data[::5].copy()
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
        print('\n Mean wave direction: {} º \n'.format(self.data.DDir.mean()))
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
        print('\n Mean wind direction: {} º \n'.format(self.data.DDirW.mean()))
        self.data['DDirW'] = self.data['DDirW'] * np.pi / 180
        
        # 3. Tides and omega_sh joining
        self.data = self.data.join(np.abs(tides-tides.max()), how='inner')
        self.data = self.data.join(data_mean['Omega'], how='inner')
        self.data = self.data.dropna(how='any', axis=0)
        
        # 4. Tidal range value
        self.TR = 3.65 # float(input('\n Select the tidal range (TR): \n'))
        
        # 5. Wave height at break calculation, Hb
        # 5.1 Wave height
        break_th = 0.8 # float(input('\n Select the value for \u03B3: \n'))
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
            indicating the type of profie that wants to be used.
            
            The way this function works is explained in Bernabeu et al. 2003,
            where 2 biparabolic profiles reconstruct the shape of the bottom
            of a requested sandy surfbreak (beachbreak)
        """
            
        if profile_type=='biparabolic':
            slopes = []
            for wave in range(len(self.data)):
                # Range to calculate the slope [m]
                #L = np.sqrt(9.8*waves.iloc[wave].H_break) * waves.iloc[wave].Tp
                #slope_range = L/2
                slope_range = 0.7 #en base a L/2
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
                x_max = 0
                for h_value in h_values:
                    if h_value<hr:
                        xapp = (h_value/A)**(3/2) + (B/(A**(3/2)))*h_value**3
                        x.append(xapp)
                        x_max = max(xapp, x_max)
                    else:
                        Xapp = (h_value/C)**(3/2) + (D/(C**(3/2)))*h_value**3
                        X.append(Xapp)
                        if (h_value-hr)<0.1:
                            try:
                                x_diff = x_max - Xapp
                            except:
                                x_diff = 0
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
            
            
    def plot_data(self, year=2018):
        """
            Plots the profile features for the selected year
        """
        
        # Plot the profile
        register_matplotlib_converters()
        ini = str(year)+'-01-01 00:00:00'
        end = str(year)+'-12-31 23:00:00'
        waves_plot = self.data.loc[ini:end]
        fig, axs = plt.subplots(4, 1, figsize=(20,10), sharex=True)
        fig.subplots_adjust(hspace=0.05, wspace=0.1)
        fig.suptitle('Year: '+str(year)+', '+self.name.upper()+', H$_{S,B}$, T$_P$, $\Omega$, Slope and Iribarren number', 
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
        axs[3].set_xticklabels(months, fontsize=16, fontweight='bold')
        
        
    def moving_profile(self, year=2018):
        """
            A function to see how the profile can variate along a year, with
            its respective wave variables
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
        ax0.legend(fontsize=14)
        ax0.grid()
        ax0.set_xlim(ini, end)
        ax0.set_xticks(np.arange(pd.to_datetime(ini), pd.to_datetime(end), td(days=30.5)))
        ax0.tick_params(direction='in')
        ax0.set_xticklabels([])
        ax1 = fig.add_subplot(gs[1, :])
        ax1.plot(slopes_list['Tp'], color='darkblue', linewidth=1, label='T$_P$ [s]')
        ax1.legend(fontsize=14)
        ax1.grid()
        ax1.set_xlim(ini, end)
        ax1.set_xticks(np.arange(pd.to_datetime(ini), pd.to_datetime(end), td(days=30.5)))
        ax1.tick_params(direction='in')
        ax1.set_xticklabels([])
        ax2 = fig.add_subplot(gs[2, :])
        ax2.plot(slopes_list['Slope'], color='darkgreen', linewidth=1, label='Actual tide slope')
        ax2.legend(fontsize=14)
        ax2.grid()
        ax2.set_xlim(ini, end)
        ax2.set_xticks(np.arange(pd.to_datetime(ini), pd.to_datetime(end), td(days=30.5)))
        ax2.tick_params(direction='in')
        ax2.set_xticklabels([])
        ax3 = fig.add_subplot(gs[3, :])
        ax3.plot(slopes_list['Iribarren'], color='darkred', linewidth=1, label='nº Iribarren')
        ax3.legend(fontsize=14)
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
                                                              'lime',
                                                              'yellow',
                                                              'darkred',
                                                              'black'])
        colors = colors(np.linspace(0, 1, len(slopes_list)))
        
        # observe constants
        Os = []
        As = []
        Bs = []
        Cs = []
        Ds = []
        
        for s in range(len(slopes_list)):
            # Discontinuity point
            hr = 1.1 * slopes_list['Hs'][s] + self.TR
            # Legal point
            # ha = 3 * slopes_list['Hs'][s] + TR
            # Empirical adjusted parameters
            Os.append(slopes_list['Omega'][s])
            A = 0.21 - 0.02*slopes_list['Omega'][s]
            As.append(A)
            B = 0.89 * np.exp(-1.24*slopes_list['Omega'][s])
            Bs.append(B)
            C = 0.06 + 0.04*slopes_list['Omega'][s]
            Cs.append(C)
            D = 0.22 * np.exp(-0.83*slopes_list['Omega'][s])
            Ds.append(D)
            # Different values for the height
            h = np.linspace(0, 10, 150)
            # Important points for the profile
            # xr = (hr/A)**(3/2) + (B/(A**(3/2)))*hr**3
            # x0 = (hr/A)**(3/2) - (hr/C)**(3/2) + \
            #      (B/(A**(3/2)))*hr**3 - (D/(C**(3/2)))*hr**3
            # Lines for the profile
            x  = []
            X  = []
            x_max = 0
            for hs in h:
                if hs<hr:
                    xapp = (hs/A)**(3/2) + (B/(A**(3/2)))*hs**3
                    x.append(xapp)
                    x_max = max(xapp, x_max)
                else:
                    Xapp = (hs/C)**(3/2) + (D/(C**(3/2)))*hs**3
                    if (hs-hr)<0.1:
                        x_diff = x_max - Xapp
                    X.append(Xapp)
                try:
                    x_tot = np.concatenate((np.array(x), 
                                            np.array(X)+x_diff))
                except:
                    if len(x)!=0:
                        x_tot = np.array(x)
                    else:
                        x_tot = np.array(X)
                        x_len = x_tot[-1]-x_tot[0]
            ax3.plot(x_tot, -h, color=colors[s], label=str(slopes_list.index[s]))
            #ax3.scatter(xr, -hr, s=10, c='red', label='Discontinuity point')
            #ax3.axhline(-ha, color='grey', ls='-.', label='Available region')
            #ax3.axhline(0, color='lightgrey', ls='--', label='HTL')
            #ax3.axhline(-TR[tr]/2, color='lightgrey', ls='--', label='MTL')
            #ax3.axhline(-TR[tr], color='lightgrey', ls='--', label='LTL')
            #ax3.axvline(x0, color='k', ls='--', label='Available region')
            ax3.legend(loc='upper right', fontsize=14, ncol=3)
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
            
        print('\n \n')
        print('The values of the profiles plotted are: \n')
        print(pd.DataFrame(data={'A': As, 'B': Bs, 'C': Cs, 'D': Ds,
                                 'Omega': Os},
                           index=slopes_list.index.values))
            
    
    def validate_profile(self, root, omega, diff_sl=3.0):
        """
            With this function, the measured profiles are used to validate
            the theoretical profiles used to calculate the breaking
            
            Parameters
            ----------
            diff_sl : TYPE, float
                DESCRIPTION. The default is 3.0. This parameter moves the
                profile so it can be compared with the theoretical
            y_max : TYPE, float
                DESCRIPTION. The default is 0.0. Maximum height to take into 
                account

            Returns
            ----------
            The mean slope of the measured profile
        """

        # lists to save the data
        profiles_x = []
        profiles_y = []
        
        # plot for the individual profiles
        fig = plt.figure(figsize=(20,8))
        gs  = gridspec.GridSpec(2, 5)
        plt.title(self.name, pad=40, fontsize=20)
        y_pos = [0,0,0,0,0,1,1,1,1,1]
        for rooot, dirs, files in os.walk(os.path.join(root, self.name),
                                          topdown=True):
            for f, file in enumerate(sorted(files)):
                file_plot = np.loadtxt(os.path.join(rooot, file))
                x = file_plot[:,0]
                y = file_plot[:,3] - diff_sl
                zero_idx = np.where((y<0.05) & (y>-0.1))[0][0]
                profiles_x.append(x[zero_idx:]-min(x[zero_idx:]))
                profiles_y.append(y[zero_idx:])
                ax = fig.add_subplot(gs[y_pos[f], f%5])
                ax.plot(x[zero_idx:]-min(x[zero_idx:]), y[zero_idx:])
                ax.plot(x[zero_idx]-min(x[zero_idx:]), y[zero_idx], '.',
                        c='red', markersize=10)
                ax.set_title(file)
                    
        # calculate the mean profile
        len_mean_profile = 0
        for p_x, profile_x in enumerate(profiles_x):
            if (len(profile_x)-len_mean_profile) > 0:
                len_mean_profile = len(profile_x)
                arg_mean_profile = p_x
        height = np.zeros(len_mean_profile)
        for p in range(len(profiles_x)):
            height[:len(profiles_y[p])] += profiles_y[p]
        max_depth_idx = np.argmin(height)
        mean_profile = (profiles_x[arg_mean_profile][:max_depth_idx],
                        height[:max_depth_idx]/len(profiles_x))
        
        # plot final validation
        fig = plt.figure(figsize=(10,4))
        plt.title(self.name, fontsize=20)
        plt.plot(profiles_x[arg_mean_profile][:max_depth_idx], 
                 height[:max_depth_idx]/len(profiles_x),
                 color='red', label='Measured')
        
        # and the theoretical profile used
        hr = 1.1 * 0.5 + self.TR
        # Legal point
        # ha = 3 * slopes_list['Hs'][s] + TR
        # Empirical adjusted parameters
        A = 0.21 - 0.02*omega
        B = 0.89 * np.exp(-1.24*omega)
        C = 0.06 + 0.04*omega
        D = 0.22 * np.exp(-0.83*omega)
        # Different values for the height
        h = np.linspace(0, 4, 150)
        # Important points for the profile
        # xr = (hr/A)**(3/2) + (B/(A**(3/2)))*hr**3
        # x0 = (hr/A)**(3/2) - (hr/C)**(3/2) + \
        #      (B/(A**(3/2)))*hr**3 - (D/(C**(3/2)))*hr**3
        # Lines for the profile
        x  = []
        X  = []
        x_max = 0
        for hs in h:
            if hs<hr:
                xapp = (hs/A)**(3/2) + (B/(A**(3/2)))*hs**3
                x.append(xapp)
                x_max = max(xapp, x_max)
            else:
                Xapp = (hs/C)**(3/2) + (D/(C**(3/2)))*hs**3
                if (hs-hr)<0.1:
                    x_diff = x_max - Xapp
                X.append(Xapp)
            try:
                x_tot = np.concatenate((np.array(x), 
                                        np.array(X)+x_diff))
            except:
                if len(x)!=0:
                    x_tot = np.array(x)
                else:
                    x_tot = np.array(X)
                    x_len = x_tot[-1]-x_tot[0]
        plt.plot(x_tot, -h, color='black', label='Theoretical')
        plt.legend(fontsize=14)
        plt.grid()
        plt.xlim(0, 160)
        plt.ylim(-4, 0)


class Slopes_SOM(object):
    """
        This class performs a clustering of the data using Self-Organizing
        Maps (SOM). The class has many functions, one for the performing
        of the clusterization, and many other functions that plots the 
        obtained data
    """
    
    
    def __init__(self, data, beach=False):
        """
            This "classic" initializator just assign the data to different 
            variables that will be used afterwards
        """
        
        if beach:
            # perform individual beach clusterization
            self.beach      = beach
            self.data       = data.where(data['beach'] == beach).dropna(
                                                                 how='any',
                                                                 axis=0).copy()
            data_som        = data.where(data['beach'] == beach).dropna(
                                                                 how='any',
                                                                 axis=0)\
                              [['H_break', 'Spr', 'Iribarren',
                                'DDir_R', 'DDirW']].copy()
            # data normalization
            data_som        = data_som - np.mean(data_som, axis=0)
            data_som       /= np.std(data_som)
            self.data_som   = data_som.values
        else:
            # total clusterization
            self.beach      = beach
            self.data       = data.copy()
            data_som        = data[['H_break', 'Spr', 'Iribarren',
                                    'DDir_R', 'DDirW']].copy()
            # data normalization
            data_som        = data_som - np.mean(data_som, axis=0)
            data_som       /= np.std(data_som)
            self.data_som   = data_som.values
        
    
    def train(self, som_shape=(20,20), sigma=1.0, learning_rate=0.5,
              num_iteration=50000, plot_results=True):
        """
            Training step to construct the neural network
            ------------
            Parameters
            som_shape: a tuple with the shape of the bidimensional required
                       final topology
            sigma: spread of the neighborhood function, needs to be adequate
                   to the dimensions of the map.
                   (at the iteration t we have sigma(t) = sigma / (1 + t/T)
                    where T is #num_iteration/2)
            learning_rate: initial learning rate
                           (at the iteration t we have
                            learning_rate(t) = learning_rate / (1 + t/T)
                            where T is #num_iteration/2)
            num_iteration: number of iterations that wants to be computed
                           by the net before stopping
            plot_results: boolean for the simple results plotting
            ------------
            Returns
            The trained model and two dataframes with data that could be
            used afterwards
        """
        
        if self.beach:
            print('The following data will be trained in: ' + self.beach)
            print('\n ')
        else:
            print('The following data will be trained: ')
            print('\n ')
            print(self.data.info())
        
        # initialization and training
        som = MiniSom(som_shape[0], som_shape[1], self.data_som.shape[1], 
                      sigma=sigma, learning_rate=learning_rate,
                      neighborhood_function='gaussian', random_seed=10, 
                      topology='hexagonal')
        som.pca_weights_init(self.data_som)
        som.train_batch(self.data_som, 
                        num_iteration=num_iteration, 
                        verbose=False)
        # each neuron represents a cluster
        winner_coord = np.array([som.winner(x) for x in self.data_som]).T
        # with np.ravel_multi_index we convert the bidimensional
        # coordinates to a monodimensional index
        cluster_index = np.ravel_multi_index(winner_coord, som_shape)
        self.data['Cluster'] = cluster_index
        
        # plotting clustering results
        if plot_results:
            # plotting the clusters using the first 2 dimentions of the data
            plt.figure(figsize=(10,10))
            for c in np.unique(cluster_index):
                plt.scatter(self.data_som[cluster_index == c, 0],
                            self.data_som[cluster_index == c,1], 
                            label='cluster='+str(c), alpha=0.3)
            # plotting centroids
            for centroid in som.get_weights():
                plt.scatter(centroid[:, 0], centroid[:, 1], marker='x', 
                            s=80, linewidths=20, color='k')
        
        data_grouped_mean  = self.data.groupby(by='Cluster').mean()
        data_grouped_count = self.data.groupby(by='Cluster').count()
        data_grouped_count = data_grouped_count.mean(axis=1)/len(self.data)
        
        return som, data_grouped_mean, data_grouped_count
    
    
    def plot_results(self, som, data_grouped_mean, data_grouped_count,
                     second_plot='Hb_index',
                     plot_months=False, plot_beaches=False,
                     x=20, y=20):
        """
            With this plot, the data can be easily plotted in the
            bidimensional grid, with all the variables existent available.
            
            x and y must be added, but this fug will be fixed
        """        
        
        # edit the angles to plot
        data_grouped_mean['DDir_R'] = - data_grouped_mean['DDir_R'] + np.pi/2
        data_grouped_mean['DDirW']  = - data_grouped_mean['DDirW'] + np.pi/2
            
        # plot the stuff
        if self.beach:
            
            f = plt.figure(figsize=(20, 20))
            f.suptitle('SOM bidimensional grids and results in: '+self.beach.upper(),
                       y=0.9, fontsize=22)
            
            ax = f.add_subplot(221)
            
            ax.set_aspect('equal')
            
            xx, yy = som.get_euclidean_coordinates()
            umatrix = som.distance_map()
            weights = som.get_weights()
            
            hb = np.zeros(x * y)
            hb[data_grouped_mean.index.values] = data_grouped_mean['H_break'].values
            hb = hb.reshape(x, y)
            cm1 = matplotlib.cm.bwr
            norm1 = Normalize(vmin=0, vmax=5)
            sm1 = matplotlib.cm.ScalarMappable(cmap=cm1, norm=norm1)
            sm1.set_array([])
            
            ir = np.zeros(x * y)
            ir[data_grouped_mean.index.values] = data_grouped_mean['Iribarren'].values
            ir = ir.reshape(x, y)
            cm2 = matplotlib.cm.hot
            norm2 = Normalize(vmin=0, vmax=3)
            sm2 = matplotlib.cm.ScalarMappable(cmap=cm2, norm=norm2)
            sm2.set_array([])   
            
            cb1 = f.colorbar(sm1, cax=f.add_axes([0.50,0.56,0.02,0.30]), 
                             orientation='vertical', alpha=.4)
            cb1.ax.get_yaxis().labelpad = -18
            cb1.ax.set_ylabel('H$_B$ [m]', rotation=270, fontsize=16)
            
            # cb2 = f.colorbar(sm2, cax=f.add_axes([0.955,0.18,0.02,0.65]), 
            #                  orientation='vertical', alpha=.4)
            # cb2.ax.get_yaxis().labelpad = -28
            # cb2.ax.set_ylabel('nº Iribarren', rotation=270, fontsize=16)
            
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy = yy[(i, j)]*2/np.sqrt(3)*3/4
                    hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, 
                                         radius=.95/np.sqrt(3),
                                         facecolor=cm1(norm1(hb[i,j])), 
                                         alpha=.4, edgecolor='gray')
                    ax.add_patch(hex)
                    hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, 
                                         radius=.35/np.sqrt(3),
                                         facecolor=cm2(norm2(ir[i,j])), 
                                         alpha=.6, edgecolor='gray')
                    ax.add_patch(hex)
                
            u = np.zeros(x * y)
            v = np.zeros(x * y)
            vel = np.zeros(x * y)
            u[data_grouped_mean.index.values] = - np.cos(data_grouped_mean['DDir_R'].values)
            v[data_grouped_mean.index.values] = - np.sin(data_grouped_mean['DDir_R'].values)
            vel[data_grouped_mean.index.values] = data_grouped_mean['Spr'].values
            u = u.reshape(x, y)
            v = v.reshape(x, y)
            cmm = matplotlib.cm.jet
            norm = Normalize(vmin=0, vmax=50)
            sm = matplotlib.cm.ScalarMappable(cmap=cmm, norm=norm)
            sm.set_array([])
            yy = yy*2/np.sqrt(3)*3/4
            Q = ax.quiver(xx, yy, u, v, color=cmm(norm(vel)))
            
            u = np.zeros(x * y)
            v = np.zeros(x * y)
            u[data_grouped_mean.index.values] = - np.cos(data_grouped_mean['DDirW'].values) * \
                data_grouped_mean['W'].values
            v[data_grouped_mean.index.values] = - np.sin(data_grouped_mean['DDirW'].values) * \
                data_grouped_mean['W'].values
            u = u.reshape(x, y)
            v = v.reshape(x, y)
            Q = ax.quiver(xx, yy, u, v, color='k')
            ax.quiverkey(Q, X=0.82, Y=0.98, U=10,
                         label='Wind speed', labelpos='E')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-1.6, 20)
            ax.set_ylim(-1.5, 18.2)
            
            ax = f.add_subplot(222)
            
            ax.set_aspect('equal')
                
            xx, yy = som.get_euclidean_coordinates()
            umatrix = som.distance_map()
            weights = som.get_weights()
                
            prob = np.zeros(x * y)
            prob[data_grouped_count.index.values] = data_grouped_count.values
            print('The sum off all probabilities is: '+str(np.sum(prob)))
            prob = prob.reshape(x, y)
                
            norm = LogNorm(vmin=0.001, vmax=0.01)
            cm = matplotlib.cm.jet
            cm.set_bad(color='red')
            sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
            sm.set_array([])
                
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy = yy[(i, j)]*2/np.sqrt(3)*3/4
                    hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                                         facecolor=cm(norm(prob[i,j])), alpha=.4, 
                                         edgecolor='gray', lw=2.5)
                    ax.add_patch(hex)
                        
            cb = f.colorbar(sm, cax=f.add_axes([0.93,0.56,0.02,0.30]), 
                            orientation='vertical', alpha=.4)
            cb.ax.get_yaxis().labelpad = -56
            cb.ax.set_ylabel('Probability · 100 (%)', rotation=270, fontsize=16)
                        
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-1.6, 20)
            ax.set_ylim(-1.5, 18.2)
            
            ax = f.add_subplot(223)
            
            ax.set_aspect('equal')
                
            xx, yy = som.get_euclidean_coordinates()
            umatrix = som.distance_map()
            weights = som.get_weights()
                
            index = np.zeros(x * y)
            index[data_grouped_mean.index.values] = data_grouped_mean['Index'].values
            index = index.reshape(x, y)
                
            norm = Normalize(vmin=0, vmax=10)
            cm = matplotlib.cm.YlGnBu
            cm_manual = LinearSegmentedColormap.from_list('mycmap', 
                        ['blue', 'green', 'yellow', 'orange', 'red', 'purple', 'black'])
            cm.set_bad(color='red')
            cm_manual.set_bad(color='white')
            sm = matplotlib.cm.ScalarMappable(cmap=cm_manual, norm=norm)
            sm.set_array([])
                
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy = yy[(i, j)]*2/np.sqrt(3)*3/4
                    hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                                         facecolor=cm_manual(norm(index[i,j])), alpha=.4, 
                                         edgecolor='gray', lw=2.5)
                    ax.add_patch(hex)
                        
            cb = f.colorbar(sm, cax=f.add_axes([0.50,0.14,0.02,0.30]), 
                            orientation='vertical', alpha=.4)
            cb.ax.get_yaxis().labelpad = -31
            cb.ax.set_ylabel('Index', rotation=270, fontsize=16)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-1.6, 20)
            ax.set_ylim(-1.5, 18.2)
            
            ax = f.add_subplot(224)
            
            ax.set_aspect('equal')
                
            xx, yy = som.get_euclidean_coordinates()
            umatrix = som.distance_map()
            weights = som.get_weights()
                
            index = np.zeros(x * y)
            index[data_grouped_mean.index.values] = data_grouped_mean[second_plot].values * 10
            index = index.reshape(x, y)
                
            norm = Normalize(vmin=0, vmax=10)
            cm = matplotlib.cm.YlGnBu
            cm_manual = LinearSegmentedColormap.from_list('mycmap', 
                        ['blue', 'green', 'yellow', 'orange', 'red', 'purple', 'black'])
            cm.set_bad(color='red')
            cm_manual.set_bad(color='white')
            sm = matplotlib.cm.ScalarMappable(cmap=cm_manual, norm=norm)
            sm.set_array([])
                
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy = yy[(i, j)]*2/np.sqrt(3)*3/4
                    hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                                         facecolor=cm_manual(norm(index[i,j])), alpha=.4, 
                                         edgecolor='gray', lw=2.5)
                    ax.add_patch(hex)
                        
            cb = f.colorbar(sm, cax=f.add_axes([0.93,0.14,0.02,0.30]), 
                            orientation='vertical', alpha=.4)
            cb.ax.get_yaxis().labelpad = -31
            cb.ax.set_ylabel(second_plot, rotation=270, fontsize=16)
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-1.6, 20)
            ax.set_ylim(-1.5, 18.2)
            
            if plot_months:
                months = np.arange(1, 13)
                month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December']
                
                fig, axs = plt.subplots(3, 4, figsize=(20,15))
                plt.subplots_adjust(hspace=0.05, wspace=0.05)
                xx, yy = som.get_euclidean_coordinates()
                umatrix = som.distance_map()
                weights = som.get_weights()
                
                norm = LogNorm(vmin=0.001, vmax=0.1)
                cm = matplotlib.cm.jet
                cm.set_bad(color='white')
                sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
                sm.set_array([])
                
                waves_grouped_month_count = self.data.groupby(by=['Month', 'Cluster'])\
                .count().mean(axis=1)/(len(self.data)/12)
                
                for i, j in [(a, b) for a in range(3) for b in range(4)]:
                    prob = np.zeros(x * y)
                    prob[waves_grouped_month_count.index.get_level_values(1)[waves_grouped_month_count.index.get_level_values(0)==months[j+i*4]]] = \
                    waves_grouped_month_count[waves_grouped_month_count.index.get_level_values(0)==months[j+i*4]]
                    prob = prob.reshape(x, y)
                    for ii in range(weights.shape[0]):
                        for jj in range(weights.shape[1]):
                            wy = yy[(ii, jj)]*2/np.sqrt(3)*3/4
                            hex = RegularPolygon((xx[(ii, jj)], wy), numVertices=6, radius=.95/np.sqrt(3),
                                                  facecolor=cm(norm(prob[ii,jj])), alpha=.4, 
                                                  edgecolor='gray', lw=2.5)
                            axs[i,j].add_patch(hex)
                    axs[i,j].set_aspect('equal')
                    axs[i,j].set_xlim(-1.6, 20)
                    axs[i,j].set_ylim(-1.5, 18.2)
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])
                    axs[i,j].set_title(month_names[j+i*4])
                            
                cb = fig.colorbar(sm, cax=fig.add_axes([0.95,0.20,0.03,0.60]), 
                                  orientation='vertical', alpha=.4)
                cb.ax.get_yaxis().labelpad = -46
                cb.ax.set_ylabel('Probability · 100 (%)', rotation=270, fontsize=18)
            
        else:
            
            f = plt.figure(figsize=(15, 15))
            
            ax = f.add_subplot(111)
            
            ax.set_aspect('equal')
            
            xx, yy = som.get_euclidean_coordinates()
            umatrix = som.distance_map()
            weights = som.get_weights()
            
            hb = np.zeros(x * y)
            hb[data_grouped_mean.index.values] = data_grouped_mean['H_break'].values
            hb = hb.reshape(x, y)
            cm1 = matplotlib.cm.bwr
            norm1 = Normalize(vmin=0, vmax=5)
            sm1 = matplotlib.cm.ScalarMappable(cmap=cm1, norm=norm1)
            sm1.set_array([])
            
            ir = np.zeros(x * y)
            ir[data_grouped_mean.index.values] = data_grouped_mean['Iribarren'].values
            ir = ir.reshape(x, y)
            cm2 = matplotlib.cm.hot
            norm2 = Normalize(vmin=0, vmax=3)
            sm2 = matplotlib.cm.ScalarMappable(cmap=cm2, norm=norm2)
            sm2.set_array([])   
            
            cb1 = f.colorbar(sm1, cax=f.add_axes([0.91,0.18,0.02,0.65]), 
                             orientation='vertical', alpha=.4)
            cb1.ax.get_yaxis().labelpad = -16
            cb1.ax.set_ylabel('H$_B$ [m]', rotation=270, fontsize=16)
            
            cb2 = f.colorbar(sm2, cax=f.add_axes([0.955,0.18,0.02,0.65]), 
                             orientation='vertical', alpha=.4)
            cb2.ax.get_yaxis().labelpad = -28
            cb2.ax.set_ylabel('nº Iribarren', rotation=270, fontsize=16)
            
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy = yy[(i, j)]*2/np.sqrt(3)*3/4
                    hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, 
                                         radius=.95/np.sqrt(3),
                                         facecolor=cm1(norm1(hb[i,j])), 
                                         alpha=.4, edgecolor='gray')
                    ax.add_patch(hex)
                    hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, 
                                         radius=.35/np.sqrt(3),
                                         facecolor=cm2(norm2(ir[i,j])), 
                                         alpha=.6, edgecolor='gray')
                    ax.add_patch(hex)
                
            u = np.zeros(x * y)
            v = np.zeros(x * y)
            vel = np.zeros(x * y)
            u[data_grouped_mean.index.values] = - np.cos(data_grouped_mean['DDir_R'].values)
            v[data_grouped_mean.index.values] = - np.sin(data_grouped_mean['DDir_R'].values)
            vel[data_grouped_mean.index.values] = data_grouped_mean['Spr'].values
            u = u.reshape(x, y)
            v = v.reshape(x, y)
            cmm = matplotlib.cm.jet
            norm = Normalize(vmin=0, vmax=50)
            sm = matplotlib.cm.ScalarMappable(cmap=cmm, norm=norm)
            sm.set_array([])
            yy = yy*2/np.sqrt(3)*3/4
            Q = ax.quiver(xx, yy, u, v, color=cmm(norm(vel)))
            cb3 = f.colorbar(sm, cax=f.add_axes([1.0,0.18,0.02,0.65]), 
                             orientation='vertical', alpha=.4)
            cb3.ax.get_yaxis().labelpad = -22
            cb3.ax.set_ylabel('$\sigma_{\Theta}$ [$\degree$]', 
                              rotation=270, fontsize=16)
            
            u = np.zeros(x * y)
            v = np.zeros(x * y)
            u[data_grouped_mean.index.values] = - np.cos(data_grouped_mean['DDirW'].values) * \
                data_grouped_mean['W'].values
            v[data_grouped_mean.index.values] = - np.sin(data_grouped_mean['DDirW'].values) * \
                data_grouped_mean['W'].values
            u = u.reshape(x, y)
            v = v.reshape(x, y)
            Q = ax.quiver(xx, yy, u, v, color='k')
            ax.quiverkey(Q, X=0.9, Y=0.98, U=10,
                         label='Wind speed', labelpos='E')
            
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-1.6, 20)
            ax.set_ylim(-1.5, 18.2)
            
            ff = plt.figure(figsize=(15, 15))
            
            ax = ff.add_subplot(111)
            
            ax.set_aspect('equal')
            
            xx, yy = som.get_euclidean_coordinates()
            umatrix = som.distance_map()
            weights = som.get_weights()
            
            index = np.zeros(x * y)
            index[data_grouped_mean.index.values] = data_grouped_mean['Index'].values
            index = index.reshape(x, y)
            cm1 = LinearSegmentedColormap.from_list('mycmap', 
                  ['blue', 'green', 'yellow', 'orange', 'red', 'purple', 'black'])
            norm1 = Normalize(vmin=0, vmax=10)
            sm1 = matplotlib.cm.ScalarMappable(cmap=cm1, norm=norm1)
            sm1.set_array([])
            
            cb1 = ff.colorbar(sm1, cax=ff.add_axes([0.92,0.18,0.04,0.65]), 
                              orientation='vertical', alpha=.4)
            cb1.ax.get_yaxis().labelpad = -32
            cb1.ax.set_ylabel('Index (from 0 to 10)', rotation=270, fontsize=18)
            
            for i in range(weights.shape[0]):
                for j in range(weights.shape[1]):
                    wy = yy[(i, j)]*2/np.sqrt(3)*3/4
                    hex = RegularPolygon((xx[(i, j)], wy), numVertices=6, 
                                         radius=.95/np.sqrt(3),
                                         facecolor=cm1(norm1(index[i,j])), 
                                         alpha=.4, edgecolor='gray')
                    ax.add_patch(hex)
                    
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(-1.6, 20)
            ax.set_ylim(-1.5, 18.2)
            
            if plot_beaches:
                beach_names = ['farolillo', 'oyambre', 'locos', 'valdearenas', 
                               'segunda', 'curva', 'brusco', 'forta']

                fig, axs = plt.subplots(2, 4, figsize=(20,10))
                plt.subplots_adjust(hspace=0.05, wspace=0.05)
                xx, yy = som.get_euclidean_coordinates()
                umatrix = som.distance_map()
                weights = som.get_weights()
                
                norm = LogNorm(vmin=0.001, vmax=0.1)
                cm = matplotlib.cm.jet
                cm.set_bad(color='white')
                sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
                sm.set_array([])
                
                waves_grouped_beach_count = self.data.groupby(by=['beach', 'Cluster'])\
                .count().mean(axis=1)/(len(self.data)/len(self.data.beach.unique()))
                
                for i, j in [(a, b) for a in range(2) for b in range(4)]:
                    prob = np.zeros(x * y)
                    prob[waves_grouped_beach_count.index.get_level_values(1)[waves_grouped_beach_count.index.get_level_values(0)==beach_names[j+i*4]]] = \
                    waves_grouped_beach_count[waves_grouped_beach_count.index.get_level_values(0)==beach_names[j+i*4]]
                    prob = prob.reshape(x, y)
                    for ii in range(weights.shape[0]):
                        for jj in range(weights.shape[1]):
                            wy = yy[(ii, jj)]*2/np.sqrt(3)*3/4
                            hex = RegularPolygon((xx[(ii, jj)], wy), numVertices=6, radius=.95/np.sqrt(3),
                                                  facecolor=cm(norm(prob[ii,jj])), alpha=.4, 
                                                  edgecolor='gray', lw=2.5)
                            axs[i,j].add_patch(hex)
                    axs[i,j].set_aspect('equal')
                    axs[i,j].set_xlim(-1.6, 20)
                    axs[i,j].set_ylim(-1.5, 18.2)
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])
                    axs[i,j].set_title(beach_names[j+i*4])
                            
                cb = fig.colorbar(sm, cax=fig.add_axes([0.95,0.20,0.03,0.60]), 
                                  orientation='vertical', alpha=.4)
                cb.ax.get_yaxis().labelpad = -46
                cb.ax.set_ylabel('Probability · 100 (%)', rotation=270, fontsize=18)
                
            if plot_months:
                months = np.arange(1, 13)
                month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December']
                
                fig, axs = plt.subplots(3, 4, figsize=(20,15))
                plt.subplots_adjust(hspace=0.05, wspace=0.05)
                xx, yy = som.get_euclidean_coordinates()
                umatrix = som.distance_map()
                weights = som.get_weights()
                
                norm = LogNorm(vmin=0.001, vmax=0.1)
                cm = matplotlib.cm.jet
                cm.set_bad(color='white')
                sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
                sm.set_array([])
                
                waves_grouped_month_count = self.data.groupby(by=['Month', 'Cluster'])\
                .count().mean(axis=1)/(len(self.data)/12)
                
                for i, j in [(a, b) for a in range(3) for b in range(4)]:
                    prob = np.zeros(x * y)
                    prob[waves_grouped_month_count.index.get_level_values(1)[waves_grouped_month_count.index.get_level_values(0)==months[j+i*4]]] = \
                    waves_grouped_month_count[waves_grouped_month_count.index.get_level_values(0)==months[j+i*4]]
                    prob = prob.reshape(x, y)
                    for ii in range(weights.shape[0]):
                        for jj in range(weights.shape[1]):
                            wy = yy[(ii, jj)]*2/np.sqrt(3)*3/4
                            hex = RegularPolygon((xx[(ii, jj)], wy), numVertices=6, radius=.95/np.sqrt(3),
                                                  facecolor=cm(norm(prob[ii,jj])), alpha=.4, 
                                                  edgecolor='gray', lw=2.5)
                            axs[i,j].add_patch(hex)
                    axs[i,j].set_aspect('equal')
                    axs[i,j].set_xlim(-1.6, 20)
                    axs[i,j].set_ylim(-1.5, 18.2)
                    axs[i,j].set_xticks([])
                    axs[i,j].set_yticks([])
                    axs[i,j].set_title(month_names[j+i*4])
                            
                cb = fig.colorbar(sm, cax=fig.add_axes([0.95,0.20,0.03,0.60]), 
                                  orientation='vertical', alpha=.4)
                cb.ax.get_yaxis().labelpad = -46
                cb.ax.set_ylabel('Probability · 100 (%)', rotation=270, fontsize=18)
                    
