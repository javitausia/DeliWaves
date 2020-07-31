# basic
import sys
import os
import os.path as op

# data libraries
import netCDF4
import numpy as np
import pandas as pd
import xarray as xr
import datetime

# plots
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from mpl_toolkits.basemap import Basemap
import imageio
from termcolor import colored

# warnings
import warnings
warnings.filterwarnings('ignore')

# custom plots
from ipyleaflet import *

# dev library
sys.path.insert(0, op.join(os.getcwd(), '..'))
# RBF module
from rbf.rbf_main import RBF_Reconstruction

# Forecast class
class Forecast(object):
    """ This forecast class has several functions that allows the user
        to correctly predict the waves and the surfing conditions with
        precision in that place where the SWAN propagations have been
        done. These SWAN propagations can be performed using the attached
        notebook and scripts, but in case the propagations are not wanted
        to be done, the global prediction in offshore points is also
        proportioned with this tool. Refer to the jupyter notebook for
        more important information, also in the repository.
    """
    
    def __init__(self, date, images_path, location):
        """ Initializes the class with all the necessary attributes that
            will be used in the different methods
            ------------
            Parameters
            date: date to initialize the forecast in format %YYYYmmdd%
            images_path: path to save the images and GIF
            location: location to obtain the forecast
            ------------
            Returns
            The initialized attributes and a GIF in path with the 
            global forecast
        """
        
        print('Pulling the data from: \n')
        url = 'https://nomads.ncep.noaa.gov:9090/dods/wave/mww3/'+date+'/multi_1.glo_30mext'+date+'_00z'
        print(url)
        print('\n')
        
        # Initialization
        self.forecast        =   netCDF4.Dataset(url)
        self.images_path     =   images_path
        self.location        =   location
        self.coast_location  =   (0, 0) # will be filled after
        
        # Changing times from Gregorian to datetime
        times = [datetime.datetime.fromordinal(int(gtime)) + \
                 datetime.timedelta(days=gtime%1) \
                 for gtime in self.forecast.variables['time'][:].data]
        print('The times with forecast go from {} to {} \n'.format(times[0], 
                                                                   times[-1]))
        # Saving corrected times as an attribute
        self.times = times
        
        # GIF generator
        print('Generating images and GIF in "path"... \n')
        fig = plt.figure(figsize=(20,15))
        m = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, 
                                      llcrnrlon=0,   urcrnrlon=360,
                    resolution='l')
        filenames = []
        # Longitude and latitude values
        lat    = self.forecast.variables['lat'][:]
        lon    = self.forecast.variables['lon'][:]
        x, y   = m(*np.meshgrid(lon, lat))
        xx     = np.arange(0, len(lon), 3)
        yy     = np.arange(0, len(lat), 3)
        points = np.meshgrid(yy, xx)
        for t in range(0, len(times), 8):
            if t>=1:
                hs.remove()
                dirr.remove()
            print('Plotting time: {}...'.format(times[t]))
            hs    = m.pcolormesh(x, y, self.forecast.variables['htsgwsfc'][t,:,:], 
                                 shading='flat', cmap=plt.cm.jet)
            tp    = self.forecast.variables['perpwsfc'][t,:,:]
            direc = self.forecast.variables['dirpwsfc'][t,:,:]
            U     = tp * np.sin((360-direc)*np.pi/180)
            V     = tp * np.cos((360-direc)*np.pi/180)
            dirr  = m.quiver(x[points], y[points], U[points], V[points])
            if t==0:
                #m.colorbar(location='right')
                m.drawcoastlines()
                m.fillcontinents(color='lightgrey', lake_color='aqua')
                m.drawmapboundary(fill_color='navy')
            plt.title(times[t], fontsize=18, fontweight='bold')
            fig.savefig(op.join(self.images_path, 
                                '{}.png'.format(times[t])))
            filenames.append('{}.png'.format(times[t]))
        # GIF
        images = []
        for filename in filenames:
            images.append(imageio.imread(op.join(self.images_path, filename)))
        imageio.mimsave(op.join(self.images_path, 'forecast.gif'), 
                        images, duration = 1.0)
        print(colored('\n GIF generated and saved!! \n', 'red', 
                      attrs=['blink']))
        print('\n')
        
        
    def select_precise_location(self):
        """ This interactive plot helps the user choose the precise
            location where the forecast wanna be done
            ------------
            Parameters (self)
            ------------
            Returns
            The precise location for the forecast
        """
            
        print('Move the marker to the exact position: \n')
        m = Map(basemap=basemaps.Esri.WorldImagery, 
                center=self.location, zoom=2)
        marker = Marker(icon=AwesomeIcon(name='check', 
                                         marker_color='green', 
                                         icon_color='darkgreen'))
        m.add_control(SearchControl(
                position='topleft',
                url='https://nominatim.openstreetmap.org/search?format=json&q={s}',
                zoom=5,
                marker=marker
                ))
        selected_location = Marker(location=self.location, 
                                   draggable=True)
        m.add_layer(selected_location)
        display(m)
        
        return selected_location
            
        
    def select_region(self, marker, delta_lon, delta_lat, zoom=6):
        """ This plot helps the user see the region that will be saved
            so more than one single forecast node will be available
            ------------
            Parameters
            marker: output from select_precise_location
            delta_lon: longitude delta distance in degrees
            delta_lat: latitude delta distance in degrees
            zoom: zoom to see the plot
            ------------
            Returns
            the data saved as an xarray dataset and a pandas dataframe to
            plot the results easily
        """
        
        # Relocate the precise location selected previously
        self.location = marker.location
        print(colored('New location in {}!! \n'.format(self.location),
                      'red', attrs=['blink']))
        lat = self.location[0]
        if self.location[1]>0:
            lon = self.location[1]
        else:
            lon = self.location[1]+360
        lat_index = np.where((self.forecast.variables['lat'][:].data < (lat+delta_lat)) & 
                             (self.forecast.variables['lat'][:].data > (lat-delta_lat)))
        lon_index = np.where((self.forecast.variables['lon'][:].data < (lon+delta_lon)) & 
                             (self.forecast.variables['lon'][:].data > (lon-delta_lon)))
        
        print('These are the coordinates in the selected region: \n')
        print(self.forecast.variables['lat'][:][list(lat_index[0])])
        print(self.forecast.variables['lon'][:][list(lon_index[0])])
        print('\n')
        
        # Map plotting to see the downloading data
        m = Map(basemap=basemaps.Esri.WorldImagery, 
                center=(lat,lon), zoom=zoom)
        rectangle = Rectangle(bounds=((lat-delta_lat, lon-delta_lon), 
                                      (lat+delta_lat, lon+delta_lon)),
                              color='red', opacity=0.1)
        m.add_layer(rectangle)
        marker = Marker(location=(lat,lon))
        m.add_layer(marker)
        display(m)
            
        print('Saving the data in the shown region... \n')
        data = xr.Dataset({'Hsea'       : (['time', 'lat', 'lon'],
                                           self.forecast.variables['wvhgtsfc'][:,lat_index[0],lon_index[0]].data),
                           'Tpsea'      : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['wvpersfc'][:,lat_index[0],lon_index[0]].data),
                           'Dirsea'     : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['wvdirsfc'][:,lat_index[0],lon_index[0]].data),
                           'Hswell1'    : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['swell_1'][:,lat_index[0],lon_index[0]].data),
                           'Tpswell1'   : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['swper_1'][:,lat_index[0],lon_index[0]].data),
                           'Dirswell1'  : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['swdir_1'][:,lat_index[0],lon_index[0]].data),
                           'Hswell2'    : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['swell_2'][:,lat_index[0],lon_index[0]].data),
                           'Tpswell2'   : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['swper_2'][:,lat_index[0],lon_index[0]].data),
                           'Dirswell2'  : (['time', 'lat', 'lon'],                                             
                                           self.forecast.variables['swdir_2'][:,lat_index[0],lon_index[0]].data),
                           'Hs'         : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['htsgwsfc'][:,lat_index[0],lon_index[0]].data),
                           'Tp'         : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['perpwsfc'][:,lat_index[0],lon_index[0]].data),
                           'Dir'        : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['dirpwsfc'][:,lat_index[0],lon_index[0]].data),
                           'Uwind'      : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['ugrdsfc'][:,lat_index[0],lon_index[0]].data),
                           'Vwind'      : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['vgrdsfc'][:,lat_index[0],lon_index[0]].data),
                           'WindSpeed'  : (['time', 'lat', 'lon'], 
                                           self.forecast.variables['windsfc'][:,lat_index[0],lon_index[0]].data)},
                          coords = {'time' : self.times,
                                    'lat'  : self.forecast.variables['lat'][:][lat_index[0]],
                                    'lon'  : self.forecast.variables['lon'][:][lon_index[0]]})
        print(data)
        
        # Saving the dataframe with the nearest point
        lat = self.location[0]
        if self.location[1]>0:
            lon = self.location[1]
        else:
            lon = self.location[1]+360
        ilat = np.where((data.lat.values < lat+0.3) & 
                        (data.lat.values > lat-0.3))[0][0]
        ilon = np.where((data.lon.values < lon+0.3) &
                        (data.lon.values > lon-0.3))[0][0]
        data_dataframe = data.isel(lat=ilat).isel(lon=ilon).to_dataframe()
        data_dataframe = data_dataframe.where(data_dataframe<1000, 0)
            
        return data, data_dataframe
    
            
    def forecast_reconstruction(self, p_data_swan, forecast_data, name, resolution, num_cases):
        """ This method reconstruct the forecast information in the
            previously selected region
            ------------
            Parameters
            p_data_swan: path to find the necessary data to reconstruct (SWAN)
            forecast_data: data in the selected region (dataframe)
            name: name of the region in folder
            resolution: resolution of the region in folder
            num_cases: num_cases of the region in folder
            ------------
            Returns
            the reconstructed forecast in the chose point as a dataframe
        """
        
        # SUBSETS
        subsetsea    = pd.read_pickle(op.join(p_data_swan, name+'-SEA-'+resolution,
                                              'sea_cases_'+num_cases+'.pkl'))
        subsetsea    = subsetsea[['hs', 'per', 'dir']]
        subsetswell  = pd.read_pickle(op.join(p_data_swan, name+'-SWELL-'+resolution,
                                              'swell_cases_'+num_cases+'.pkl'))
        subsetswell  = subsetswell[['hs', 'per', 'dir']]
        print(colored('SUBSETS: \n', 'blue', attrs=['blink', 'reverse']))
        print(colored('SEA', 'red', attrs=['blink']))
        print(subsetsea.info())
        print(colored('SWELL', 'red', attrs=['blink']))
        print(subsetswell.info())
        print('\n')
        
        # TARGETS
        targetsea    = xr.open_dataset(op.join(p_data_swan, name+'-SEA-'+resolution,
                                               'sea_propagated_'+num_cases+'.nc'))
        targetswell  = xr.open_dataset(op.join(p_data_swan, name+'-SWELL-'+resolution,
                                               'swell_propagated_'+num_cases+'.nc'))
        # Selection of the desired point
        print(colored('Select the desired point to reconstruct as it is given in Google Maps: \n',
                      'blue', attrs=['blink', 'reverse']))
        latT = float(input('Latitude location to obtain the forecast reconstruction: '))
        lonT = float(input('Longitude location to obtain the forecast reconstruction: \n'))
        # Reinitialize the attribute in coast
        self.coast_location = (latT, lonT)
        print('\n')
        ilat = np.where((targetsea.Y.values < latT+0.005) & 
                        (targetsea.Y.values > latT-0.005))[0][0]
        ilon = np.where((targetswell.X.values < lonT+0.005) &
                        (targetswell.X.values > lonT-0.005))[0][0]
        targetsea   = targetsea.isel(X=ilon).isel(Y=ilat)
        targetswell = targetswell.isel(X=ilon).isel(Y=ilat)
        targetsea   = pd.DataFrame({'hs': targetsea.Hsig.values,
                                    'per': targetsea.TPsmoo.values,
                                    'perM': targetsea.Tm02.values,
                                    'dir': targetsea.Dir.values,
                                    'spr': targetsea.Dspr.values})
        seaedit         = subsetsea.mean()
        seaedit['perM'] = 7.0
        seaedit['spr']  = 22.0
        targetsea       = targetsea.fillna(seaedit)
        targetswell = pd.DataFrame({'hs': targetswell.Hsig.values,
                                    'per': targetswell.TPsmoo.values,
                                    'perM': targetswell.Tm02.values,
                                    'dir': targetswell.Dir.values,
                                    'spr': targetswell.Dspr.values})
        swelledit         = subsetswell.mean()
        swelledit['perM'] = 12.0
        swelledit['spr']  = 12.0
        targetswell       = targetswell.fillna(swelledit)
        print(colored('TARGETS: \n', 'blue', attrs=['blink', 'reverse']))
        print(colored('SEA', 'red', attrs=['blink']))
        print(targetsea.info())
        print(colored('SWELL', 'red', attrs=['blink']))
        print(targetswell.info())
        print('\n')
            
        # DATASETS
        print(colored('Forecast in the selected region has the shape: \n',
                      'blue', attrs=['blink', 'reverse']))
        print(forecast_data.info())
        print('\n')
            
        # Preprocess the data
        labels_input   = [['Hsea', 'Tpsea', 'Dirsea'],
                          ['Hswell1', 'Tpswell1','Dirswell1'],
                          ['Hswell2', 'Tpswell2','Dirswell2']]
        labels_output  = [['Hsea', 'Tpsea', 'Tm_02', 'Dirsea', 'Sprsea'],
                          ['Hswell1', 'Tpswell1', 'Tm_02','Dirswell1', 'Sprswell1'],
                          ['Hswell2', 'Tpswell2', 'Tm_02','Dirswell2', 'Sprswell2']]
        # Initialize the datasets to reconstruct
        datasets = []
        for ss in labels_input:
            dataset_ss = forecast_data[ss]
            dataset_ss = dataset_ss.dropna(axis=0, how='any')
            datasets.append(dataset_ss) 
        # Initialize the dataframes to save
        dataframes = []

        print('Performing RFB reconstruction... \n')
        # RBF             
        for count, dat in enumerate(datasets):
            # Scalar and directional columns
            ix_scalar_subset = [0,1]
            ix_directional_subset = [2]
            ix_scalar_target = [0,1,2,4]
            ix_directional_target = [3] 
            # RBF for the seas
            if count==0:
                # Calculating subset, target and dataset
                subset  = subsetsea.to_numpy()
                target  = targetsea.to_numpy()
                dat_index = dat.index
                dataset = dat.to_numpy()
                # Performing RBF
                output = RBF_Reconstruction(
                            subset, ix_scalar_subset, ix_directional_subset,
                            target, ix_scalar_target, ix_directional_target,
                            dataset
                            )
                # Reconstrucing the new dataframe
                for l, lab in enumerate(labels_output[count]):
                    if l==0:
                        output_dataframe = pd.DataFrame({lab: output[:,l]}, 
                                                         index=dat_index)
                    else:
                        output_dataframe[lab] = output[:,l]
                # Appending all new dataframes    
                dataframes.append(output_dataframe)
            # RBF for the swellls    
            else:
                # Calculating subset, target and dataset
                subset  = subsetswell.to_numpy()
                target  = targetswell.to_numpy()
                dat_index = dat.index
                dataset = dat.to_numpy()
                # Performing RBF
                output = RBF_Reconstruction(
                            subset, ix_scalar_subset, ix_directional_subset,
                            target, ix_scalar_target, ix_directional_target,
                            dataset
                            )
                # Reconstrucing the new dataframe
                for l, lab in enumerate(labels_output[count]):
                    if l==0:
                        output_dataframe = pd.DataFrame({lab: output[:,l]}, 
                                                         index=dat_index)
                    else:   
                        output_dataframe[lab] = output[:,l]
                # Appending all new dataframes        
                dataframes.append(output_dataframe)

        # SAVE final file
        reconstructed_dataframe = pd.concat(dataframes, axis=1)
        forecast_data_red = forecast_data[['Hsea', 'Tpsea', 'Dirsea',
                                           'Hswell1', 'Tpswell1', 'Dirswell1',
                                           'Hswell2', 'Tpswell2', 'Dirswell2']]
        forecast = reconstructed_dataframe[['Hsea', 'Tpsea', 'Dirsea',
                                            'Hswell1', 'Tpswell1', 'Dirswell1',
                                            'Hswell2', 'Tpswell2', 'Dirswell2']].where(forecast_data_red>0.01, 
                                                                                       forecast_data_red)
        
        # BULK PARAMETERS
        # First copy to play with NaNs
        agg = forecast.copy()
        tp = agg[['Tpsea', 'Tpswell1', 'Tpswell2']].copy()
        tp = tp.where(tp>0.01, np.inf)
        # Bulk Hs
        forecast['Hs'] = np.sqrt(agg['Hsea']**2 +
                                 agg['Hswell1']**2 +
                                 agg['Hswell2']**2)
        # Bulk Tp
        forecast['Tp'] = np.sqrt(
                forecast['Hs']**2 / (agg['Hsea']**2/tp['Tpsea']**2 + 
                                     agg['Hswell1']**2/tp['Tpswell1']**2 +
                                     agg['Hswell2']**2/tp['Tpswell2']**2
                                ))
        # Bulk Dir
        forecast['Dir'] = np.arctan(
                (agg['Hsea']*agg['Tpsea']*np.sin(agg['Dirsea']*np.pi/180) +
                 agg['Hswell1']*agg['Tpswell1']*np.sin(agg['Dirswell1']*np.pi/180) +
                 agg['Hswell2']*agg['Tpswell2']*np.sin(agg['Dirswell2']*np.pi/180)) /
                (agg['Hsea']*agg['Tpsea']*np.cos(agg['Dirsea']*np.pi/180) +
                 agg['Hswell1']*agg['Tpswell1']*np.cos(agg['Dirswell1']*np.pi/180) +
                 agg['Hswell2']*agg['Tpswell2']*np.cos(agg['Dirswell2']*np.pi/180)))
        forecast['Dir'] = forecast['Dir'] * 180/np.pi
        forecast['Dir'] = forecast['Dir'].where(forecast['Dir']>0, 
                                                forecast['Dir']+360)
        forecast['Uwind']     = forecast_data['Uwind']
        forecast['Vwind']     = forecast_data['Vwind']
        forecast['WindSpeed'] = forecast_data['WindSpeed']
        print('\n')
        print('Saving the data in path="data/reconstructed/.." ... \n')
        forecast.to_pickle(op.join(p_data_swan, '..', 'reconstructed',
                                   'reconstructed_partitioned_'+name+'.pkl'))
        print(forecast)
        print(colored('\n SAVED!!! \n', 'red', attrs=['blink']))
        
        return forecast
    
            
    def plot_results(self, forecast, coast=True):
        """ This interactive plot helps the user see the reconstructed
            forecast and the normal plot shows the original forecast 
            prediction
            ------------
            Parameters
            forecast: the output of forecast_reconstruction() and also the
            output of select_region() as a dataframe
            ------------
            Returns
            Three different subplots in a plot
        """
        
        # Errors...
        register_matplotlib_converters()
        # ...
        
        labels = ['$H_S$ [m]', '$T_P$ [s]', '$\u03B8$ [$\degree$]']
        ini = str(self.times[0])
        end = str(self.times[-1])
        fig, axs = plt.subplots(3, 1, figsize=(20,15), sharex=True)
        fig.subplots_adjust(hspace=0.05, wspace=0.1)
        if coast:
            fig.suptitle('Forecast prediction in ' + str(self.coast_location) + ', COAST!',
                         fontsize=22, y=0.94, fontweight='bold')
        else:
            fig.suptitle('Forecast prediction in ' + str(self.location) + ', OFFSHORE!',
                         fontsize=22, y=0.94, fontweight='bold')
        i = 0
        while i < 3:
            if i==2:
                axs[i].plot(forecast[forecast.columns.values[i]], '.', markersize=8, color='darkblue')
                axs[i].plot(forecast[forecast.columns.values[i+3]], '.', markersize=8, color='red')
                axs[i].plot(forecast[forecast.columns.values[i+6]], '.', markersize=8, color='darkgreen')
                axs[i].plot(forecast[forecast.columns.values[i+9]], '.', markersize=8, color='orange')
                axs[i].set_ylabel(labels[i], fontsize=12, fontweight='bold')
                axs[i].grid()
                axs[i].set_xlim(ini, end)
                axs[i].set_xticks(np.arange(pd.to_datetime(ini), pd.to_datetime(end), 
                   datetime.timedelta(days=1)))
                axs[i].set_xticklabels([str(day)[5:10] for day in np.arange(pd.to_datetime(ini), 
                                                                            pd.to_datetime(end), 
                                                                            datetime.timedelta(days=1))], 
                            fontsize=18, fontweight='bold')
                axs[i].tick_params(direction='in')
            else:
                axs[i].plot(forecast[forecast.columns.values[i]], color='darkblue', linewidth=1)
                axs[i].plot(forecast[forecast.columns.values[i+3]], color='red', linewidth=1)
                axs[i].plot(forecast[forecast.columns.values[i+6]], color='darkgreen', linewidth=1)
                axs[i].plot(forecast[forecast.columns.values[i+9]], color='orange', linewidth=1)
                axs[i].set_ylabel(labels[i], fontsize=18, fontweight='bold')
                axs[i].grid()
                axs[i].tick_params(direction='in')
            fig.legend(['Sea', 'Swell1', 'Swell2', 'Bulk'], 
                       loc=(0.63, 0.03), ncol=4, fontsize=14)
            i += 1
            
