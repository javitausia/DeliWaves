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
import matplotlib
import matplotlib.pyplot as plt
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
class Forecast_CAN(object):
    """ This forecast class has several functions that allows the user
        to correctly predict the waves and the surfing conditions with
        precision in that place where the SWAN propagations have been
        done. These SWAN propagations can be performed using the attached
        notebook and scripts, but in case the propagations are not wanted
        to be done, the global prediction in offshore points is also
        proportioned with this tool. Refer to the jupyter notebook for
        more important information, also in the repository.
    """
    
    def __init__(self, date, images_path, location, delta_lon, delta_lat):
        """ Initializes the class with all the necessary attributes that
            will be used in the different methods
            ------------
            Parameters
            date: date to initialize the forecast in format %YYYYmmdd%
            images_path: path to save the images and GIF
            location: location to obtain the forecast
            delta_lon: width (longitude) for the area to download around the
                       selected location
            delta_lat: height (latitude) for the area to download around the
                       selected location
            ------------
            Returns
            The initialized attributes and a GIF in path with the 
            global forecast
        """
        
        print('Pulling the data from: \n')
        url = 'https://nomads.ncep.noaa.gov/dods/wave/mww3/'+date+'/multi_1.glo_30mext'+date+'_00z'
        print(url)
        print('\n')
        
        # Initialization
        self.forecast        =   netCDF4.Dataset(url)
        self.images_path     =   images_path
        self.location        =   location
        self.delta_lon       =   delta_lon
        self.delta_lat       =   delta_lat
        # self.coast_location  =   (0, 0) # will be filled after
        
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
        fig = plt.figure(figsize=(15,15))
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
        msg  = 'Number of images to plot from the total? \n'
        msg += 'TOTAL: {}, To plot: \n'.format(len(times))
        print('\n')
        num_images_plot = int(input(msg))
        step = int(len(times)/num_images_plot)
        for t in range(0, len(times), step):
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
        
        
    def plot_region(self, zoom=6):
        """ This plot helps the user see the region that will be saved
            so more than one single forecast node will be available
            ------------
            Parameters
            zoom: zoom to see the plot
            ------------
            Returns
            the data saved as an xarray dataset and a pandas dataframe to
            plot the results easily
        """
        
        # Relocate the precise location selected previously
        print(colored('Location in {}!! \n'.format(self.location),
                      'red', attrs=['blink']))
        lat = self.location[0]
        if self.location[1]>0:
            lon = self.location[1]
        else:
            lon = self.location[1]+360
        lat_index = np.where((self.forecast.variables['lat'][:].data < (lat+self.delta_lat)) & 
                             (self.forecast.variables['lat'][:].data > (lat-self.delta_lat)))
        lon_index = np.where((self.forecast.variables['lon'][:].data < (lon+self.delta_lon)) & 
                             (self.forecast.variables['lon'][:].data > (lon-self.delta_lon)))
        
        print('These are the coordinates in the selected region: \n')
        print(self.forecast.variables['lat'][:][list(lat_index[0])])
        print(self.forecast.variables['lon'][:][list(lon_index[0])])
        print('\n')
        
        # Map plotting to see the downloading data
        m = Map(basemap=basemaps.Esri.WorldImagery,
                center=(lat,lon), zoom=zoom)
        rectangle = Rectangle(bounds=((lat-self.delta_lat, lon-self.delta_lon), 
                                      (lat+self.delta_lat, lon+self.delta_lon)),
                              color='red', opacity=0.1)
        m.add_layer(rectangle)
        marker = Marker(location=(lat,lon))
        m.add_layer(marker)
        display(m)
        
        print('\n If selected coordinates are not desired, ')
        print('please rerun the notebook with the desired parameters!! \n')
            
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
                                           self.forecast.variables['windsfc'][:,lat_index[0],lon_index[0]].data),
                           'DirWind'    : (['time', 'lat', 'lon'],
                                           self.forecast.variables['wdirsfc'][:,lat_index[0],lon_index[0]].data)},
                          coords = {'time' : self.times,
                                    'lat'  : self.forecast.variables['lat'][:][lat_index[0]],
                                    'lon'  : self.forecast.variables['lon'][:][lon_index[0]]})
        print(data)
        
        print('\n DONE!! \n')
            
        return data
    
    
    def select_recon_points(self):
        """
            Explanation
        """
        pass
    
    
    def forecast_reconstruction(self, p_data_swan, info, forecast_data):
        """ This method reconstruct the forecast information in the
            previously selected region
            ------------
            Parameters
            p_data_swan: path to find the necessary data to 
                         reconstruct (SWAN)
            info: dictionary with the important data variables:
                    name: name of the region in folder
                    resolution: resolution of the region in folder
                    num_cases: num_cases of the region in folder
            forecast_data: data in the selected region (dataset)
            ------------
            Returns
            the reconstructed forecast in the chose point as a dataframe
        """
        
        # List to save all the forecasts reconstructed
        forecasts_list = []
        
        # Here, we choose the locations we will use for the offshores
        for off_loni in range(4):
            
            # SUBSETS
            subsetsea    = pd.read_pickle(op.join(p_data_swan, 
                                                  info['name'][off_loni]+'-SEA-'+info['resolution'][off_loni],
                                                  'sea_cases_'+info['num_cases'][off_loni]+'.pkl'))
            subsetsea    = subsetsea[['hs', 'per', 'dir']]
            subsetswell  = pd.read_pickle(op.join(p_data_swan, 
                                                  info['name'][off_loni]+'-SWELL-'+info['resolution'][off_loni],
                                                  'swell_cases_'+info['num_cases'][off_loni]+'.pkl'))
            subsetswell  = subsetswell[['hs', 'per', 'dir']]
            
            # print(colored('SUBSETS: \n', 'blue', attrs=['blink', 'reverse']))
            # print(colored('SEA', 'red', attrs=['blink']))
            # print(subsetsea.info())
            # print(colored('SWELL', 'red', attrs=['blink']))
            # print(subsetswell.info())
            # print('\n')
        
            # TARGETS
            targetsea    = xr.open_dataset(op.join(p_data_swan, 
                                                   info['name'][off_loni]+'-SEA-'+info['resolution'][off_loni],
                                                   'sea_propagated_'+info['num_cases'][off_loni]+'.nc'))
            targetswell  = xr.open_dataset(op.join(p_data_swan, 
                                                   info['name'][off_loni]+'-SWELL-'+info['resolution'][off_loni],
                                                   'swell_propagated_'+info['num_cases'][off_loni]+'.nc'))
            
            # Selection of the desired point
            num_recons_grid_msg = '\n Select the number of points to '
            num_recons_grid_msg += 'reconstruct in GRID ' + str(off_loni +1) + ' : \n'
            num_recons_grid = int(input(num_recons_grid_msg))
            print('\n')
            
            # DATASETS
            forecasts_list.append(forecast_data.isel(lat=0).isel(lon=off_loni))
            forecast_data_new = forecast_data.isel(lat=0).isel(lon=off_loni).to_dataframe()
            forecast_data_new = forecast_data_new.where(forecast_data_new<1000, 0)
            
            # Reconstructios
            for recons in range(num_recons_grid):
                
                print(colored('Select the desired point to reconstruct in GRID ' 
                              + str(off_loni + 1) + 
                              ' as it is given in Google Maps: \n',
                              'blue', attrs=['blink', 'reverse']))
                lat_new = float(input('Latitude location to obtain the forecast reconstruction: '))
                lon_new = float(input('Longitude location to obtain the forecast reconstruction: \n'))
                print('\n')
                ilat = np.where((targetsea.Y.values < lat_new+0.01) & 
                                (targetsea.Y.values > lat_new-0.01))[0][0]
                ilon = np.where((targetswell.X.values < lon_new+0.01) &
                                (targetswell.X.values > lon_new-0.01))[0][0]
                
                targetsea_new   = targetsea.isel(X=ilon).isel(Y=ilat)
                targetswell_new = targetswell.isel(X=ilon).isel(Y=ilat)
                
                targetsea_new   = pd.DataFrame({'hs': targetsea_new.Hsig.values,
                                                'per': targetsea_new.TPsmoo.values,
                                                'perM': targetsea_new.Tm02.values,
                                                'dir': targetsea_new.Dir.values,
                                                'spr': targetsea_new.Dspr.values})
                seaedit_new         = subsetsea.mean()
                seaedit_new['perM'] = 7.0
                seaedit_new['spr']  = 22.0
                targetsea_new       = targetsea_new.fillna(seaedit_new)
                
                targetswell_new     = pd.DataFrame({'hs': targetswell_new.Hsig.values,
                                                    'per': targetswell_new.TPsmoo.values,
                                                    'perM': targetswell_new.Tm02.values,
                                                    'dir': targetswell_new.Dir.values,
                                                    'spr': targetswell_new.Dspr.values})
                swelledit_new         = subsetswell.mean()
                swelledit_new['perM'] = 12.0
                swelledit_new['spr']  = 12.0
                targetswell_new       = targetswell_new.fillna(swelledit_new)
                
                # print(colored('TARGETS: \n', 'blue', attrs=['blink', 'reverse']))
                # print(colored('SEA', 'red', attrs=['blink']))
                # print(targetsea.info())
                # print(colored('SWELL', 'red', attrs=['blink']))
                # print(targetswell.info())
                # print('\n')
                
                print(colored('\n Forecast in the selected region will be calculated!! \n',
                              'blue', attrs=['blink', 'reverse']))
            
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
                    dataset_ss = forecast_data_new[ss]
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
                        target  = targetsea_new.to_numpy()
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
                        target  = targetswell_new.to_numpy()
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
                forecast_data_red = forecast_data_new[['Hsea', 'Tpsea', 'Dirsea',
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
                forecast['Uwind']     = forecast_data_new['Uwind']
                forecast['Vwind']     = forecast_data_new['Vwind']
                forecast['WindSpeed'] = forecast_data_new['WindSpeed']
                forecast['DirWind']   = forecast_data_new['DirWind']
                
                print('\n')
                print('Saving the data in list... \n')
                forecasts_list.append(forecast.to_xarray().assign_coords(lat=lat_new).assign_coords(lon=lon_new))
                print(colored('\n SAVED!!! \n', 'red', attrs=['blink']))
                
        # We concat all the datasets and create the dataset
        # forecast_dataset = xr.merge(forecasts_list, compat='override')
        # print(forecast_dataset)
        
        return forecasts_list
    
    
    def plot_results(self, forecasts_list):
        """ This method plots the forecasts in the selected points
            ------------
            Parameters
            forecasts_list : a python list with all the necessary forecast
                             points reconstructed nearshore. This list 
                             is the output of forecast_reconstruction()
            ------------
            Returns
            the plotted forecasts in the chose points
        """
        
        # Figure intialization
        fig = plt.figure(figsize=(15,15))
        
        ini_lon = -4.8
        end_lon = -2.8
        ini_lat = 43.2
        end_lat = 44.2
        
        # Plot the Basemap
        m = Basemap(llcrnrlon=ini_lon,  
                    llcrnrlat=ini_lat, 
                    urcrnrlon=end_lon, 
                    urcrnrlat=end_lat, 
                    resolution='l')
         
        # Then add element: draw coast line, map boundary, and fill continents:
        m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels=1000) # becareful with adding xpixels
        grid_step_lon = round(abs(end_lon - ini_lon) / 8, 3)
        grid_step_lat = round(abs(end_lat - ini_lat) / 5, 3)
        m.drawmeridians(np.arange(ini_lon, end_lon+grid_step_lon, grid_step_lon), 
                        linewidth=0.5, labels=[1,0,0,1])
        m.drawparallels(np.arange(ini_lat, end_lat+grid_step_lat, grid_step_lat), 
                        linewidth=0.5, labels=[1,0,0,1])
        
        # Filenames for the images
        filenames = []
        
        # Now we construct the matrices for the lat and lon
        lat  = np.array([fp['lat'].values for fp in forecasts_list])
        lon  = np.array([fp['lon'].values for fp in forecasts_list])
        lon  = np.where(lon>180, lon-360, lon)
        msg  = '\n Number of images to plot from the total? \n'
        msg += 'TOTAL: {}, To plot: \n'.format(len(self.times))
        print('\n \n')
        num_images_plot = int(input(msg))
        step = int(len(self.times)/num_images_plot)
        for t in range(0, len(self.times), step):
            if t>=1:
                for quiv in quivs:
                    quiv.remove()
            print('Plotting time: {}...'.format(self.times[t]))
            hs       = [fp.isel(time=t).Hs.values for fp in forecasts_list]
            norm     = matplotlib.colors.Normalize(vmin=0, vmax=3)
            cmap     = matplotlib.cm.jet
            sm       = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([]) 
            direc    = [fp.isel(time=t).Dir.values for fp in forecasts_list]
            tp       = [fp.isel(time=t).Tp.values for fp in forecasts_list]
            U, V, new_direc = self.direc_transformation(direc=direc,
                                                        tp=tp)
            # Now, we plot the quivs!!
            quivs = []
            for tt in range(len(hs)):
                quivs.append(m.quiver(lon[tt], lat[tt], 
                                      U[tt], V[tt], 
                                      color=cmap(norm(hs[tt]))))
            m.colorbar(sm)
            plt.title(self.times[t], fontsize=18, fontweight='bold')
            fig.savefig(op.join(self.images_path, 
                                '{}.png'.format(self.times[t])))
            filenames.append('{}.png'.format(self.times[t]))
            
        # GIF
        images = []
        for filename in filenames:
            images.append(imageio.imread(op.join(self.images_path, filename)))
        imageio.mimsave(op.join(self.images_path, 'forecast_CAN.gif'), 
                        images, duration = 1.0)
        print(colored('\n GIF generated and saved!! \n', 'red', 
                      attrs=['blink']))
        print('\n')
        
        
    def direc_transformation(self, direc, tp):
        """
            This simple function arranges the directions so the plot
            is understandable in our region of interest
        """
        
        direcs = []
        us     = []
        vs     = []
        
        for d, t in zip(direc, tp):
            if d<90:
                direcs.append(90-d)
                us.append(- np.cos((90-d)*np.pi/180) * t)
                vs.append(- np.sin((90-d)*np.pi/180) * t)
            elif d<180:
                direcs.append(360-(d-90))
                us.append(- np.cos((360-(d-90))*np.pi/180) * t)
                vs.append(- np.sin((360-(d-90))*np.pi/180) * t)
            elif d<270:
                direcs.append(180+(270-d))
                us.append(- np.cos((180+(270-d))*np.pi/180) * t)
                vs.append(- np.sin((180+(270-d))*np.pi/180) * t)
            else:
                direcs.append(90+(360-d))
                us.append(- np.cos((90+(360-d))*np.pi/180) * t)
                vs.append(- np.sin((90+(360-d))*np.pi/180) * t)
        
        return us, vs, direcs
    
