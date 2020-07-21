# common 
import sys
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr

from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap

from time import time

import warnings
warnings.filterwarnings("ignore")

t0 = time()

# dev library 
sys.path.insert(0, op.join(op.dirname(__file__)))

# swan wrap module
from lib.wrap import SwanProject, SwanWrap_STAT

# --------------------------------------------------------------------------- #
# data
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_hind = op.join(p_data, 'hindcast')
waves = pd.read_pickle(op.join(p_hind, 'sea_cases_300.pkl'))[::50]

waves.rename(columns={'Hsea'   : 'hs',
                      'Tpsea'  : 'per',
                      'Dirsea' : 'dir',
                      'Sprsea' : 'spr',
                      'W'      : 'vel',
                      'DirW'   : 'dirw'}, inplace=True)

print(waves.info())

# --------------------------------------------------------------------------- #
# GRID PARAMETERS
# --------------------------------------------------------------------------- #
# ----------------  EDIT ONLY THIS PART  ------------------------------------ #
# --------------------------------------------------------------------------- #
name = 'SAF' # please choose a short name (max 3 letters)
# Coordinates section
# Place the coordinates as they are proportioned in Google Maps
ini_lon = 24.74
end_lon = 25.02
ini_lat = -34.26
end_lat = -33.95
# --------------------------------------------------------------------------- #
# ----------------  END EDIT  ----------------------------------------------- #

# --------------------------------------------------------------------------- #
# Depth auto-selection
# This file can substituted by the local bathymetry GEBCO file or other
# bathymetry file with the same aspect
p_depth = op.join(p_data, 'bathymetry', 'GEBCO_2020.nc')
depth = xr.open_dataset(p_depth)
depth = depth.sel(lat=slice(ini_lat,end_lat)).sel(lon=slice(ini_lon,end_lon))
x_point = len(depth.lon.values)
y_point = len(depth.lat.values)
resolution = round(abs(end_lon - ini_lon) / x_point, 4)

# --------------------------------------------------------------------------- #
# SWAN project 
p_proj = op.join(p_data, 'projects-swan')    # swan projects main directory
n_proj = name + '-SEA-' + str(resolution)    # project name

sp = SwanProject(p_proj, n_proj)

# depth grid description (input bathymetry grid)
sp.mesh_main.dg = {
    'xpc': ini_lon,            # x origin
    'ypc': ini_lat,            # y origin
    'alpc': 0,                 # x-axis direction 
    'xlenc': end_lon-ini_lon,  # grid length in x
    'ylenc': end_lat-ini_lat,  # grid length in y
    'mxc': x_point-1,          # number mesh x
    'myc': y_point-1,          # number mesh y
    'dxinp': resolution,       # size mesh x
    'dyinp': resolution,       # size mesh y
}

# depth swan init
sp.mesh_main.depth = - depth.elevation.values

# computational grid description
sp.mesh_main.cg = {
    'xpc': ini_lon,
    'ypc': ini_lat,
    'alpc': 0,
    'xlenc': end_lon-ini_lon,
    'ylenc': end_lat-ini_lat,
    'mxc': x_point,
    'myc': y_point,
    'dxinp': resolution,
    'dyinp': resolution,
}

# SWAN parameters (sea level, jonswap gamma)
sp.params = {
    'sea_level': 0,
    'jonswap_gamma': 3,
    'cdcap': None,
    'coords_spherical': None,
    'waves_period': 'PEAK',
    'maxerr': None,
}

# SWAN wrap STAT (create case files, launch SWAN num. model, extract output)
sw = SwanWrap_STAT(sp)

# build stationary cases from waves data
sw.build_cases(waves)

# run SWAN
sw.run_cases()

# extract output from main mesh 
waves_propagated = sw.extract_output()

# save to netCDF file and cases propagated to dataframe
waves.to_pickle(op.join(p_proj, n_proj, 'sea_cases_300.pkl'))
waves_propagated.to_netcdf(op.join(p_proj, n_proj, 
                           'sea_propagated_300.nc'))

print('Time transcurred: ' + str(round((time()-t0)/3600, 2)) + ' h')

plt.figure(figsize=(20,20))

# Plot the Basemap
m = Basemap(llcrnrlon=ini_lon,  llcrnrlat=ini_lat, 
            urcrnrlon=end_lon, urcrnrlat=end_lat, 
            resolution='l')
 
# Then add element: draw coast line, map boundary, and fill continents:
m.arcgisimage(service='NatGeo_World_Map')
grid_step_lon = round(abs(end_lon - ini_lon) / 10, 3)
grid_step_lat = round(abs(end_lat - ini_lat) / 10, 3)
m.drawmeridians(np.arange(ini_lon, end_lon+grid_step_lon, grid_step_lon), 
                linewidth=0.5, labels=[1,0,0,1])
m.drawparallels(np.arange(ini_lat, end_lat+grid_step_lat, grid_step_lat), 
                linewidth=0.5, labels=[1,0,0,1])

# -------------- CASE TO PLOT ----------------------------------------------- #
case = 2 # case to plot
# --------------------------------------------------------------------------- #
waves_case = waves.iloc[case]
plt.title(str(case), fontsize=18, fontweight='bold')
# --------------------------------------------------------------------------- #
# Hsig
xx = np.linspace(ini_lon, end_lon, x_point)
yy = np.linspace(ini_lat, end_lat, y_point)
X, Y = np.meshgrid(xx, yy)
hsig = waves_propagated.sel(case=case).Hsig.values.T
P = plt.pcolor(X, Y, hsig, cmap='hot_r', vmin=0, vmax=5)
PC = plt.colorbar(P)
PC.set_label('$H_{S}$ [m]', fontsize=22, fontweight='bold')
# --------------------------------------------------------------------------- #
# Dir, Dspr and Tp
dir_step = 2 # not all arrows are plotted
xx = xx[::dir_step]
yy = yy[::dir_step]
X, Y = np.meshgrid(xx, yy)
dirr = waves_propagated.sel(case=case).Dir.values.T
dirr = (dirr*np.pi/180)[::dir_step,::dir_step]
perr = waves_propagated.sel(case=case).TPsmoo.values.T
perr = perr[::dir_step,::dir_step]
U = -(np.sin(dirr) * perr)
V = -(np.cos(dirr) * perr)
plt.quiver(X, Y, U, V, color='k')
plt.xticks([])
plt.yticks([])

textstr = '\n'.join((
        r' $H_S$ = %.2f m' % (waves_case['hs'], ),
        r' $T_P$ = %.2f s' % (waves_case['per'], ),
        r' $\theta _{m}$ = %.2f $\degree$' % (waves_case['dir'], ),
        r' $\sigma _\theta$ = %.2f $\degree$' % (waves_case['spr'], ),
        r' $W$ = %.2f m/s' % (waves_case['vel'], ),
        r' $\theta _{W}$ = %.2f $\degree$' % (waves_case['dirw'], )))
plt.text(0.04, 0.1, textstr, 
         {'color': 'k', 'fontsize': 8},
         horizontalalignment='left',
         verticalalignment='center',
         transform=plt.gca().transAxes,
         bbox={'facecolor': 'white', 'alpha': 0.8, 'pad': 6})