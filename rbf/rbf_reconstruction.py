# common 
import sys
import os.path as op

# basic 
import xarray as xr
import numpy as np
import pandas as pd

# dev library 
sys.path.insert(0, op.join(op.dirname(__file__)))

# RBF module 
from rbf_main import RBF_Reconstruction

print('Reading data and preprocessing... \n')

p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data', 
                            'projects-swan'))

# -------------- EDIT THIS PART --------------------------------------------- #
name = 'SAF' # used name in the SWAN section
resolution = str(0.0042) # used resolution in the SWAN section
# --------------------------------------------------------------------------- #

# SUBSETS
subsetsea    = pd.read_pickle(op.join(p_data, name+'-SEA-'+resolution,
                                      'sea_cases_300.pkl'))
subsetsea    = subsetsea[['hs', 'per', 'dir', 'spr']]

subsetswell  = pd.read_pickle(op.join(p_data, name+'-SWELL-'+resolution,
                                      'swell_cases_300.pkl'))
subsetswell  = subsetswell[['hs', 'per', 'dir', 'spr']]

# TARGETS
targetsea    = xr.open_dataset(op.join(p_data, name+'-SEA-'+resolution,
                                       'sea_propagated_300.nc'))
targetswell  = xr.open_dataset(op.join(p_data, name+'-SWELL-'+resolution,
                                       'swell_propagated_300.nc'))

# Reconstruction desired point
lat = -34.14
lon = 24.90

lat = np.where((targetsea.Y.values<lat+0.005) & 
               (targetsea.Y.values>lat-0.005))[0][0]
lon = np.where((targetswell.X.values<lon+0.005) &
               (targetswell.X.values>lon-0.005))[0][0]

targetsea   = targetsea.isel(X=lon).isel(Y=lat)
targetswell = targetswell.isel(X=lon).isel(Y=lat)

targetsea   = pd.DataFrame({'hs': targetsea.Hsig.values,
                            'per': targetsea.TPsmoo.values,
                            'perM': targetsea.Tm02.values,
                            'dir': targetsea.Dir.values,
                            'spr': targetsea.Dspr.values})
seaedit         = subsetsea.mean()
seaedit['perM'] = 7.0
targetsea       = targetsea.fillna(seaedit)

targetswell = pd.DataFrame({'hs': targetswell.Hsig.values,
                            'per': targetswell.TPsmoo.values,
                            'perM': targetswell.Tm02.values,
                            'dir': targetswell.Dir.values,
                            'spr': targetswell.Dspr.values})
swelledit         = subsetswell.mean()
swelledit['perM'] = 12.0
targetswell       = targetswell.fillna(swelledit)

# DATASETS
dataset_tot = pd.read_pickle(op.join(p_data, '..', 'hindcast',
                                     'csiro_dataframe_sat_corr.pkl'))

print(dataset_tot.info())

labels_input   = [['Hsea', 'Tpsea', 'Dirsea', 'Sprsea'],
                  ['Hswell1', 'Tpswell1','Dirswell1', 'Sprswell1'],
                  ['Hswell2', 'Tpswell2','Dirswell2', 'Sprswell2'],
                  ['Hswell3', 'Tpswell3','Dirswell3', 'Sprswell3']]
labels_output  = [['Hsea', 'Tpsea', 'Tm_02', 'Dirsea', 'Sprsea'],
                  ['Hswell1', 'Tpswell1', 'Tm_02','Dirswell1', 'Sprswell1'],
                  ['Hswell2', 'Tpswell2', 'Tm_02','Dirswell2', 'Sprswell2'],
                  ['Hswell3', 'Tpswell3', 'Tm_02','Dirswell3', 'Sprswell3']]

datasets = []

for ss in labels_input:
    dataset_ss = dataset_tot[ss]
    dataset_ss = dataset_ss.dropna(axis=0, how='any')
    #new_labels = ['hs', 'per', 'dir', 'spr']
    #for l, lab in enumerate(ss):
    #     dataset_ss.rename(columns={lab: new_labels[l]}, inplace=True)
    datasets.append(dataset_ss)
    
dataframes = []

print('Performing RFB reconstruction... \n')

# RBF 
for count, dat in enumerate(datasets):
    # Scalar and directional columns
    ix_scalar_subset = [0,1,3]
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

reconstructed_dataframe.to_pickle(op.join(p_data, 'reconstructed',
                                  'reconstructed_partitioned_'+name+'.pkl'))