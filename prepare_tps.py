# Prepare labelled input for the N
# (i.e. locations where moss&lichen fractional cover changes can be related meteorological parameters from ERA5-Land)

# Copernicus Global Land Cover data  from 2015-01-01 to 2019-12-31 already available as a netCDF file stored locally
## Troms og Finnmark
### Only mosses and lichens

# Both the fractional cover, the ERA5 (2m temperature, total precipitation) values are normalized

import numpy as np
import pandas as pd
import os
import xarray as xr
import vaex

path = '/opt/uio/data/'

# World Land cover data from 2015-01-01 to 2019-12-31- already available as a netCDF file stored locally

GLC_filename = os.path.join(path, 'C_GlobalLandCover_20150101_20190101_Troms-Finnmark.nc')
print('Reading ', GLC_filename)
GLC_AOI = xr.open_dataset(GLC_filename, engine = 'netcdf4')
print('Read GLC_AOI')

# ERA5-land data already available as a netCDF file stored locally
## **For now will only use t2p and tp in the ML algorithm** although it may be useful to add sd

ERA5_filename = os.path.join(path, 'reanalysis-era5-land_hourly_2015-01-01_2022-12-31_Troms-Finnmark_T2m-SD-TP.nc')
print('Reading ', ERA5_filename)
ERA5land = xr.open_dataset(ERA5_filename, engine = 'netcdf4')
print('Read ERA5land')

GLC_AOI = GLC_AOI.rename(x='lon', y='lat', t='time')

# Drop variables not directly of interest here
GLC_AOI = GLC_AOI.drop_vars(['crs',
                             'Bare_CoverFraction_layer',
                             'Crops_CoverFraction_layer',
                             'Grass_CoverFraction_layer',
                             'Discrete_Classification_map', 
                             'Discrete_Classification_proba',
                             'Forest_Type_layer',
                             'Shrub_CoverFraction_layer',
                             'Snow_CoverFraction_layer',
                             'Tree_CoverFraction_layer',
                             'BuiltUp_CoverFraction_layer',
                             'PermanentWater_CoverFraction_layer',
                             'SeasonalWater_CoverFraction_layer',
                             'DataDensityIndicator',
                             'Change_Confidence_layer',
                             'dataMask'])

GLC_AOI = GLC_AOI.rename(MossLichen_CoverFraction_layer = 'Lichen')

# Troms & Finnmark Global Land Cover area
GLC_AOI_min_lon = GLC_AOI.lon.min()
GLC_AOI_max_lon = GLC_AOI.lon.max()
GLC_AOI_min_lat = GLC_AOI.lat.min()
GLC_AOI_max_lat = GLC_AOI.lat.max()
print(GLC_AOI_min_lon.values, GLC_AOI_max_lon.values, GLC_AOI_min_lat.values, GLC_AOI_max_lat.values)

## Use the mask to only keep pixels with lichen every year
mask = GLC_AOI['Lichen'].where(GLC_AOI['Lichen'] <= 100)
mask = xr.where(mask > 0, 1, 0)
mask = mask.sum(dim = 'time', min_count = 5, skipna=True)
mask = xr.where(mask == 5, 1, 0)

de = GLC_AOI.where(mask == 1)
de = de.to_dataframe()

print('GLC_AOI.to_dataframe')

# Drop the Rows with NaN Values
de = de.dropna()

de = de.reset_index()

## Each year in a separate dataset

# Loop starts here <------------------------------------------------------------------------------

Number_of_days = 183
for Year in range(2015, 2020):
    print('x = WLC(' + str(Year)+ ') joined with ERA5land(' + str(Year + 1) + ')')
    print('y = WLC(' + str(Year + 1) + ')')

# Only keep locations with lichen for the current year
    df = de[de['time'] == str(Year) + '-01-01']
    dg = de[de['time'] == str(Year + 1) + '-01-01']

# Normalize the fractional cover
    df['Lichen'] = df['Lichen'].div(100)
    dg['Lichen'] = dg['Lichen'].div(100)

# Convert to VAEX
    dvx = vaex.from_pandas(df)
    dvy = vaex.from_pandas(dg)

# Find the correspondind ERA5-land lat-lon
# Careful with the latitude, in reverse order
    dvx['ERA5_lon_index'] = ((dvx.lon - 15.59) / 0.1).astype('int').values
    dvx['ERA5_lat_index'] = 28 - ((dvx.lat - 68.35) / 0.1).astype('int').values
    dvy['ERA5_lon_index'] = ((dvy.lon - 15.59) / 0.1).astype('int').values
    dvy['ERA5_lat_index'] = 28 - ((dvy.lat - 68.35) / 0.1).astype('int').values

# Adding columns with the ERA5-land longitude and latitude to dv

    dvx['ERA5_lon'] = ERA5land.sel(time="2015-01-01").longitude[dvx['ERA5_lon_index'].values].values
    dvx['ERA5_lat'] = ERA5land.sel(time="2015-01-01").latitude[dvx['ERA5_lat_index'].values].values
    dvy['ERA5_lon'] = ERA5land.sel(time="2015-01-01").longitude[dvy['ERA5_lon_index'].values].values
    dvy['ERA5_lat'] = ERA5land.sel(time="2015-01-01").latitude[dvy['ERA5_lat_index'].values].values

## Extract ERA5 data for the selected period of the year (when RoS events mostly occur)

    ERA5 = ERA5land.sel(time=slice(str(Year + 1) + '-01-01', str(Year + 1)  + '-12-31'))
    ERA5 = ERA5.isel(time=range(Number_of_days * 24)) 
    ERA5 = ERA5.isel(expver = 0)

# Extract ERA5 t2m, tp and sd field 
    ERA5_t2m = ERA5.where(ERA5['latitude'].isin(dvx['ERA5_lat'].values) & ERA5['longitude'].isin(dvx['ERA5_lon'].values))['t2m']
    ERA5_tp = ERA5.where(ERA5['latitude'].isin(dvx['ERA5_lat'].values) & ERA5['longitude'].isin(dvx['ERA5_lon'].values))['tp']
    ERA5_sd = ERA5.where(ERA5['latitude'].isin(dvx['ERA5_lat'].values) & ERA5['longitude'].isin(dvx['ERA5_lon'].values))['sd']
    
    print('ERA5_t2m ', ERA5_t2m)

# Rain on Snow criteria (according to https://www.hydrol-earth-syst-sci.net/23/2983/2019/hess-23-2983-2019.pdf)
#
# total rainfall volume of at least 20 mm within 12 h
# or
# air temperatures above 0C (273.15K)
# and¶
# initial snowpack depth of at least 10 cm

# Normalizing temperature, total precipitation and snow depth values accordidng to these criteria
    ERA5_t2m = ERA5_t2m / 273.15
    ERA5_tp = ERA5_tp / 0.02 * 12.
    ERA5_sd = ERA5_sd / 0.1

    df_t2m = ERA5_t2m.stack(z=['latitude', 'longitude']).to_pandas().transpose().reset_index()
    df_tp = ERA5_tp.stack(z=['latitude', 'longitude']).to_pandas().transpose().reset_index()
    df_sd = ERA5_sd.stack(z=['latitude', 'longitude']).to_pandas().transpose().reset_index()
    df_tp = df_tp.drop(columns=['latitude', 'longitude'])
    df_sd = df_sd.drop(columns=['latitude', 'longitude'])

# Create labels for ERA5-land variables to replace the dates
    label_t2m = ['latitude', 'longitude']
    label_tp = list()
    label_sd = list()
    for i in range(Number_of_days * 24):
        label_t2m.append('t2m_'+ str(i))
        label_tp.append('tp_'+ str(i))
        label_sd.append('sd_'+ str(i))

    df_t2m.set_axis(label_t2m, axis="columns", inplace=True)
    df_tp.set_axis(label_tp, axis="columns", inplace=True)
    df_sd.set_axis(label_sd, axis="columns", inplace=True)

## Glue together df_t2m, df_tp and df_sd
    df = pd.concat([df_t2m, df_tp, df_sd], axis = 1)

# Add combined lon_lat column to df
    df['lon_lat'] = (df['longitude'] * 100).astype('int') + (df['latitude'] * 100).astype('int') / 100000

# Drop latitude and longitude columns which are not used anymore in df
    df = df.drop(columns=['latitude', 'longitude'])
    
# Add combined lon_lat column to dv x & y
    dvx['lon_lat'] = (dvx['ERA5_lon'] * 100).astype('int') + (dvx['ERA5_lat'] * 100).astype('int') / 100000

# Drop unused columns in dv x & y
    dwx = dvx.drop(columns=['time', 'ERA5_lon_index', 'ERA5_lat_index', 'ERA5_lon', 'ERA5_lat'])
    dwy = dvy.drop(columns=['time', 'ERA5_lon_index', 'ERA5_lat_index', 'ERA5_lon', 'ERA5_lat'])

# Convert to panda dw x & y
    dwx_pandas = dwx.to_pandas_df()
    dwy_pandas = dwy.to_pandas_df()

## Join dwx (WLC) with df (ERA5 t2m-tp-sd)
    dx = dwx_pandas.set_index('lon_lat').join(df.set_index('lon_lat'), on='lon_lat')
    print('dwx (WLC) joined with df (ERA5 t2m-tp-sd)')

# Drop the Rows with NaN Values
    dx = dx.dropna()
    dx = dx.reset_index()
    dx = dx.drop(columns=['lon_lat'])    
    
## Save into local HDF5 file without index
    x_filename = os.path.join(path, 'x_tps_' + str(Year) + '.hdf')
    print(x_filename)
    dx.to_hdf(x_filename, key='df', mode="w", index=False)

## Find locations with lichen in the following corresponding to those in current year
    dwx_pandas = dx[['lon', 'lat']]

# Add combined lat-lon column to dv x & y
    dwx_pandas['lon_lat'] = (dwx_pandas['lon'] * 100000).astype('int') + (dwx_pandas['lat'] * 100000).astype('int') / 10000000
    dwy_pandas['lon_lat'] = (dwy_pandas['lon'] * 100000).astype('int') + (dwy_pandas['lat'] * 100000).astype('int') / 10000000

    dwx_pandas = dwx_pandas.drop(columns=['lon', 'lat'])
    dwy_pandas = dwy_pandas.drop(columns=['lon', 'lat'])

## Join dwx with dwy
    dy = dwx_pandas.set_index('lon_lat').join(dwy_pandas.set_index('lon_lat'), on='lon_lat')
    print('dwx joined with dwy')

    dy = dy.reset_index().drop(columns=['lon_lat'])

## Save into local HDF5 file without index
    y_filename = os.path.join(path, 'y_' + str(Year) + '.hdf')
    print(y_filename)
    dy.to_hdf(y_filename, key='dg', mode="w", index=False)

print('Finished!')
