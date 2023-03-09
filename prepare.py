# Prepare labelled input for the NN
# (i.e. locations where moss&lichen fractional cover changed and related meteorological parameters from ERA5-Land)

# Copernicus Global Land Cover data  from 2015-01-01 to 2019-12-31 already available as a netCDF file stored locally
## Troms og Finnmark
### Mosses and lichens, bare, grass, shrubs and trees

# Both the fractional cover and 2m temperature values are normalized

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

# ERA5-land data from 2015-01-01 to 2022-12-31 - already available as a netCDF file stored locally
## **For now will only use t2p in the ML algorithm** although it may be useful to know about rain and snow depth

ERA5_filename = os.path.join(path, 'reanalysis-era5-t2m_land_hourly_2015-01-01_2022-12-31_Troms-Finnmark_T2m-SD-TP.nc')
print('Reading ', ERA5_filename)
ERA5land = xr.open_dataset(ERA5_filename, engine = 'netcdf4')
print('Read ERA5land')

GLC_AOI = GLC_AOI.rename(x='lon', y='lat', t='time')

# Drop variables not directly of interest here
GLC_AOI = GLC_AOI.drop_vars(['crs',
                             'Crops_CoverFraction_layer',
                             'Discrete_Classification_map', 
                             'Discrete_Classification_proba',
                             'Forest_Type_layer',
                             'Snow_CoverFraction_layer',
                             'BuiltUp_CoverFraction_layer',
                             'PermanentWater_CoverFraction_layer',
                             'SeasonalWater_CoverFraction_layer',
                             'DataDensityIndicator',
                             'Change_Confidence_layer',
                             'dataMask'])

GLC_AOI = GLC_AOI.rename(Bare_CoverFraction_layer = 'Bare',
                         Grass_CoverFraction_layer = 'Grass',
                         MossLichen_CoverFraction_layer = 'Lichen',
                         Shrub_CoverFraction_layer = 'Shrub',
                         Tree_CoverFraction_layer = 'Tree')

# Troms & Finnmark Global Land Cover area
GLC_AOI_min_lon = GLC_AOI.lon.min()
GLC_AOI_max_lon = GLC_AOI.lon.max()
GLC_AOI_min_lat = GLC_AOI.lat.min()
GLC_AOI_max_lat = GLC_AOI.lat.max()
print(GLC_AOI_min_lon, GLC_AOI_max_lon, GLC_AOI_min_lat, GLC_AOI_max_lat)

de = GLC_AOI.to_dataframe()

print('GLC_AOI.to_dataframe')

de = de.reset_index()

# Change here to keep locations where there has been lichen at least once <-------------------------------------------------------------------------------
# Only keep the locations where there is lichen
dd = de.loc[(de['Lichen'] > 0) & (de['Lichen'] <= 100)]

## Each year in a separate dataset

# Start loop here 

Number_of_days = 183
for Year in range(2015, 2020):
    print('x = WLC(' + str(Year)+ ') joined with ERA5land(' + str(Year) + ')')
    print('y = WLC(' + str(Year + 1) + ')')

# Only keep locations for the current year
    df = dd.loc[de['time'] == str(Year) + '-01-01']
    dg = dd.loc[dd['time'] == str(Year + 1) + '-01-01']

# Replace NaNs by 0
    for col in ['Bare', 'Grass', 'Lichen', 'Shrub', 'Tree']:
        print(col)
        df[col] = df[col].fillna(0)
        dg[col] = dg[col].fillna(0)

# Calculate total fractional coverage of bare, grass, lichen, shrub and tree (should be 100)
    df['Total']  = (df['Bare'] + df['Grass'] + df['Lichen'] + df['Shrub'] + df['Tree'])
    dg['Total']  = (dg['Bare'] + dg['Grass'] + dg['Lichen'] + dg['Shrub'] + dg['Tree'])

# Normalize the fractional cover
    for col in ['Bare', 'Grass', 'Lichen', 'Shrub', 'Tree']:
        print(col)
        df[col] = df[col] / df['Total']
        dg[col] = dg[col] / dg['Total']

# Drop the *Total* column
    df = df.drop(['Total'], axis=1)
    dg = dg.drop(['Total'], axis=1)

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

## Extract ERA5 data for  the selected period of the year (when RoS events mostly occur)

    ERA5 = ERA5land.sel(time=slice(str(Year + 1) + '-01-01', str(Year + 1)  + '-12-31'))
    ERA5 = ERA5.isel(time=range(Number_of_days * 24))
    ERA5 = ERA5.isel(expver = 0)

# Extract ERA5 't2m' field 
    ERA5 = ERA5.where(ERA5['latitude'].isin(dvx['ERA5_lat'].values) & ERA5['longitude'].isin(dvx['ERA5_lon'].values))['t2m']
    print('ERA5 ', ERA5)

# Calculate the first time using the 2015 values
    t2m_mean = ERA5.mean(skipna=True).values
    print('Mean of the ERA5-Land 2m temperature: ', t2m_mean)
    t2m_std = ERA5.std(skipna=True).values
    print('Standard deviation of the ERA5-Land 2m temperature: ', t2m_std)

# Set once and for all
    t2m_mean = 267.1025
    t2m_std = 14.740288734436035

# Normalizing temperature values
    ERA5 = (ERA5 -t2m_mean) / t2m_std

    df = ERA5.stack(z=['latitude', 'longitude']).to_pandas().transpose().reset_index()

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

## Join dwx (WLC) with df (ERA5 t2m)

    dx = dwx_pandas.set_index('lon_lat').join(df.set_index('lon_lat'), on='lon_lat')
    print('dwx (WLC) joined with df (ERA5 t2m)')

# Drop the Rows with NaN Values
    dx = dx.dropna()

    dx = dx.reset_index()

    dx = dx.drop(columns=['lon_lat'])

## Save into **local** HDF5 file with header and no indices
    x_filename = os.path.join(path, 'x_' + str(Year) + '.hdf')
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

# Replace NaNs by 0
    for col in ['Bare', 'Grass', 'Lichen', 'Shrub', 'Tree']:
        print(col)
        dy[col] = dy[col].fillna(0)

    dy = dy.reset_index().drop(columns=['lon_lat'])

## Save into **local** HDF5 file with header and no indices
    y_filename = os.path.join(path, 'y_' + str(Year) + '.hdf')
    print(y_filename)
    dy.to_hdf(y_filename, key='dg', mode="w", index=False)

print('Finished!')