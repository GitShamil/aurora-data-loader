#!/usr/bin/env python3
"""
Скрипт для загрузки данных ERA5 из WeatherBench2 в формате Aurora.
"""

import xarray as xr
from pathlib import Path
import argparse
from datetime import datetime
import json

class Datasets:
    def __init__(self, ds_static, ds_surface, ds_atmospheric):
        self.ds_static = ds_static
        self.ds_surface = ds_surface
        self.ds_atmospheric = ds_atmospheric
        

def download_aurora_data(static_vars, surface_vars, atmospheric_vars, start_date, end_date, 
                         resolution='1440x721'):
    
    # Определяем путь к данным
    if resolution == '1440x721':
        era5_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    elif resolution == '256x128':
        era5_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-256x128_equiangular_conservative.zarr"
    else:
        raise ValueError(f"Неподдерживаемое разрешение: {resolution}")
    
    # Загружаем данные
    ds = xr.open_zarr(era5_path, storage_options={'token': 'anon'})
    ds_week = ds.sel(time=slice(start_date, end_date))
    
    # 1. Статические переменные
    static_vars = ['geopotential_at_surface', 'land_sea_mask', 'soil_type']
    static_data = {var: ds[var] for var in static_vars if var in ds.data_vars}
    if static_data:
        ds_static = xr.Dataset(static_data)
    
    # 2. Поверхностные переменные
    surface_vars = ['2m_temperature', '10m_u_component_of_wind', 
                    '10m_v_component_of_wind', 'mean_sea_level_pressure']
    surface_data = {var: ds_week[var] for var in surface_vars if var in ds_week.data_vars}
    if surface_data:
        ds_surface = xr.Dataset(surface_data)
    
    # 3. Атмосферные переменные
    atmospheric_vars = ['temperature', 'u_component_of_wind', 'v_component_of_wind',
                        'specific_humidity', 'geopotential']
    atmospheric_data = {var: ds_week[var] for var in atmospheric_vars if var in ds_week.data_vars}
    if atmospheric_data:
        ds_atmospheric = xr.Dataset(atmospheric_data)

    return Datasets(ds_static, ds_surface, ds_atmospheric)