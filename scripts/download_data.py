#!/usr/bin/env python3
"""
Скрипт для загрузки данных ERA5 из WeatherBench2 в формате Aurora.
"""

import xarray as xr
from pathlib import Path
import argparse
from datetime import datetime
import json

def download_aurora_data(start_date='2018-01-01', end_date='2018-01-07', 
                         resolution='1440x721', output_dir='./data'):
    
    # Определяем путь к данным
    if resolution == '1440x721':
        era5_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"
    elif resolution == '256x128':
        era5_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-6h-256x128_equiangular_conservative.zarr"
    else:
        raise ValueError(f"Неподдерживаемое разрешение: {resolution}")
    
    # Создаем директорию
    download_path = Path(output_dir).expanduser()
    download_path.mkdir(parents=True, exist_ok=True)
    
    # Загружаем данные
    ds = xr.open_zarr(era5_path, storage_options={'token': 'anon'})
    ds_week = ds.sel(time=slice(start_date, end_date))
    
    # 1. Статические переменные
    static_vars = ['geopotential_at_surface', 'land_sea_mask', 'soil_type']
    static_data = {var: ds[var] for var in static_vars if var in ds.data_vars}
    if static_data:
        ds_static = xr.Dataset(static_data)
        ds_static.to_netcdf(download_path / "static.nc")
    
    # 2. Поверхностные переменные
    surface_vars = ['2m_temperature', '10m_u_component_of_wind', 
                    '10m_v_component_of_wind', 'mean_sea_level_pressure']
    surface_data = {var: ds_week[var] for var in surface_vars if var in ds_week.data_vars}
    if surface_data:
        ds_surface = xr.Dataset(surface_data)
        ds_surface.to_netcdf(download_path / f"surface_{start_date}_{end_date}.nc")
    
    # 3. Атмосферные переменные
    atmospheric_vars = ['temperature', 'u_component_of_wind', 'v_component_of_wind',
                        'specific_humidity', 'geopotential']
    atmospheric_data = {var: ds_week[var] for var in atmospheric_vars if var in ds_week.data_vars}
    if atmospheric_data:
        ds_atmospheric = xr.Dataset(atmospheric_data)
        ds_atmospheric.to_netcdf(download_path / f"atmospheric_{start_date}_{end_date}.nc")
    
    # 4. Метаданные
    metadata = {
        'data_source': 'WeatherBench2 ERA5',
        'resolution': resolution,
        'time_range': {'start': start_date, 'end': end_date},
        'num_timesteps': len(ds_week.time),
        'download_date': datetime.now().isoformat()
    }
    
    metadata_file = download_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Данные сохранены в {download_path}")
    return download_path

def main():
    parser = argparse.ArgumentParser(description='Загрузка данных ERA5')
    
    parser.add_argument('--start-date', default='2018-01-01')
    parser.add_argument('--end-date', default='2018-01-07')
    parser.add_argument('--resolution', default='1440x721', choices=['1440x721', '256x128'])
    parser.add_argument('--output-dir', default='./data')
    
    args = parser.parse_args()
    
    download_aurora_data(
        start_date=args.start_date,
        end_date=args.end_date,
        resolution=args.resolution,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()