#!/usr/bin/env python3
"""
Dataset для загрузки и преобразования данных ERA5 в формат Aurora Batch.
Ленивая загрузка - данные загружаются только при обращении к элементу.
"""

import torch
from torch.utils.data import Dataset
import xarray as xr
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

from .batch import Batch, Metadata
from .variable_mapper import VariableMapper
from .download_data import download_aurora_data


class ERA5Dataset(Dataset):
    """Датасет для работы с данными ERA5 в формате Aurora.

    Особенности:
    - Ленивая загрузка данных (не загружает все в память)
    - Поддержка временных окон (history steps)
    - Автоматическое преобразование названий переменных
    - Создание объектов Batch для работы с моделью
    """

    def __init__(
        self,
        atmos_levels: List[float] = None,
        atmos_vars: List[str] = None,
        surface_vars: List[str] = None,
        static_vars: List[str] = None,
        start_date: str = "2018-01-01",
        end_date: str = "2018-01-07",
    ):
        """
        Args:
            data_dir: Директория с данными
            atmos_levels: Список атмосферных уровней
        """
        if atmos_levels is None:
            self.atmos_levels = [
                50,
                100,
                150,
                200,
                250,
                300,
                400,
                500,
                600,
                700,
                850,
                925,
                1000,
            ]
        else:
            self.atmos_levels = atmos_levels

            # Устанавливаем переменные по умолчанию если не указаны
        if atmos_vars is None:
            self.atmos_vars = [
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "specific_humidity",
                "geopotential",
            ]
        else:
            self.atmos_vars = atmos_vars

        if surface_vars is None:
            self.surface_vars = [
                "2m_temperature",
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "mean_sea_level_pressure",
            ]
        else:
            self.surface_vars = surface_vars

        if static_vars is None:
            self.static_vars = ["geopotential_at_surface", "land_sea_mask", "soil_type"]
        else:
            self.static_vars = static_vars

        self.data = download_aurora_data(
            static_vars, surface_vars, atmos_vars, start_date, end_date
        )

        self.times = list(self.data.ds_surface.time.values)
        self._load_static_data()

        self.variable_mapper = VariableMapper()

    def _load_static_data(self):
        """Загружает статические данные в память."""
        self.static_tensors = {}
        for var in self.static_vars:
            if var in self.data.ds_static:
                # Конвертируем в тензор [H, W]
                data = self.data.ds_static[var].values
                self.static_tensors[var] = torch.from_numpy(data.astype(np.float32))

    def __len__(self):
        """Количество временных снимков в датасете."""
        return len(self.times)

    def __getitem__(self, idx):
        """
        Возвращает один временной снимок в формате Aurora Batch.

        Args:
            idx: Индекс временного шага (0 до num_times-1)

        Returns:
            Batch объект с batch_size=1, history_steps=1
        """
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range")

        return self._create_single_timestep_batch(idx)

    def _create_single_timestep_batch(self, time_idx):
        """
        Создает объект Batch для одного временного шага.

        Args:
            time_idx: Индекс временного шага

        Returns:
            Batch объект
        """
        # Собираем поверхностные переменные
        surf_tensors = {}
        for wb2_var in self.surface_vars:
            if wb2_var in self.data.ds_surface:
                # Берем данные для выбранного временного шага
                var_data = (
                    self.data.ds_surface[wb2_var].isel(time=time_idx).values
                )  # [H, W]

                # Добавляем batch и time размерности: [1, 1, H, W]
                var_tensor = torch.from_numpy(var_data.astype(np.float32))
                var_tensor = var_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

                # Маппинг названия в формат Aurora
                aurora_var_name = self.variable_mapper.map_surface(wb2_var)
                surf_tensors[aurora_var_name] = var_tensor

        # Собираем атмосферные переменные
        atmos_tensors = {}
        for wb2_var in self.atmos_vars:
            if wb2_var in self.data.ds_atmospheric:
                # Берем данные для выбранного временного шага
                var_data = (
                    self.data.ds_atmospheric[wb2_var].isel(time=time_idx).values
                )  # [levels, H, W]

                # В Aurora каждая атмосферная переменная содержит ВСЕ уровни
                # Формат: [1, 1, levels, H, W] где levels = len(self.atmos_levels)
                var_tensor = torch.from_numpy(var_data.astype(np.float32))
                
                # Добавляем batch и time размерности: [1, 1, levels, H, W]
                var_tensor = var_tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, levels, H, W]

                # Маппинг названия в формат Aurora (без указания уровня!)
                # В Aurora название переменной не включает уровень
                aurora_var_name = self.variable_mapper.map_atmospheric(wb2_var, level=None)
                atmos_tensors[aurora_var_name] = var_tensor

        # Статические переменные
        static_tensors = {}
        for wb2_var in self.static_vars:
            if wb2_var in self.static_tensors:
                # Маппинг названия
                aurora_var_name = self.variable_mapper.map_static(wb2_var)
                # Статические данные: [H, W]
                static_tensors[aurora_var_name] = self.static_tensors[wb2_var]

        # Создаем метаданные
        metadata = self._create_metadata(time_idx)

        # Создаем объект Batch
        batch = Batch(
            surf_vars=surf_tensors,
            static_vars=static_tensors,
            atmos_vars=atmos_tensors,
            metadata=metadata,
        )

        return batch

    def _create_metadata(self, time_idx):
        """
        Создает объект Metadata.

        Args:
            time_idx: Индекс временного шага

        Returns:
            Metadata объект
        """
        # Получаем координаты из данных
        lat = torch.from_numpy(self.data.ds_surface.latitude.values.astype(np.float32))
        lon = torch.from_numpy(self.data.ds_surface.longitude.values.astype(np.float32))

        # Время для этого батча
        time = self.times[time_idx]
        dt_str = np.datetime_as_string(time, unit='s')  # '2021-05-03T06:00:00'
        dt_str = dt_str.replace('T', ' ')
        time = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')

        # Создаем Metadata
        metadata = Metadata(
            lat=lat,
            lon=lon,
            time=(time,),  # tuple с одним временем
            atmos_levels=tuple(self.atmos_levels),
        )

        return metadata
