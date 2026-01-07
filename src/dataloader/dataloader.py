import torch
from typing import List
from .batch import Batch, Metadata
from .dataset import ERA5Dataset
from torch.utils.data import DataLoader

def collate_batches(batches: List[Batch]) -> Batch:
    """
    Объединяет несколько Batch объектов в один.
    
    Args:
        batches: Список Batch объектов
        
    Returns:
        Один объединенный Batch объект
    """
    if len(batches) == 1:
        return batches[0]
    
    # Проверяем, что все батчи имеют одинаковую структуру
    first_batch = batches[0]
    
    # 1. Объединяем поверхностные переменные
    surf_vars = {}
    for var_name in first_batch.surf_vars.keys():
        # Собираем тензоры из всех батчей
        tensors = [batch.surf_vars[var_name] for batch in batches]
        # Конкатенируем по batch dimension (dimension 0)
        # Исходная форма: [1, 1, H, W] -> после конкатенации: [B, 1, H, W]
        surf_vars[var_name] = torch.cat(tensors, dim=0)
    
    # 2. Объединяем атмосферные переменные
    atmos_vars = {}
    for var_name in first_batch.atmos_vars.keys():
        tensors = [batch.atmos_vars[var_name] for batch in batches]
        # Исходная форма: [1, 1, 1, H, W] -> после конкатенации: [B, 1, 1, H, W]
        atmos_vars[var_name] = torch.cat(tensors, dim=0)
    
    # 3. Статические переменные одинаковы для всех батчей
    # Берем из первого батча (они все одинаковые)
    static_vars = first_batch.static_vars
    
    # 4. Объединяем метаданные
    # Собираем все времена
    all_times = []
    for batch in batches:
        all_times.extend(batch.metadata.time)
    
    # Проверяем, что все координаты одинаковые
    for batch in batches[1:]:
        if not torch.allclose(batch.metadata.lat, first_batch.metadata.lat):
            raise ValueError("Latitudes don't match between batches")
        if not torch.allclose(batch.metadata.lon, first_batch.metadata.lon):
            raise ValueError("Longitudes don't match between batches")
        if batch.metadata.atmos_levels != first_batch.metadata.atmos_levels:
            raise ValueError("Atmos levels don't match between batches")
    
    # Создаем новые метаданные
    metadata = Metadata(
        lat=first_batch.metadata.lat,
        lon=first_batch.metadata.lon,
        time=tuple(all_times),
        atmos_levels=first_batch.metadata.atmos_levels
    )
    
    # Создаем объединенный Batch
    return Batch(
        surf_vars=surf_vars,
        static_vars=static_vars,
        atmos_vars=atmos_vars,
        metadata=metadata
    )

def create_dataloader(
    atmos_vars: List[str] = None,
    surface_vars: List[str] = None,
    static_vars: List[str] = None,
    start_date: str = "2018-01-01",
    end_date: str = "2018-01-07",
    batch_size: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    **kwargs
):
    """
    Создает DataLoader для ERA5 данных.
    """
    # Создаем датасет
    dataset = ERA5Dataset(
        atmos_vars = atmos_vars,
        surface_vars = surface_vars,
        static_vars = static_vars,
        start_date=start_date,
        end_date=end_date,
        **kwargs
    )
    
    # Создаем DataLoader с нашей collate функцией
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0, 
        pin_memory=pin_memory,
        collate_fn=collate_batches
    )
    
    return dataloader