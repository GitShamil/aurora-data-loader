import torch
import sys
import os
from datetime import timedelta

# Добавляем путь к твоим модулям
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.model.aurora import AuroraAE
from src.dataloader.dataloader import create_dataloader
from src.dataloader.batch import Batch

def test_aurora_ae():
    """Тест AuroraAE модели на реальном батче данных."""
    
    # 1. Создаем DataLoader
    print("\n1. Создаем DataLoader...")
    atmos_vars = [
        "temperature",
        "u_component_of_wind", 
        "v_component_of_wind",
        "specific_humidity",
        "geopotential",
    ]
    surface_vars = [
        "2m_temperature",
        "10m_u_component_of_wind",
        "10m_v_component_of_wind", 
        "mean_sea_level_pressure",
    ]
    static_vars = ["geopotential_at_surface", "land_sea_mask", "soil_type"]
    
    dataloader = create_dataloader(
        atmos_vars=atmos_vars,
        surface_vars=surface_vars,
        static_vars=static_vars,
        start_date='2021-05-03',
        end_date='2021-05-04',  # Меньше дат для быстрого теста
        batch_size=3,
        shuffle=False
    )
    
    # 2. Загружаем батч
    print("\n2. Загружаем батч данных...")
    batch_saved = None
    for batch in dataloader:
        batch_saved = batch
        break
    
    if batch_saved is None:
        raise ValueError("Не удалось загрузить батч данных!")
    
    # Проверяем переменные
    print("\n   Переменные в батче:")
    print(f"   - surf_vars: {list(batch_saved.surf_vars.keys())}")
    for name, tensor in batch_saved.surf_vars.items():
        print(f"     {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    print(f"\n   - atmos_vars: {list(batch_saved.atmos_vars.keys())}")
    for name, tensor in batch_saved.atmos_vars.items():
        print(f"     {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    print(f"\n   - static_vars: {list(batch_saved.static_vars.keys())}")
    for name, tensor in batch_saved.static_vars.items():
        print(f"     {name}: shape={tensor.shape}, dtype={tensor.dtype}")
    
    # 4. Создаем модель AuroraAE
    print("\n4. Создаем модель AuroraAE...")
    # Маппинг твоих переменных к стандартным Aurora
    # Aurora ожидает: surf_vars = ("2t", "10u", "10v", "msl")
    # Твои переменные: surface_vars = ["2m_temperature", "10m_u_component_of_wind", ...]
    
    # Для теста используем стандартные имена Aurora
    model = AuroraAE(
        surf_vars=("2t", "10u", "10v", "msl"),  # Стандартные имена Aurora
        static_vars=("lsm", "z", "slt"),        # Стандартные имена Aurora  
        atmos_vars=("z", "u", "v", "t", "q"),   # Стандартные имена Aurora
        patch_size=4,
        embed_dim=64,  # Уменьшаем для быстрого теста
        num_heads=4,   # Уменьшаем для быстрого теста
        latent_levels=4,
        enc_depth=1,
        dec_depth=1,
    )
    
    # 5. Переводим модель в eval mode и на GPU если есть
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n5. Переводим модель на {device}...")
    model = model.to(device)
    model.eval()

    # 7. Forward pass
    print("\n7. Запускаем forward pass...")
    with torch.no_grad():
        batch_saved = batch_saved.to(device)
        output = model(batch_saved)
        
    print("   ✅ Forward pass успешен!")
        
    # 8. Проверяем выход
    print("\n8. Проверяем выход модели:")
    print(f"   Output type: {type(output)}")
    print(f"   Output spatial shape: {output.spatial_shape}")
    
    print(f"\n   Выходные переменные:")
    print(f"   - surf_vars: {list(output.surf_vars.keys())}")
    for name, tensor in output.surf_vars.items():
        print(f"     {name}: shape={tensor.shape}, min={tensor.min():.3f}, max={tensor.max():.3f}")
    
    print(f"\n   - atmos_vars: {list(output.atmos_vars.keys())}")
    for name, tensor in output.atmos_vars.items():
        print(f"     {name}: shape={tensor.shape}, min={tensor.min():.3f}, max={tensor.max():.3f}")
    
    # 9. Проверяем, что размеры совпадают
    print("\n9. Проверка размерностей:")
    all_ok = True
    
    for var_name in batch_saved.surf_vars:
        if var_name in output.surf_vars:
            input_shape = batch_saved.surf_vars[var_name].shape
            output_shape = output.surf_vars[var_name].shape
            if input_shape == output_shape:
                print(f"   ✅ {var_name}: {input_shape} == {output_shape}")
            else:
                print(f"   ❌ {var_name}: {input_shape} != {output_shape}")
                all_ok = False
        
        if all_ok:
            print("\n" + "=" * 80)
            print("✅ ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠️  Есть проблемы с размерностями!")
            print("=" * 80)
            

if __name__ == "__main__":
    print("Запуск тестов AuroraAE...")
    test_aurora_ae()
    