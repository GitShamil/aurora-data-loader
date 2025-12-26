#!/usr/bin/env python3
"""
Минимальный маппинг названий переменных из WeatherBench2 в Aurora.
"""

class VariableMapper:
    """Класс для преобразования названий переменных в формат Aurora."""
    
    # Маппинг для поверхностных переменных: WB2 -> Aurora
    SURFACE_MAP = {
        '2m_temperature': '2t',
        '10m_u_component_of_wind': '10u',
        '10m_v_component_of_wind': '10v',
        'mean_sea_level_pressure': 'msl',
    }
    
    # Маппинг для статических переменных: WB2 -> Aurora
    STATIC_MAP = {
        'geopotential_at_surface': 'z',
        'land_sea_mask': 'lsm',
        'soil_type': 'slt',
    }
    
    # Маппинг для атмосферных переменных (без уровней): WB2 -> Aurora
    ATMOS_MAP = {
        'temperature': 't',
        'u_component_of_wind': 'u',
        'v_component_of_wind': 'v',
        'specific_humidity': 'q',
        'geopotential': 'z',
    }
    
    @staticmethod
    def map_surface(wb2_name: str) -> str:
        """Маппинг названий поверхностных переменных."""
        return VariableMapper.SURFACE_MAP.get(wb2_name, wb2_name)
    
    @staticmethod
    def map_static(wb2_name: str) -> str:
        """Маппинг названий статических переменных."""
        return VariableMapper.STATIC_MAP.get(wb2_name, wb2_name)
    
    @staticmethod
    def map_atmospheric(wb2_name: str, level: float) -> str:
        """Маппинг атмосферных переменных с указанием уровня."""
        base = VariableMapper.ATMOS_MAP.get(wb2_name, wb2_name)
        level_str = str(int(level)) if level.is_integer() else str(level).replace('.', '_')
        return f"{base}_{level_str}"