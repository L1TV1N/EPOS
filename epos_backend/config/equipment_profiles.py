"""
Профили оборудования для системы ЭПОС
Разработано командой TechLitCode
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import time
import json


@dataclass
class EquipmentProfile:
    """Профиль оборудования для оптимизации"""
    name: str
    equipment_type: str  # compressor, pump, oven, etc.
    power_nominal: float  # кВт, номинальная мощность
    power_min: float  # кВт, минимальная мощность
    power_max: float  # кВт, максимальная мощность
    min_on_time: int  # часов, минимальное время работы
    min_off_time: int  # часов, минимальное время простоя
    startup_time: float  # часов, время запуска
    shutdown_time: float  # часов, время остановки
    efficiency: float  # КПД
    ramp_rate: float = 100  # %/час, скорость изменения нагрузки
    maintenance_windows: List[Dict] = None  # Окна техобслуживания

    # Целевые параметры
    target_production: Optional[float] = None  # Целевой объем производства
    production_rate: Optional[float] = None  # Производительность на кВт*ч

    # График работы по умолчанию
    default_schedule: Dict[str, List[int]] = None

    def __post_init__(self):
        if self.maintenance_windows is None:
            self.maintenance_windows = []
        if self.default_schedule is None:
            self.default_schedule = {"work_hours": list(range(8, 18))}

    def validate(self) -> List[str]:
        """Валидация параметров оборудования"""
        errors = []

        if self.power_min > self.power_max:
            errors.append("Минимальная мощность не может быть больше максимальной")

        if self.power_nominal > self.power_max:
            errors.append("Номинальная мощность не может быть больше максимальной")

        if self.efficiency <= 0 or self.efficiency > 1:
            errors.append(f"Некорректный КПД: {self.efficiency}")

        if self.min_on_time < 0:
            errors.append("Минимальное время работы не может быть отрицательным")

        return errors

    def to_dict(self) -> Dict:
        """Конвертация в словарь"""
        return {
            "name": self.name,
            "equipment_type": self.equipment_type,
            "power_nominal": self.power_nominal,
            "power_min": self.power_min,
            "power_max": self.power_max,
            "min_on_time": self.min_on_time,
            "min_off_time": self.min_off_time,
            "startup_time": self.startup_time,
            "shutdown_time": self.shutdown_time,
            "efficiency": self.efficiency,
            "ramp_rate": self.ramp_rate,
            "maintenance_windows": self.maintenance_windows,
            "target_production": self.target_production,
            "production_rate": self.production_rate,
            "default_schedule": self.default_schedule
        }

    def save_profile(self, filename: str = None):
        """Сохранение профиля в файл"""
        if filename is None:
            filename = f"equipment_{self.name.replace(' ', '_')}.json"

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

        return filename


class EquipmentManager:
    """Менеджер профилей оборудования"""

    # Стандартные профили оборудования
    STANDARD_PROFILES = {
        "compressor_air": EquipmentProfile(
            name="Воздушный компрессор",
            equipment_type="compressor",
            power_nominal=200,
            power_min=50,
            power_max=200,
            min_on_time=2,
            min_off_time=1,
            startup_time=0.5,
            shutdown_time=0.25,
            efficiency=0.85,
            ramp_rate=80
        ),

        "pump_water_cooling": EquipmentProfile(
            name="Насос системы охлаждения",
            equipment_type="pump",
            power_nominal=150,
            power_min=30,
            power_max=150,
            min_on_time=1,
            min_off_time=0.5,
            startup_time=0.25,
            shutdown_time=0.25,
            efficiency=0.88,
            ramp_rate=100
        ),

        "oven_induction": EquipmentProfile(
            name="Индукционная печь",
            equipment_type="oven",
            power_nominal=500,
            power_min=100,
            power_max=500,
            min_on_time=4,
            min_off_time=2,
            startup_time=1,
            shutdown_time=0.5,
            efficiency=0.75,
            ramp_rate=50
        ),

        "ventilation_main": EquipmentProfile(
            name="Главная вентиляция",
            equipment_type="ventilation",
            power_nominal=80,
            power_min=20,
            power_max=80,
            min_on_time=0.5,
            min_off_time=0.25,
            startup_time=0.1,
            shutdown_time=0.1,
            efficiency=0.92,
            ramp_rate=150
        ),

        "conveyor_belt": EquipmentProfile(
            name="Конвейерная лента",
            equipment_type="conveyor",
            power_nominal=45,
            power_min=15,
            power_max=45,
            min_on_time=0.5,
            min_off_time=0.25,
            startup_time=0.2,
            shutdown_time=0.2,
            efficiency=0.95,
            ramp_rate=200
        )
    }

    @classmethod
    def get_profile(cls, profile_name: str) -> EquipmentProfile:
        """Получить стандартный профиль по имени"""
        if profile_name in cls.STANDARD_PROFILES:
            return cls.STANDARD_PROFILES[profile_name]
        else:
            raise ValueError(f"Профиль '{profile_name}' не найден. Доступные: {list(cls.STANDARD_PROFILES.keys())}")

    @classmethod
    def list_profiles(cls) -> Dict[str, Dict]:
        """Список всех стандартных профилей"""
        return {name: profile.to_dict() for name, profile in cls.STANDARD_PROFILES.items()}

    @classmethod
    def create_custom_profile(cls, **kwargs) -> EquipmentProfile:
        """Создать кастомный профиль оборудования"""
        required_fields = [
            'name', 'equipment_type', 'power_nominal',
            'power_min', 'power_max', 'efficiency'
        ]

        for field in required_fields:
            if field not in kwargs:
                raise ValueError(f"Необходимо указать поле: {field}")

        return EquipmentProfile(**kwargs)

    @classmethod
    def validate_all_profiles(cls):
        """Валидация всех стандартных профилей"""
        results = {}
        for name, profile in cls.STANDARD_PROFILES.items():
            errors = profile.validate()
            results[name] = {
                "valid": len(errors) == 0,
                "errors": errors
            }
        return results


# Утилиты для работы с оборудованием
def calculate_energy_consumption(profile: EquipmentProfile,
                                 hours: int,
                                 load_factor: float = 1.0) -> float:
    """Расчет потребления энергии"""
    if load_factor < 0 or load_factor > 1:
        raise ValueError("Коэффициент нагрузки должен быть между 0 и 1")

    power = profile.power_min + (profile.power_max - profile.power_min) * load_factor
    energy = power * hours / profile.efficiency
    return energy


def estimate_cost_saving(profile: EquipmentProfile,
                         old_schedule: List[float],
                         new_schedule: List[float],
                         prices: List[float]) -> Dict:
    """Оценка экономии от оптимизации"""
    old_cost = sum(p * l for p, l in zip(prices, old_schedule))
    new_cost = sum(p * l for p, l in zip(prices, new_schedule))

    saving_abs = old_cost - new_cost
    saving_percent = (saving_abs / old_cost * 100) if old_cost > 0 else 0

    return {
        "equipment": profile.name,
        "old_cost_rub": round(old_cost, 2),
        "new_cost_rub": round(new_cost, 2),
        "saving_rub": round(saving_abs, 2),
        "saving_percent": round(saving_percent, 2),
        "roi_days": None  # Можно рассчитать если знаем стоимость системы
    }