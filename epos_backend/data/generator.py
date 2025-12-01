"""
Генератор синтетических данных для системы ЭПОС
Разработано командой TechLitCode
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import random
from pathlib import Path
from config.settings import EPOSConfig
from config.equipment_profiles import EquipmentProfile
from utils.logger import log_info, log_error, log_warning, log_debug
from utils.helpers import generate_timestamps, add_noise, normalize_data, calculate_statistics
from utils import constants


class DataGenerator:
    """Генератор синтетических данных для различных типов оборудования"""

    def __init__(self, seed: int = 42):
        """
        Инициализация генератора данных

        Args:
            seed: Сид для воспроизводимости результатов
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        log_info(f"Инициализация генератора данных с seed={seed}")
        log_info(f"Разработано командой {EPOSConfig.DEVELOPER}")

    def generate_load_profile(self,
                              equipment_type: str,
                              hours: int = 24,
                              pattern: str = "industrial",
                              noise_level: float = 0.05) -> List[float]:
        """
        Генерация профиля нагрузки для оборудования

        Args:
            equipment_type: Тип оборудования (compressor, pump, oven, etc.)
            hours: Количество часов
            pattern: Паттерн нагрузки (industrial, continuous, shift, random)
            noise_level: Уровень шума

        Returns:
            Список значений нагрузки в кВт
        """
        log_info(f"Генерация профиля нагрузки для {equipment_type} ({pattern})")

        # Базовые профили в зависимости от типа оборудования
        base_profiles = {
            "compressor": {
                "min": 50,
                "max": 200,
                "pattern_weights": {"industrial": 0.7, "continuous": 0.2, "shift": 0.1}
            },
            "pump": {
                "min": 30,
                "max": 150,
                "pattern_weights": {"continuous": 0.6, "industrial": 0.3, "shift": 0.1}
            },
            "oven": {
                "min": 100,
                "max": 500,
                "pattern_weights": {"shift": 0.8, "industrial": 0.2}
            },
            "ventilation": {
                "min": 20,
                "max": 80,
                "pattern_weights": {"continuous": 0.9, "industrial": 0.1}
            },
            "conveyor": {
                "min": 15,
                "max": 45,
                "pattern_weights": {"shift": 0.7, "continuous": 0.3}
            }
        }

        if equipment_type not in base_profiles:
            log_warning(f"Неизвестный тип оборудования {equipment_type}, используется compressor")
            equipment_type = "compressor"

        profile = base_profiles[equipment_type]
        min_power = profile["min"]
        max_power = profile["max"]

        # Генерация базового паттерна
        if pattern == "industrial":
            # Промышленный паттерн: пики днем, снижение ночью
            base_pattern = self._generate_industrial_pattern(hours)
        elif pattern == "continuous":
            # Непрерывный режим с небольшими колебаниями
            base_pattern = self._generate_continuous_pattern(hours)
        elif pattern == "shift":
            # Сменный график работы
            base_pattern = self._generate_shift_pattern(hours)
        elif pattern == "random":
            # Случайные колебания
            base_pattern = self._generate_random_pattern(hours)
        else:
            log_warning(f"Неизвестный паттерн {pattern}, используется industrial")
            base_pattern = self._generate_industrial_pattern(hours)

        # Масштабирование к реальным значениям мощности
        scaled_load = []
        for value in base_pattern:
            power = min_power + (max_power - min_power) * value
            scaled_load.append(power)

        # Добавление шума
        if noise_level > 0:
            scaled_load = add_noise(scaled_load, noise_level, "percentage")

        # Валидация
        scaled_load = [max(min_power, min(max_power, x)) for x in scaled_load]

        # Логирование статистики
        stats = calculate_statistics(scaled_load)
        log_debug(f"Статистика нагрузки {equipment_type}: среднее={stats['mean']:.1f} кВт, "
                  f"макс={stats['max']:.1f} кВт, мин={stats['min']:.1f} кВт")

        return scaled_load

    def _generate_industrial_pattern(self, hours: int) -> List[float]:
        """Генерация промышленного паттерна нагрузки"""
        pattern = []
        for hour in range(hours):
            # Ночное время (0-6) - низкая нагрузка
            if 0 <= hour < 6:
                base = 0.2 + 0.1 * np.sin(hour * np.pi / 12)
            # Утренний подъем (6-9)
            elif 6 <= hour < 9:
                base = 0.3 + 0.4 * (hour - 6) / 3
            # Рабочий день (9-18) - высокая нагрузка
            elif 9 <= hour < 18:
                base = 0.7 + 0.2 * np.sin((hour - 12) * np.pi / 12)
            # Вечернее снижение (18-24)
            else:
                base = 0.5 - 0.3 * (hour - 18) / 6

            # Добавляем случайные всплески в рабочее время
            if 9 <= hour < 18 and np.random.random() < 0.3:
                base += np.random.uniform(0.1, 0.3)

            pattern.append(max(0.1, min(1.0, base)))

        return pattern

    def _generate_continuous_pattern(self, hours: int) -> List[float]:
        """Генерация непрерывного паттерна нагрузки"""
        # Базовый уровень с медленными изменениями
        base_level = 0.6
        pattern = []

        for hour in range(hours):
            # Медленные синусоидальные изменения
            slow_var = 0.2 * np.sin(hour * np.pi / 24)
            # Быстрые случайные колебания
            fast_var = 0.1 * np.random.randn()

            value = base_level + slow_var + fast_var
            pattern.append(max(0.3, min(0.9, value)))

        return pattern

    def _generate_shift_pattern(self, hours: int) -> List[float]:
        """Генерация сменного паттерна нагрузки"""
        pattern = []

        for hour in range(hours):
            # Первая смена (8-16) - полная нагрузка
            if 8 <= hour < 16:
                base = 0.8 + 0.1 * np.random.random()
            # Вторая смена (16-24) - средняя нагрузка
            elif 16 <= hour < 24:
                base = 0.6 + 0.2 * np.random.random()
            # Третья смена (0-8) - низкая нагрузка
            else:
                base = 0.3 + 0.2 * np.random.random()

            pattern.append(max(0.1, min(1.0, base)))

        return pattern

    def _generate_random_pattern(self, hours: int) -> List[float]:
        """Генерация случайного паттерна нагрузки"""
        # Используем случайное блуждание
        pattern = [0.5]
        for _ in range(hours - 1):
            step = pattern[-1] + np.random.randn() * 0.1
            pattern.append(max(0.1, min(1.0, step)))

        return pattern

    def generate_price_profile(self,
                               hours: int = 24,
                               base_price: float = 4.0,
                               variability: float = 0.3,
                               include_peak_pricing: bool = True) -> List[float]:
        """
        Генерация профиля цен на электроэнергию

        Args:
            hours: Количество часов
            base_price: Базовая цена за кВт*ч в рублях
            variability: Волатильность цен (0-1)
            include_peak_pricing: Включать ли пиковые цены

        Returns:
            Список цен по часам
        """
        log_info(f"Генерация профиля цен (база={base_price} руб/кВт*ч, волатильность={variability})")

        prices = []

        for hour in range(hours):
            # Базовое значение
            price = base_price

            # Суточные колебания
            daily_factor = 1.0 + 0.3 * np.sin((hour - 14) * np.pi / 12)

            # Пиковые часы (9-21)
            if include_peak_pricing and 9 <= hour < 21:
                peak_factor = 1.0 + (hour - 9) / 12 * 0.5
                daily_factor *= peak_factor

            # Максимальный пик в 14-16 часов
            if include_peak_pricing and 14 <= hour < 16:
                daily_factor *= 1.3

            # Ночное снижение (0-6)
            if 0 <= hour < 6:
                daily_factor *= 0.6

            # Выходные (суббота и воскресенье)
            # Здесь упрощенно - можно добавить детализацию по дням
            weekend_factor = 0.9 if (hour // 24) % 7 in [5, 6] else 1.0

            # Сезонные колебания (упрощенно)
            seasonal_factor = 1.0  # Можно добавить логику по месяцам

            # Случайная составляющая
            random_factor = 1.0 + variability * np.random.randn() * 0.1

            # Итоговая цена
            price = price * daily_factor * weekend_factor * seasonal_factor * random_factor

            # Ограничения
            price = max(base_price * 0.3, min(base_price * 3.0, price))
            prices.append(round(price, 2))

        # Сглаживание цен
        window_size = 3
        if len(prices) >= window_size:
            smoothed = []
            for i in range(len(prices)):
                start = max(0, i - window_size // 2)
                end = min(len(prices), i + window_size // 2 + 1)
                window = prices[start:end]
                smoothed.append(round(np.mean(window), 2))
            prices = smoothed

        stats = calculate_statistics(prices)
        log_debug(f"Статистика цен: среднее={stats['mean']:.2f} руб, "
                  f"макс={stats['max']:.2f} руб, мин={stats['min']:.2f} руб")

        return prices

    def generate_weather_data(self,
                              hours: int = 24,
                              base_temp: float = 20.0) -> Dict[str, List[float]]:
        """
        Генерация данных о погоде

        Args:
            hours: Количество часов
            base_temp: Базовая температура

        Returns:
            Словарь с данными о погоде
        """
        log_info(f"Генерация данных о погоде (базовая температура={base_temp}°C)")

        temperatures = []
        humidities = []
        solar_radiation = []  # Вт/м²

        for hour in range(hours):
            # Суточный ход температуры
            temp = base_temp + 8 * np.sin((hour - 14) * np.pi / 24)

            # Случайные колебания
            temp += np.random.randn() * 2

            temperatures.append(round(temp, 1))

            # Влажность обратно пропорциональна температуре
            humidity = 60 - 0.5 * (temp - base_temp)
            humidity += np.random.randn() * 10
            humidity = max(30, min(90, humidity))
            humidities.append(round(humidity, 1))

            # Солнечная радиация (днем выше)
            if 6 <= hour < 18:
                radiation = 300 + 400 * np.sin((hour - 6) * np.pi / 12)
            else:
                radiation = 0

            radiation += np.random.randn() * 50
            radiation = max(0, radiation)
            solar_radiation.append(round(radiation, 1))

        weather_data = {
            "temperature_c": temperatures,
            "humidity_percent": humidities,
            "solar_radiation_wm2": solar_radiation,
            "timestamp": [f"Hour_{i:02d}" for i in range(hours)]
        }

        return weather_data

    def generate_production_schedule(self,
                                     equipment_profile: EquipmentProfile,
                                     hours: int = 24,
                                     production_target: Optional[float] = None) -> Dict[str, Any]:
        """
        Генерация производственного расписания

        Args:
            equipment_profile: Профиль оборудования
            hours: Количество часов
            production_target: Целевой объем производства

        Returns:
            Словарь с производственным расписанием
        """
        log_info(f"Генерация производственного расписания для {equipment_profile.name}")

        if production_target is None:
            # Автоматический расчет целевого производства
            avg_power = (equipment_profile.power_min + equipment_profile.power_max) / 2
            production_target = avg_power * hours * 0.7  # 70% загрузки

        # Генерация графика работы
        schedule = []
        production_achieved = 0

        hour = 0
        while hour < hours and production_achieved < production_target:
            # Решение: работать или нет в этот час
            if hour < 8 or hour >= 20:
                # Ночное время - низкая вероятность работы
                work_probability = 0.3
            else:
                # Рабочее время - высокая вероятность
                work_probability = 0.8

            if np.random.random() < work_probability and production_achieved < production_target:
                # Оборудование работает
                load_factor = np.random.uniform(0.6, 1.0)
                power = equipment_profile.power_min + (
                            equipment_profile.power_max - equipment_profile.power_min) * load_factor
                production = power * equipment_profile.efficiency

                # Учет минимального времени работы
                min_hours = max(1, equipment_profile.min_on_time)
                for _ in range(min_hours):
                    if hour < hours:
                        schedule.append({
                            "hour": hour,
                            "status": "on",
                            "power_kw": round(power, 2),
                            "load_factor": round(load_factor, 2),
                            "production": round(production, 2)
                        })
                        production_achieved += production
                        hour += 1

                # Минимальное время простоя
                min_off = equipment_profile.min_off_time
                for _ in range(min_off):
                    if hour < hours:
                        schedule.append({
                            "hour": hour,
                            "status": "off",
                            "power_kw": 0,
                            "load_factor": 0,
                            "production": 0
                        })
                        hour += 1
            else:
                # Оборудование не работает
                schedule.append({
                    "hour": hour,
                    "status": "off",
                    "power_kw": 0,
                    "load_factor": 0,
                    "production": 0
                })
                hour += 1

        # Заполняем оставшиеся часы
        while hour < hours:
            schedule.append({
                "hour": hour,
                "status": "off",
                "power_kw": 0,
                "load_factor": 0,
                "production": 0
            })
            hour += 1

        # Сортируем по часам
        schedule.sort(key=lambda x: x["hour"])

        result = {
            "equipment_name": equipment_profile.name,
            "production_target": production_target,
            "production_achieved": round(production_achieved, 2),
            "achievement_percent": round(production_achieved / production_target * 100,
                                         1) if production_target > 0 else 0,
            "schedule": schedule,
            "total_hours_on": sum(1 for s in schedule if s["status"] == "on"),
            "total_energy_kwh": sum(s["power_kw"] for s in schedule),
            "efficiency_avg": equipment_profile.efficiency
        }

        log_debug(f"Производственное расписание: цель={production_target:.1f}, "
                  f"достигнуто={production_achieved:.1f} ({result['achievement_percent']:.1f}%)")

        return result

    def generate_complete_dataset(self,
                                  equipment_type: str = "compressor",
                                  hours: int = 24,
                                  save_to_file: bool = True) -> Dict[str, Any]:
        """
        Генерация полного набора данных для анализа

        Args:
            equipment_type: Тип оборудования
            hours: Количество часов
            save_to_file: Сохранять ли данные в файл

        Returns:
            Полный набор данных
        """
        log_info(f"Генерация полного набора данных для {equipment_type} на {hours} часов")

        # Генерация временных меток
        timestamps = generate_timestamps(hours)

        # Генерация профиля нагрузки
        load_profile = self.generate_load_profile(
            equipment_type=equipment_type,
            hours=hours,
            pattern="industrial",
            noise_level=0.05
        )

        # Генерация профиля цен
        price_profile = self.generate_price_profile(
            hours=hours,
            base_price=4.0,
            variability=0.2,
            include_peak_pricing=True
        )

        # Генерация данных о погоде
        weather_data = self.generate_weather_data(hours=hours)

        # Расчет стоимости энергии
        energy_cost = [load * price for load, price in zip(load_profile, price_profile)]

        # Сборка полного набора данных
        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "equipment_type": equipment_type,
                "hours": hours,
                "seed": self.seed,
                "system": EPOSConfig.SYSTEM_NAME,
                "developer": EPOSConfig.DEVELOPER,
                "version": EPOSConfig.VERSION
            },
            "timestamps": [ts.isoformat() for ts in timestamps],
            "hour_of_day": list(range(hours)),
            "load_profile_kw": load_profile,
            "price_profile_rub_kwh": price_profile,
            "energy_cost_rub": energy_cost,
            "weather": weather_data,
            "statistics": {
                "load": calculate_statistics(load_profile),
                "price": calculate_statistics(price_profile),
                "cost": calculate_statistics(energy_cost)
            }
        }

        # Сохранение в файл
        if save_to_file:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"dataset_{equipment_type}_{hours}h_{timestamp_str}.json"

            import json
            from pathlib import Path

            output_dir = EPOSConfig.OUTPUTS_DIR / "json"
            output_dir.mkdir(exist_ok=True)

            filepath = output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)

            log_info(f"Набор данных сохранен: {filepath}")
            dataset["saved_to"] = str(filepath)

        return dataset

    def generate_multiple_equipment_dataset(self,
                                            equipment_types: List[str] = None,
                                            hours: int = 24,
                                            correlation: float = 0.7) -> Dict[str, Any]:
        """
        Генерация данных для нескольких единиц оборудования

        Args:
            equipment_types: Список типов оборудования
            hours: Количество часов
            correlation: Корреляция между профилями нагрузки

        Returns:
            Набор данных для нескольких единиц оборудования
        """
        if equipment_types is None:
            equipment_types = ["compressor", "pump", "ventilation"]

        log_info(f"Генерация данных для {len(equipment_types)} единиц оборудования "
                 f"с корреляцией {correlation}")

        # Генерация базового профиля
        base_profile = self.generate_load_profile(
            equipment_type="compressor",
            hours=hours,
            pattern="industrial",
            noise_level=0.05
        )

        # Нормализация базового профиля
        base_norm = normalize_data(base_profile, "minmax")

        # Генерация профилей для каждого оборудования
        equipment_data = {}

        for i, eq_type in enumerate(equipment_types):
            # Создаем коррелированный профиль
            if i == 0:
                # Первое оборудование использует базовый профиль
                correlated = base_norm
            else:
                # Смешиваем с случайным профилем для заданной корреляции
                random_profile = normalize_data(
                    self.generate_load_profile(eq_type, hours, "random", 0.1),
                    "minmax"
                )

                # Линейная комбинация для достижения нужной корреляции
                correlated = []
                for b, r in zip(base_norm, random_profile):
                    value = correlation * b + (1 - correlation) * r
                    correlated.append(value)

            # Денормализация к мощности оборудования
            profile_info = {
                "compressor": (50, 200),
                "pump": (30, 150),
                "oven": (100, 500),
                "ventilation": (20, 80),
                "conveyor": (15, 45)
            }

            min_power, max_power = profile_info.get(eq_type, (50, 200))
            load_profile = [min_power + (max_power - min_power) * val for val in correlated]

            equipment_data[eq_type] = {
                "load_profile_kw": load_profile,
                "min_power": min_power,
                "max_power": max_power,
                "statistics": calculate_statistics(load_profile)
            }

        # Генерация общих данных
        price_profile = self.generate_price_profile(hours=hours)
        weather_data = self.generate_weather_data(hours=hours)

        # Расчет суммарной нагрузки
        total_load = [0] * hours
        for eq_data in equipment_data.values():
            for i, load in enumerate(eq_data["load_profile_kw"]):
                total_load[i] += load

        dataset = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "equipment_count": len(equipment_types),
                "equipment_types": equipment_types,
                "hours": hours,
                "correlation": correlation,
                "system": EPOSConfig.SYSTEM_NAME
            },
            "hour_of_day": list(range(hours)),
            "price_profile_rub_kwh": price_profile,
            "weather": weather_data,
            "equipment_data": equipment_data,
            "total_load_kw": total_load,
            "total_cost_rub": [
                total_load[i] * price_profile[i] for i in range(hours)
            ]
        }

        return dataset


# Утилитарные функции для быстрого доступа
def generate_quick_dataset(equipment_type: str = "compressor",
                           hours: int = 24) -> pd.DataFrame:
    """Быстрая генерация датасета в формате DataFrame"""
    generator = DataGenerator()

    # Генерация данных
    load = generator.generate_load_profile(equipment_type, hours)
    prices = generator.generate_price_profile(hours)

    # Создание DataFrame
    df = pd.DataFrame({
        'hour': list(range(hours)),
        'load_kw': load,
        'price_rub_kwh': prices,
        'cost_rub': [l * p for l, p in zip(load, prices)]
    })

    # Добавление временных меток
    timestamps = generate_timestamps(hours)
    df['timestamp'] = [ts.strftime('%Y-%m-%d %H:%M') for ts in timestamps]

    return df


def save_dataset_to_csv(dataset: Dict[str, Any],
                        filename: str = None) -> Path:
    """Сохранение набора данных в CSV"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eq_type = dataset.get('metadata', {}).get('equipment_type', 'unknown')
        filename = f"dataset_{eq_type}_{timestamp}.csv"

    output_dir = EPOSConfig.REPORTS_DIR / "csv"
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / filename

    # Конвертация в DataFrame
    df_data = {
        'hour': dataset.get('hour_of_day', []),
        'load_kw': dataset.get('load_profile_kw', []),
        'price_rub_kwh': dataset.get('price_profile_rub_kwh', []),
        'cost_rub': dataset.get('energy_cost_rub', [])
    }

    if 'weather' in dataset:
        weather = dataset['weather']
        df_data['temperature_c'] = weather.get('temperature_c', [])
        df_data['humidity_percent'] = weather.get('humidity_percent', [])
        df_data['solar_radiation'] = weather.get('solar_radiation_wm2', [])

    df = pd.DataFrame(df_data)
    df.to_csv(filepath, index=False, encoding='utf-8')

    log_info(f"Набор данных сохранен в CSV: {filepath}")
    return filepath