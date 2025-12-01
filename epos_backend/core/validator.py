"""
Валидатор данных и ограничений системы ЭПОС
Разработано командой TechLitCode
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

from config.settings import EPOSConfig
from config.equipment_profiles import EquipmentProfile
from utils.logger import log_info, log_error, log_warning, log_debug
from utils.helpers import calculate_statistics, generate_timestamps
from utils import constants


class ValidationLevel(Enum):
    """Уровни валидации"""
    BASIC = "basic"  # Базовая проверка
    STRICT = "strict"  # Строгая проверка
    FULL = "full"  # Полная проверка


class ValidationStatus(Enum):
    """Статусы валидации"""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    ERROR = "error"


@dataclass
class ValidationResult:
    """Результат валидации"""
    status: ValidationStatus
    level: ValidationLevel
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    passed_checks: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Проверка, что валидация пройдена"""
        return self.status in [ValidationStatus.VALID, ValidationStatus.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'status': self.status.value,
            'level': self.level.value,
            'errors': self.errors,
            'warnings': self.warnings,
            'passed_checks': self.passed_checks,
            'metadata': self.metadata,
            'is_valid': self.is_valid(),
            'timestamp': datetime.now().isoformat()
        }

    def print_summary(self):
        """Печать сводки результатов"""
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ")
        print("=" * 60)
        print(f"Статус: {self.status.value}")
        print(f"Уровень: {self.level.value}")

        if self.passed_checks:
            print(f"\nПройдено проверок: {len(self.passed_checks)}")

        if self.warnings:
            print(f"\nПредупреждения ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        if self.errors:
            print(f"\nОшибки ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        print(f"\nВалидация {'ПРОЙДЕНА' if self.is_valid() else 'НЕ ПРОЙДЕНА'}")


class EPOSValidator:
    """Валидатор системы ЭПОС"""

    def __init__(self, level: ValidationLevel = ValidationLevel.BASIC):
        """
        Инициализация валидатора

        Args:
            level: Уровень валидации
        """
        self.level = level
        self.results: Dict[str, ValidationResult] = {}

        log_info(f"Инициализация валидатора ЭПОС (уровень: {level.value})")
        log_info(f"Разработано командой {EPOSConfig.DEVELOPER}")

    def validate_equipment_profile(self,
                                   profile: EquipmentProfile) -> ValidationResult:
        """
        Валидация профиля оборудования

        Args:
            profile: Профиль оборудования

        Returns:
            Результат валидации
        """
        log_info(f"Валидация профиля оборудования: {profile.name}")

        errors = []
        warnings = []
        passed = []

        # 1. Проверка основных параметров
        if profile.power_min <= 0:
            errors.append("Минимальная мощность должна быть положительной")
        else:
            passed.append("Минимальная мощность > 0")

        if profile.power_max <= 0:
            errors.append("Максимальная мощность должна быть положительной")
        else:
            passed.append("Максимальная мощность > 0")

        if profile.power_min > profile.power_max:
            errors.append(f"Минимальная мощность ({profile.power_min}) > максимальной ({profile.power_max})")
        else:
            passed.append("Минимальная мощность ≤ максимальной")

        if profile.power_nominal < profile.power_min or profile.power_nominal > profile.power_max:
            warnings.append(
                f"Номинальная мощность ({profile.power_nominal}) вне диапазона [{profile.power_min}, {profile.power_max}]")
        else:
            passed.append("Номинальная мощность в допустимом диапазоне")

        # 2. Проверка временных параметров
        if profile.min_on_time < 0:
            errors.append("Минимальное время работы не может быть отрицательным")
        else:
            passed.append("Минимальное время работы ≥ 0")

        if profile.min_off_time < 0:
            errors.append("Минимальное время простоя не может быть отрицательным")
        else:
            passed.append("Минимальное время простоя ≥ 0")

        if profile.startup_time < 0:
            errors.append("Время запуска не может быть отрицательным")
        else:
            passed.append("Время запуска ≥ 0")

        if profile.shutdown_time < 0:
            errors.append("Время остановки не может быть отрицательным")
        else:
            passed.append("Время остановки ≥ 0")

        # 3. Проверка КПД
        if profile.efficiency <= 0 or profile.efficiency > 1:
            errors.append(f"Некорректный КПД: {profile.efficiency}. Должен быть в диапазоне (0, 1]")
        else:
            passed.append("КПД в допустимом диапазоне (0, 1]")

        # 4. Проверка скорости изменения нагрузки
        if profile.ramp_rate <= 0 or profile.ramp_rate > 500:
            warnings.append(f"Скорость изменения нагрузки ({profile.ramp_rate}%) вне типичного диапазона [1, 500]%")
        else:
            passed.append("Скорость изменения нагрузки в разумных пределах")

        # 5. Дополнительные проверки для строгого уровня
        if self.level in [ValidationLevel.STRICT, ValidationLevel.FULL]:
            if profile.min_on_time == 0:
                warnings.append("Минимальное время работы = 0, оборудование может часто включаться/выключаться")

            if profile.min_off_time == 0:
                warnings.append("Минимальное время простоя = 0, оборудование может часто включаться/выключаться")

            if profile.efficiency < 0.5:
                warnings.append(f"Низкий КПД: {profile.efficiency}. Рассмотрите модернизацию оборудования")

        # Определение статуса
        status = self._determine_status(errors, warnings)

        result = ValidationResult(
            status=status,
            level=self.level,
            errors=errors,
            warnings=warnings,
            passed_checks=passed,
            metadata={
                'equipment_name': profile.name,
                'equipment_type': profile.equipment_type,
                'validation_time': datetime.now().isoformat()
            }
        )

        self.results[f"equipment_{profile.name}"] = result
        return result

    def validate_load_profile(self,
                              load_profile: List[float],
                              equipment_profile: Optional[EquipmentProfile] = None,
                              profile_name: str = "unknown") -> ValidationResult:
        """
        Валидация профиля нагрузки

        Args:
            load_profile: Профиль нагрузки (кВт)
            equipment_profile: Профиль оборудования (для проверки ограничений)
            profile_name: Имя профиля для идентификации

        Returns:
            Результат валидации
        """
        log_info(f"Валидация профиля нагрузки: {profile_name}")

        errors = []
        warnings = []
        passed = []

        # 1. Проверка базовых свойств
        if not load_profile:
            errors.append("Профиль нагрузки пуст")
            status = ValidationStatus.INVALID
            return ValidationResult(
                status=status,
                level=self.level,
                errors=errors,
                warnings=warnings,
                passed_checks=passed,
                metadata={'profile_name': profile_name}
            )

        passed.append("Профиль не пуст")

        # 2. Проверка длины профиля
        if len(load_profile) < 1:
            errors.append("Профиль нагрузки должен содержать хотя бы одно значение")
        else:
            passed.append(f"Длина профиля: {len(load_profile)} часов")

        # 3. Проверка на NaN и бесконечности
        load_array = np.array(load_profile)

        if np.any(np.isnan(load_array)):
            errors.append("Обнаружены значения NaN")
        else:
            passed.append("Отсутствуют значения NaN")

        if np.any(np.isinf(load_array)):
            errors.append("Обнаружены бесконечные значения")
        else:
            passed.append("Отсутствуют бесконечные значения")

        # 4. Проверка на отрицательные значения
        negative_mask = load_array < 0
        negative_count = np.sum(negative_mask)

        if negative_count > 0:
            errors.append(f"Обнаружено {negative_count} отрицательных значений нагрузки")
        else:
            passed.append("Все значения нагрузки неотрицательны")

        # 5. Статистический анализ
        stats = calculate_statistics(load_profile)

        if stats['min'] < 0:
            errors.append(f"Минимальная нагрузка отрицательная: {stats['min']:.2f} кВт")

        if stats['max'] == 0:
            warnings.append("Максимальная нагрузка равна 0")

        # 6. Проверка соответствия профилю оборудования
        if equipment_profile:
            # Проверка превышения максимальной мощности
            exceeding_mask = load_array > equipment_profile.power_max * 1.01  # 1% допуск
            exceeding_count = np.sum(exceeding_mask)

            if exceeding_count > 0:
                max_exceeding = np.max(load_array[exceeding_mask])
                errors.append(
                    f"Превышение максимальной мощности ({equipment_profile.power_max} кВт) "
                    f"в {exceeding_count} часах, максимум: {max_exceeding:.2f} кВт"
                )
            else:
                passed.append(f"Нагрузка не превышает максимальную мощность ({equipment_profile.power_max} кВт)")

            # Проверка минимальной мощности при работающем оборудовании
            # Считаем, что оборудование работает если нагрузка > 1% от максимальной
            working_mask = load_array > equipment_profile.power_max * 0.01
            under_min_mask = working_mask & (load_array < equipment_profile.power_min * 0.99)
            under_min_count = np.sum(under_min_mask)

            if under_min_count > 0:
                warnings.append(
                    f"Нагрузка ниже минимальной ({equipment_profile.power_min} кВт) "
                    f"в {under_min_count} часах при работающем оборудовании"
                )
            else:
                passed.append(f"Нагрузка соответствует минимальной мощности ({equipment_profile.power_min} кВт)")

        # 7. Проверка резких изменений (скачков)
        if len(load_profile) > 1:
            changes = np.diff(load_array)
            abs_changes = np.abs(changes)

            # Определяем скачки как изменения > 50% от среднего значения
            avg_load = np.mean(load_array)
            if avg_load > 0:
                spike_threshold = avg_load * 0.5
                spike_mask = abs_changes > spike_threshold
                spike_count = np.sum(spike_mask)

                if spike_count > 0:
                    warnings.append(f"Обнаружено {spike_count} резких изменений нагрузки (> {spike_threshold:.1f} кВт)")
                else:
                    passed.append("Резкие изменения нагрузки отсутствуют")

            # Проверка на постоянное значение
            if np.all(load_array == load_array[0]):
                warnings.append("Нагрузка постоянна на всем горизонте")

        # 8. Дополнительные проверки для строгого уровня
        if self.level in [ValidationLevel.STRICT, ValidationLevel.FULL]:
            # Проверка на нулевую дисперсию
            if stats['std'] == 0:
                warnings.append("Нулевая дисперсия нагрузки")

            # Проверка на выбросы
            if avg_load > 0:
                # Используем межквартильный размах для обнаружения выбросов
                q25 = stats['q25']
                q75 = stats['q75']
                iqr = q75 - q25

                if iqr > 0:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr

                    outlier_mask = (load_array < lower_bound) | (load_array > upper_bound)
                    outlier_count = np.sum(outlier_mask)

                    if outlier_count > 0:
                        warnings.append(f"Обнаружено {outlier_count} выбросов в данных нагрузки")

        # Определение статуса
        status = self._determine_status(errors, warnings)

        result = ValidationResult(
            status=status,
            level=self.level,
            errors=errors,
            warnings=warnings,
            passed_checks=passed,
            metadata={
                'profile_name': profile_name,
                'statistics': stats,
                'length': len(load_profile),
                'validation_time': datetime.now().isoformat()
            }
        )

        self.results[f"load_{profile_name}"] = result
        return result

    def validate_price_profile(self,
                               price_profile: List[float],
                               profile_name: str = "unknown") -> ValidationResult:
        """
        Валидация профиля цен

        Args:
            price_profile: Профиль цен (руб/кВт*ч)
            profile_name: Имя профиля для идентификации

        Returns:
            Результат валидации
        """
        log_info(f"Валидация профиля цен: {profile_name}")

        errors = []
        warnings = []
        passed = []

        # 1. Проверка базовых свойств
        if not price_profile:
            errors.append("Профиль цен пуст")
            status = ValidationStatus.INVALID
            return ValidationResult(
                status=status,
                level=self.level,
                errors=errors,
                warnings=warnings,
                passed_checks=passed,
                metadata={'profile_name': profile_name}
            )

        passed.append("Профиль не пуст")

        # 2. Проверка длины профиля
        if len(price_profile) < 1:
            errors.append("Профиль цен должен содержать хотя бы одно значение")
        else:
            passed.append(f"Длина профиля: {len(price_profile)} часов")

        # 3. Проверка на NaN и бесконечности
        price_array = np.array(price_profile)

        if np.any(np.isnan(price_array)):
            errors.append("Обнаружены значения NaN")
        else:
            passed.append("Отсутствуют значения NaN")

        if np.any(np.isinf(price_array)):
            errors.append("Обнаружены бесконечные значения")
        else:
            passed.append("Отсутствуют бесконечные значения")

        # 4. Проверка на отрицательные значения
        negative_mask = price_array < 0
        negative_count = np.sum(negative_mask)

        if negative_count > 0:
            errors.append(f"Обнаружено {negative_count} отрицательных цен")
        else:
            passed.append("Все цены неотрицательны")

        # 5. Статистический анализ
        stats = calculate_statistics(price_profile)

        if stats['min'] < 0:
            errors.append(f"Минимальная цена отрицательная: {stats['min']:.2f} руб/кВт*ч")

        # 6. Проверка реалистичности цен
        avg_price = stats['mean']

        if avg_price < 0.1:
            warnings.append(f"Средняя цена слишком низкая: {avg_price:.2f} руб/кВт*ч")
        elif avg_price > 20:
            warnings.append(f"Средняя цена слишком высокая: {avg_price:.2f} руб/кВт*ч")
        else:
            passed.append(f"Средняя цена в реалистичном диапазоне: {avg_price:.2f} руб/кВт*ч")

        # 7. Проверка резких изменений (скачков цен)
        if len(price_profile) > 1:
            changes = np.diff(price_array)
            abs_changes = np.abs(changes)

            # Определяем скачки как изменения > 100% от среднего значения
            if avg_price > 0:
                spike_threshold = avg_price * 1.0
                spike_mask = abs_changes > spike_threshold
                spike_count = np.sum(spike_mask)

                if spike_count > 0:
                    max_spike = np.max(abs_changes[spike_mask])
                    warnings.append(
                        f"Обнаружено {spike_count} резких изменений цен (> {spike_threshold:.2f} руб), максимум: {max_spike:.2f} руб")
                else:
                    passed.append("Резкие изменения цен отсутствуют")

        # 8. Проверка типичных паттернов цен
        if len(price_profile) >= 24:
            # Проверка суточного паттерна
            daily_prices = price_array[:24]
            day_hours = list(range(9, 21))  # 9:00 - 20:59
            night_hours = list(set(range(24)) - set(day_hours))

            day_avg = np.mean(daily_prices[day_hours]) if day_hours else 0
            night_avg = np.mean(daily_prices[night_hours]) if night_hours else 0

            if day_avg > 0 and night_avg > 0:
                day_night_ratio = day_avg / night_avg

                if day_night_ratio < 1.1:
                    warnings.append("Не выражен суточный паттерн: дневные и ночные цены почти равны")
                else:
                    passed.append(f"Выражен суточный паттерн: дневные цены в {day_night_ratio:.1f} раза выше ночных")

        # 9. Дополнительные проверки для строгого уровня
        if self.level in [ValidationLevel.STRICT, ValidationLevel.FULL]:
            # Проверка на нулевую дисперсию
            if stats['std'] == 0:
                warnings.append("Нулевая дисперсия цен")

            # Проверка на монотонность
            if len(price_profile) > 2:
                is_increasing = all(price_profile[i] <= price_profile[i + 1] for i in range(len(price_profile) - 1))
                is_decreasing = all(price_profile[i] >= price_profile[i + 1] for i in range(len(price_profile) - 1))

                if is_increasing or is_decreasing:
                    warnings.append("Цены монотонны (постоянно растут или падают)")

        # Определение статуса
        status = self._determine_status(errors, warnings)

        result = ValidationResult(
            status=status,
            level=self.level,
            errors=errors,
            warnings=warnings,
            passed_checks=passed,
            metadata={
                'profile_name': profile_name,
                'statistics': stats,
                'length': len(price_profile),
                'validation_time': datetime.now().isoformat()
            }
        )

        self.results[f"price_{profile_name}"] = result
        return result

    def validate_optimization_solution(self,
                                       solution: Dict[str, List[float]],
                                       equipment_profile: EquipmentProfile,
                                       prices: List[float],
                                       horizon: int) -> ValidationResult:
        """
        Валидация решения оптимизации

        Args:
            solution: Решение оптимизации
            equipment_profile: Профиль оборудования
            prices: Цены на электроэнергию
            horizon: Горизонт планирования

        Returns:
            Результат валидации
        """
        log_info(f"Валидация решения оптимизации для {equipment_profile.name}")

        errors = []
        warnings = []
        passed = []

        # 1. Проверка наличия необходимых ключей
        required_keys = ['power', 'on']
        for key in required_keys:
            if key not in solution:
                errors.append(f"Отсутствует ключ решения: {key}")
                continue
            else:
                passed.append(f"Ключ решения присутствует: {key}")

        if 'power' not in solution or 'on' not in solution:
            # Не можем продолжать без основных данных
            status = self._determine_status(errors, warnings)
            return ValidationResult(
                status=status,
                level=self.level,
                errors=errors,
                warnings=warnings,
                passed_checks=passed,
                metadata={
                    'equipment_name': equipment_profile.name,
                    'horizon': horizon
                }
            )

        power = solution['power']
        on_status = solution['on']

        # 2. Проверка длины решения
        if len(power) != horizon:
            errors.append(f"Длина решения по мощности ({len(power)}) не соответствует горизонту ({horizon})")
        else:
            passed.append(f"Длина решения соответствует горизонту: {horizon}")

        if len(on_status) != horizon:
            errors.append(f"Длина решения по статусу ({len(on_status)}) не соответствует горизонту ({horizon})")

        # 3. Проверка соответствия мощности и статуса
        for t in range(min(len(power), len(on_status))):
            # Если оборудование выключено, мощность должна быть 0 (с допуском)
            if on_status[t] < 0.5 and power[t] > 0.1:  # 0.1 кВт допуск
                errors.append(
                    f"Час {t}: мощность {power[t]:.2f} кВт при выключенном оборудовании (статус: {on_status[t]:.2f})")
                break
            # Если оборудование включено, мощность должна быть > 0
            elif on_status[t] > 0.5 and power[t] < 0.1:
                warnings.append(f"Час {t}: нулевая мощность при включенном оборудовании")
                break

        if len(power) == len(on_status):
            passed.append("Мощность и статус коррелируют")

        # 4. Проверка ограничений мощности
        for t, p in enumerate(power):
            if p < 0:
                errors.append(f"Час {t}: отрицательная мощность {p:.2f} кВт")
            elif p > equipment_profile.power_max * 1.01:  # 1% допуск
                errors.append(
                    f"Час {t}: превышение максимальной мощности {equipment_profile.power_max} кВт: {p:.2f} кВт")

            # Проверка минимальной мощности для работающего оборудования
            if on_status[t] > 0.5 and p < equipment_profile.power_min * 0.99:  # 1% допуск
                warnings.append(
                    f"Час {t}: мощность {p:.2f} кВт ниже минимальной {equipment_profile.power_min} кВт при работающем оборудовании")

        passed.append("Проверка ограничений мощности завершена")

        # 5. Проверка минимального времени работы
        if equipment_profile.min_on_time > 1:
            current_run = 0
            for t in range(len(on_status)):
                if on_status[t] > 0.5:
                    current_run += 1
                else:
                    if 0 < current_run < equipment_profile.min_on_time and current_run > 0:
                        # Проверяем, что это не начало горизонта
                        if t - current_run > 0:
                            warnings.append(
                                f"Час {t - current_run}-{t - 1}: работа {current_run} часов "
                                f"< минимального времени работы ({equipment_profile.min_on_time} часов)"
                            )
                    current_run = 0

            # Проверка последнего сегмента
            if 0 < current_run < equipment_profile.min_on_time:
                warnings.append(
                    f"Конец горизонта: работа {current_run} часов "
                    f"< минимального времени работы ({equipment_profile.min_on_time} часов)"
                )

        # 6. Проверка минимального времени простоя
        if equipment_profile.min_off_time > 1:
            current_off = 0
            for t in range(len(on_status)):
                if on_status[t] < 0.5:
                    current_off += 1
                else:
                    if 0 < current_off < equipment_profile.min_off_time and current_off > 0:
                        # Проверяем, что это не начало горизонта
                        if t - current_off > 0:
                            warnings.append(
                                f"Час {t - current_off}-{t - 1}: простой {current_off} часов "
                                f"< минимального времени простоя ({equipment_profile.min_off_time} часов)"
                            )
                    current_off = 0

            # Проверка последнего сегмента
            if 0 < current_off < equipment_profile.min_off_time:
                warnings.append(
                    f"Конец горизонта: простой {current_off} часов "
                    f"< минимального времени простоя ({equipment_profile.min_off_time} часов)"
                )

        # 7. Проверка ограничений на изменение нагрузки (ramping)
        if hasattr(equipment_profile, 'ramp_rate') and equipment_profile.ramp_rate < 100:
            ramp_limit = equipment_profile.ramp_rate / 100 * equipment_profile.power_max

            for t in range(1, len(power)):
                delta = abs(power[t] - power[t - 1])
                if delta > ramp_limit * 1.01:  # 1% допуск
                    warnings.append(
                        f"Час {t}: изменение нагрузки {delta:.2f} кВт "
                        f"> ограничения ({ramp_limit:.2f} кВт/час)"
                    )

        # 8. Проверка целевой функции (стоимости)
        if len(power) == len(prices):
            total_cost = sum(p * prices[i] for i, p in enumerate(power))
            passed.append(f"Общая стоимость решения: {total_cost:.2f} руб")

            # Проверка на разумность стоимости
            avg_power = np.mean(power)
            avg_price = np.mean(prices)
            expected_cost = avg_power * avg_price * horizon

            if expected_cost > 0:
                cost_ratio = total_cost / expected_cost
                if cost_ratio > 2:
                    warnings.append(f"Стоимость решения в {cost_ratio:.1f} раза выше ожидаемой")
                elif cost_ratio < 0.5:
                    warnings.append(f"Стоимость решения в {1 / cost_ratio:.1f} раза ниже ожидаемой")

        # 9. Дополнительные проверки для строгого уровня
        if self.level in [ValidationLevel.STRICT, ValidationLevel.FULL]:
            # Проверка на постоянное значение
            if len(power) > 1 and np.all(np.abs(np.diff(power)) < 0.01):
                warnings.append("Мощность практически постоянна на всем горизонте")

            # Проверка на чередование включений/выключений
            if len(on_status) > 3:
                changes = sum(1 for i in range(1, len(on_status))
                              if abs(on_status[i] - on_status[i - 1]) > 0.5)

                if changes > horizon / 4:  # Более 25% часов с изменением статуса
                    warnings.append(f"Частое переключение статуса: {changes} изменений за {horizon} часов")

        # Определение статуса
        status = self._determine_status(errors, warnings)

        result = ValidationResult(
            status=status,
            level=self.level,
            errors=errors,
            warnings=warnings,
            passed_checks=passed,
            metadata={
                'equipment_name': equipment_profile.name,
                'horizon': horizon,
                'solution_keys': list(solution.keys()),
                'validation_time': datetime.now().isoformat()
            }
        )

        self.results[f"optimization_{equipment_profile.name}"] = result
        return result

    def validate_dataset(self,
                         dataset: Dict[str, Any],
                         dataset_name: str = "unknown") -> ValidationResult:
        """
        Валидация полного набора данных

        Args:
            dataset: Набор данных
            dataset_name: Имя набора данных

        Returns:
            Результат валидации
        """
        log_info(f"Валидация набора данных: {dataset_name}")

        errors = []
        warnings = []
        passed = []

        # 1. Проверка наличия обязательных ключей
        required_sections = ['metadata', 'load_profile_kw', 'price_profile_rub_kwh']

        for section in required_sections:
            if section not in dataset:
                errors.append(f"Отсутствует обязательный раздел: {section}")
            else:
                passed.append(f"Раздел присутствует: {section}")

        # 2. Проверка метаданных
        if 'metadata' in dataset:
            metadata = dataset['metadata']
            required_meta = ['equipment_type', 'hours']

            for field in required_meta:
                if field not in metadata:
                    warnings.append(f"Отсутствует поле метаданных: {field}")
                else:
                    passed.append(f"Поле метаданных присутствует: {field}")

        # 3. Проверка профиля нагрузки
        if 'load_profile_kw' in dataset:
            load_profile = dataset['load_profile_kw']

            if not isinstance(load_profile, list):
                errors.append("Профиль нагрузки должен быть списком")
            else:
                # Рекурсивная валидация профиля нагрузки
                load_result = self.validate_load_profile(load_profile, profile_name=f"{dataset_name}_load")

                if not load_result.is_valid():
                    errors.extend([f"Нагрузка: {e}" for e in load_result.errors])
                    warnings.extend([f"Нагрузка: {w}" for w in load_result.warnings])

                passed.append(f"Профиль нагрузки проверен: {len(load_profile)} точек")

        # 4. Проверка профиля цен
        if 'price_profile_rub_kwh' in dataset:
            price_profile = dataset['price_profile_rub_kwh']

            if not isinstance(price_profile, list):
                errors.append("Профиль цен должен быть списком")
            else:
                # Рекурсивная валидация профиля цен
                price_result = self.validate_price_profile(price_profile, profile_name=f"{dataset_name}_price")

                if not price_result.is_valid():
                    errors.extend([f"Цены: {e}" for e in price_result.errors])
                    warnings.extend([f"Цены: {w}" for w in price_result.warnings])

                passed.append(f"Профиль цен проверен: {len(price_profile)} точек")

        # 5. Проверка согласованности данных
        if ('load_profile_kw' in dataset and 'price_profile_rub_kwh' in dataset and
                'hour_of_day' in dataset):
            load_len = len(dataset['load_profile_kw'])
            price_len = len(dataset['price_profile_rub_kwh'])
            hour_len = len(dataset['hour_of_day'])

            if not (load_len == price_len == hour_len):
                errors.append(
                    f"Несогласованность длин: нагрузка={load_len}, цены={price_len}, часы={hour_len}"
                )
            else:
                passed.append(f"Данные согласованы: все профили по {load_len} точек")

        # 6. Проверка погодных данных (если есть)
        if 'weather' in dataset:
            weather = dataset['weather']

            if isinstance(weather, dict):
                required_weather = ['temperature_c', 'humidity_percent']

                for field in required_weather:
                    if field in weather:
                        data = weather[field]
                        if isinstance(data, list):
                            if len(data) != len(dataset.get('load_profile_kw', [])):
                                warnings.append(f"Длина погодных данных '{field}' не соответствует нагрузке")
                            else:
                                passed.append(f"Погодные данные '{field}' проверены")

        # 7. Дополнительные проверки для строгого уровня
        if self.level in [ValidationLevel.STRICT, ValidationLevel.FULL]:
            # Проверка временных меток
            if 'timestamps' in dataset:
                timestamps = dataset['timestamps']
                if len(timestamps) > 1:
                    try:
                        # Проверка последовательности временных меток
                        from datetime import datetime
                        times = [datetime.fromisoformat(ts) for ts in timestamps]

                        # Проверка равномерности интервалов
                        intervals = [(times[i + 1] - times[i]).total_seconds()
                                     for i in range(len(times) - 1)]

                        if len(set(intervals)) > 1:
                            warnings.append("Неравномерные временные интервалы")
                        else:
                            passed.append("Временные интервалы равномерны")

                    except Exception as e:
                        warnings.append(f"Ошибка проверки временных меток: {e}")

        # Определение статуса
        status = self._determine_status(errors, warnings)

        result = ValidationResult(
            status=status,
            level=self.level,
            errors=errors,
            warnings=warnings,
            passed_checks=passed,
            metadata={
                'dataset_name': dataset_name,
                'validation_time': datetime.now().isoformat(),
                'sections_present': list(dataset.keys())
            }
        )

        self.results[f"dataset_{dataset_name}"] = result
        return result

    def _determine_status(self, errors: List[str], warnings: List[str]) -> ValidationStatus:
        """Определение статуса валидации на основе ошибок и предупреждений"""
        if errors:
            return ValidationStatus.INVALID
        elif warnings:
            return ValidationStatus.WARNING
        else:
            return ValidationStatus.VALID

    def save_validation_report(self,
                               result: ValidationResult,
                               filename: str = None) -> Path:
        """
        Сохранение отчета о валидации

        Args:
            result: Результат валидации
            filename: Имя файла

        Returns:
            Путь к сохраненному файлу
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"validation_{result.metadata.get('equipment_name', 'unknown')}_{timestamp}"

        # Сохранение в JSON
        report_data = result.to_dict()

        output_dir = EPOSConfig.REPORTS_DIR / "json" / "validation"
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / f"{filename}.json"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        log_info(f"Отчет о валидации сохранен: {json_path}")
        return json_path

    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Генерация сводного отчета по всем выполненным валидациям

        Returns:
            Сводный отчет
        """
        summary = {
            'total_validations': len(self.results),
            'valid_count': 0,
            'warning_count': 0,
            'invalid_count': 0,
            'error_count': 0,
            'by_category': {},
            'details': {}
        }

        for name, result in self.results.items():
            # Подсчет по статусам
            if result.status == ValidationStatus.VALID:
                summary['valid_count'] += 1
            elif result.status == ValidationStatus.WARNING:
                summary['warning_count'] += 1
            elif result.status == ValidationStatus.INVALID:
                summary['invalid_count'] += 1
            elif result.status == ValidationStatus.ERROR:
                summary['error_count'] += 1

            # Группировка по категориям
            category = name.split('_')[0]
            if category not in summary['by_category']:
                summary['by_category'][category] = {
                    'count': 0,
                    'valid': 0,
                    'warning': 0,
                    'invalid': 0,
                    'error': 0
                }

            summary['by_category'][category]['count'] += 1
            summary['by_category'][category][result.status.value] += 1

            # Детали
            summary['details'][name] = {
                'status': result.status.value,
                'errors_count': len(result.errors),
                'warnings_count': len(result.warnings),
                'passed_count': len(result.passed_checks),
                'metadata': result.metadata
            }

        summary['success_rate'] = (
                (summary['valid_count'] + summary['warning_count']) /
                summary['total_validations'] * 100
        ) if summary['total_validations'] > 0 else 0

        return summary


# Утилитарные функции для быстрого доступа
def quick_validate_equipment(equipment_type: str = "compressor") -> ValidationResult:
    """Быстрая валидация профиля оборудования"""
    from config.equipment_profiles import EquipmentManager

    profile = EquipmentManager.get_profile(f"{equipment_type}_air")
    validator = EPOSValidator(level=ValidationLevel.BASIC)
    return validator.validate_equipment_profile(profile)


def validate_data_file(filepath: Path) -> ValidationResult:
    """Валидация данных из файла"""
    import json

    if not filepath.exists():
        return ValidationResult(
            status=ValidationStatus.ERROR,
            level=ValidationLevel.BASIC,
            errors=[f"Файл не найден: {filepath}"],
            metadata={'filepath': str(filepath)}
        )

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        validator = EPOSValidator(level=ValidationLevel.FULL)
        return validator.validate_dataset(data, dataset_name=filepath.stem)

    except Exception as e:
        return ValidationResult(
            status=ValidationStatus.ERROR,
            level=ValidationLevel.BASIC,
            errors=[f"Ошибка загрузки файла: {str(e)}"],
            metadata={'filepath': str(filepath)}
        )


def batch_validate_directory(directory: Path, pattern: str = "*.json") -> Dict[str, ValidationResult]:
    """Пакетная валидация всех файлов в директории"""
    results = {}

    if not directory.exists():
        log_error(f"Директория не существует: {directory}")
        return results

    files = list(directory.glob(pattern))
    log_info(f"Найдено {len(files)} файлов для валидации")

    for filepath in files:
        log_info(f"Валидация файла: {filepath.name}")
        result = validate_data_file(filepath)
        results[filepath.name] = result

        if result.is_valid():
            log_info(f"  ✓ Файл валиден")
        else:
            log_warning(f"  ✗ Файл невалиден: {len(result.errors)} ошибок")

    return results