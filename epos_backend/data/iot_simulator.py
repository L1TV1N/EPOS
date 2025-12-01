"""
Симулятор IoT-датчиков для системы ЭПОС
Разработано командой TechLitCode
"""

import time
import random
import threading
import queue
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from pathlib import Path
from config.settings import EPOSConfig
from config.equipment_profiles import EquipmentProfile
from utils.logger import log_info, log_error, log_warning, log_debug
from utils.helpers import generate_timestamps, add_noise, calculate_statistics


class SensorType(Enum):
    """Типы IoT-датчиков"""
    POWER = "power"  # Мощность, кВт
    CURRENT = "current"  # Ток, А
    VOLTAGE = "voltage"  # Напряжение, В
    TEMPERATURE = "temperature"  # Температура, °C
    PRESSURE = "pressure"  # Давление, бар
    FLOW = "flow"  # Расход, м³/ч
    VIBRATION = "vibration"  # Вибрация, мм/с
    STATUS = "status"  # Статус оборудования


@dataclass
class SensorReading:
    """Показание датчика"""
    sensor_id: str
    sensor_type: SensorType
    value: float
    unit: str
    timestamp: datetime
    equipment_id: str
    quality: float = 1.0  # Качество измерения (0-1)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp.isoformat(),
            'equipment_id': self.equipment_id,
            'quality': self.quality,
            'metadata': self.metadata
        }


@dataclass
class SensorConfig:
    """Конфигурация датчика"""
    sensor_id: str
    sensor_type: SensorType
    equipment_id: str
    min_value: float
    max_value: float
    unit: str
    sampling_rate: float  # Частота опроса, Гц
    noise_level: float = 0.02  # Уровень шума (2%)
    drift_rate: float = 0.001  # Скорость дрейфа в час
    failure_probability: float = 0.001  # Вероятность отказа

    def __post_init__(self):
        self.current_drift = 0.0
        self.failed = False
        self.last_maintenance = datetime.now()


class IoTDevice:
    """IoT-устройство (оборудование с датчиками)"""

    def __init__(self,
                 device_id: str,
                 equipment_profile: EquipmentProfile,
                 location: str = "Цех 1",
                 manufacturer: str = "TechLitCode"):
        """
        Инициализация IoT-устройства

        Args:
            device_id: Уникальный ID устройства
            equipment_profile: Профиль оборудования
            location: Местоположение
            manufacturer: Производитель
        """
        self.device_id = device_id
        self.profile = equipment_profile
        self.location = location
        self.manufacturer = manufacturer

        self.sensors: Dict[str, SensorConfig] = {}
        self.readings_history: List[SensorReading] = []
        self.status = "OFFLINE"
        self.last_update = None
        self.operating_hours = 0

        # Создание датчиков по умолчанию
        self._create_default_sensors()

        log_info(f"Создано IoT-устройство: {device_id} ({equipment_profile.name})")

    def _create_default_sensors(self):
        """Создание датчиков по умолчанию для оборудования"""
        base_sensors = [
            SensorConfig(
                sensor_id=f"{self.device_id}_PWR_01",
                sensor_type=SensorType.POWER,
                equipment_id=self.device_id,
                min_value=self.profile.power_min,
                max_value=self.profile.power_max,
                unit="кВт",
                sampling_rate=1.0  # 1 раз в секунду
            ),
            SensorConfig(
                sensor_id=f"{self.device_id}_CUR_01",
                sensor_type=SensorType.CURRENT,
                equipment_id=self.device_id,
                min_value=10,
                max_value=500,
                unit="А",
                sampling_rate=2.0
            ),
            SensorConfig(
                sensor_id=f"{self.device_id}_VOL_01",
                sensor_type=SensorType.VOLTAGE,
                equipment_id=self.device_id,
                min_value=380,
                max_value=400,
                unit="В",
                sampling_rate=2.0
            ),
            SensorConfig(
                sensor_id=f"{self.device_id}_TEMP_01",
                sensor_type=SensorType.TEMPERATURE,
                equipment_id=self.device_id,
                min_value=20,
                max_value=120,
                unit="°C",
                sampling_rate=0.5
            ),
            SensorConfig(
                sensor_id=f"{self.device_id}_STAT_01",
                sensor_type=SensorType.STATUS,
                equipment_id=self.device_id,
                min_value=0,
                max_value=1,
                unit="bool",
                sampling_rate=0.2
            )
        ]

        # Добавление специализированных датчиков в зависимости от типа оборудования
        if self.profile.equipment_type == "compressor":
            base_sensors.append(SensorConfig(
                sensor_id=f"{self.device_id}_PRESS_01",
                sensor_type=SensorType.PRESSURE,
                equipment_id=self.device_id,
                min_value=0,
                max_value=10,
                unit="бар",
                sampling_rate=5.0
            ))
        elif self.profile.equipment_type == "pump":
            base_sensors.append(SensorConfig(
                sensor_id=f"{self.device_id}_FLOW_01",
                sensor_type=SensorType.FLOW,
                equipment_id=self.device_id,
                min_value=0,
                max_value=100,
                unit="м³/ч",
                sampling_rate=5.0
            ))
        elif self.profile.equipment_type == "oven":
            base_sensors.append(SensorConfig(
                sensor_id=f"{self.device_id}_TEMP_02",  # Дополнительный датчик температуры
                sensor_type=SensorType.TEMPERATURE,
                equipment_id=self.device_id,
                min_value=100,
                max_value=1200,
                unit="°C",
                sampling_rate=10.0
            ))

        # Регистрация датчиков
        for sensor in base_sensors:
            self.add_sensor(sensor)

    def add_sensor(self, sensor_config: SensorConfig):
        """Добавление датчика к устройству"""
        self.sensors[sensor_config.sensor_id] = sensor_config
        log_debug(f"Добавлен датчик: {sensor_config.sensor_id} ({sensor_config.sensor_type.value})")

    def read_sensor(self, sensor_id: str) -> Optional[SensorReading]:
        """
        Чтение показаний датчика

        Args:
            sensor_id: ID датчика

        Returns:
            Показание датчика или None если датчик не найден
        """
        if sensor_id not in self.sensors:
            log_warning(f"Датчик не найден: {sensor_id}")
            return None

        sensor = self.sensors[sensor_id]

        # Проверка на отказ
        if sensor.failed:
            log_warning(f"Датчик {sensor_id} неисправен")
            return SensorReading(
                sensor_id=sensor_id,
                sensor_type=sensor.sensor_type,
                value=0.0,
                unit=sensor.unit,
                timestamp=datetime.now(),
                equipment_id=self.device_id,
                quality=0.0,
                metadata={'error': 'sensor_failed'}
            )

        # Моделирование дрейфа
        hours_since_maintenance = (datetime.now() - sensor.last_maintenance).total_seconds() / 3600
        sensor.current_drift = hours_since_maintenance * sensor.drift_rate

        # Генерация значения в зависимости от типа датчика
        value = self._generate_sensor_value(sensor)

        # Добавление шума
        if sensor.noise_level > 0:
            noise = np.random.randn() * sensor.noise_level * (sensor.max_value - sensor.min_value)
            value += noise

        # Добавление дрейфа
        value += sensor.current_drift * (sensor.max_value - sensor.min_value)

        # Ограничение диапазона
        value = max(sensor.min_value, min(sensor.max_value, value))

        # Расчет качества
        quality = self._calculate_sensor_quality(sensor, value)

        # Случайный отказ
        if random.random() < sensor.failure_probability:
            sensor.failed = True
            log_error(f"Датчик {sensor_id} вышел из строя!")

        reading = SensorReading(
            sensor_id=sensor_id,
            sensor_type=sensor.sensor_type,
            value=round(value, 3),
            unit=sensor.unit,
            timestamp=datetime.now(),
            equipment_id=self.device_id,
            quality=quality,
            metadata={
                'drift': sensor.current_drift,
                'hours_since_maintenance': hours_since_maintenance,
                'operating_hours': self.operating_hours
            }
        )

        # Сохранение в историю
        self.readings_history.append(reading)

        # Обновление статуса
        self.last_update = datetime.now()
        self.status = "ONLINE"

        return reading

    def _generate_sensor_value(self, sensor: SensorConfig) -> float:
        """Генерация значения датчика на основе типа оборудования и состояния"""

        # Базовое значение в зависимости от типа датчика
        if sensor.sensor_type == SensorType.POWER:
            # Мощность зависит от режима работы
            if self.status == "RUNNING":
                base_value = self.profile.power_min + (self.profile.power_max - self.profile.power_min) * 0.7
            else:
                base_value = 0.0

        elif sensor.sensor_type == SensorType.CURRENT:
            # Ток пропорционален мощности
            if self.status == "RUNNING":
                base_value = 100 + (self.profile.power_max / 10) * 0.7
            else:
                base_value = 0.0

        elif sensor.sensor_type == SensorType.VOLTAGE:
            # Напряжение с небольшими колебаниями
            base_value = 390 + random.uniform(-10, 10)

        elif sensor.sensor_type == SensorType.TEMPERATURE:
            # Температура зависит от работы оборудования
            if self.status == "RUNNING":
                base_value = 80 + random.uniform(-5, 20)
            else:
                base_value = 25 + random.uniform(-5, 5)

        elif sensor.sensor_type == SensorType.PRESSURE:
            # Давление для компрессора
            if self.status == "RUNNING":
                base_value = 6 + random.uniform(-1, 2)
            else:
                base_value = 0.0

        elif sensor.sensor_type == SensorType.FLOW:
            # Расход для насоса
            if self.status == "RUNNING":
                base_value = 50 + random.uniform(-10, 20)
            else:
                base_value = 0.0

        elif sensor.sensor_type == SensorType.STATUS:
            # Статус (0=выкл, 1=вкл)
            base_value = 1.0 if self.status == "RUNNING" else 0.0

        else:
            # Для неизвестных датчиков - случайное значение
            base_value = sensor.min_value + (sensor.max_value - sensor.min_value) * 0.5

        return base_value

    def _calculate_sensor_quality(self, sensor: SensorConfig, value: float) -> float:
        """Расчет качества показаний датчика"""
        quality = 1.0

        # Снижение качества из-за дрейфа
        if abs(sensor.current_drift) > 0.01:  # Дрейф более 1%
            quality -= 0.2

        # Снижение качества из-за времени работы
        hours_since_maintenance = (datetime.now() - sensor.last_maintenance).total_seconds() / 3600
        if hours_since_maintenance > 720:  # Более месяца
            quality -= 0.3

        # Проверка на выход за пределы
        if value < sensor.min_value * 1.1 or value > sensor.max_value * 0.9:
            quality -= 0.1

        return max(0.0, min(1.0, quality))

    def set_status(self, status: str):
        """Установка статуса оборудования"""
        valid_statuses = ["OFFLINE", "STANDBY", "RUNNING", "FAULT", "MAINTENANCE"]

        if status not in valid_statuses:
            log_warning(f"Неизвестный статус: {status}. Используется OFFLINE")
            status = "OFFLINE"

        old_status = self.status
        self.status = status

        if status == "RUNNING" and old_status != "RUNNING":
            log_info(f"Оборудование {self.device_id} запущено")
        elif status != "RUNNING" and old_status == "RUNNING":
            log_info(f"Оборудование {self.device_id} остановлено")

        # Обновление времени работы
        if status == "RUNNING":
            self.operating_hours += 0.1  # Упрощенное накопление

    def read_all_sensors(self) -> List[SensorReading]:
        """Чтение всех датчиков устройства"""
        readings = []

        for sensor_id in self.sensors:
            reading = self.read_sensor(sensor_id)
            if reading:
                readings.append(reading)

        return readings

    def get_sensor_history(self,
                           sensor_id: str,
                           time_window: Optional[timedelta] = None) -> List[SensorReading]:
        """
        Получение истории показаний датчика

        Args:
            sensor_id: ID датчика
            time_window: Временное окно (например, последние 24 часа)

        Returns:
            Список показаний
        """
        if time_window:
            cutoff_time = datetime.now() - time_window
            history = [
                r for r in self.readings_history
                if r.sensor_id == sensor_id and r.timestamp >= cutoff_time
            ]
        else:
            history = [r for r in self.readings_history if r.sensor_id == sensor_id]

        return history

    def perform_maintenance(self):
        """Выполнение технического обслуживания датчиков"""
        for sensor in self.sensors.values():
            sensor.last_maintenance = datetime.now()
            sensor.current_drift = 0.0
            sensor.failed = False

        log_info(f"Техническое обслуживание выполнено для {self.device_id}")

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация устройства в словарь"""
        return {
            'device_id': self.device_id,
            'equipment_profile': self.profile.to_dict(),
            'location': self.location,
            'manufacturer': self.manufacturer,
            'status': self.status,
            'operating_hours': self.operating_hours,
            'sensor_count': len(self.sensors),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }


class IoTSimulator:
    """Симулятор сети IoT-устройств"""

    def __init__(self, num_devices: int = 3):
        """
        Инициализация симулятора

        Args:
            num_devices: Количество симулируемых устройств
        """
        self.num_devices = num_devices
        self.devices: Dict[str, IoTDevice] = {}
        self.data_queue = queue.Queue()
        self.running = False
        self.simulation_thread = None

        # Типы оборудования для симуляции
        self.equipment_types = ["compressor", "pump", "oven", "ventilation", "conveyor"]

        log_info(f"Инициализация IoT-симулятора для {num_devices} устройств")
        log_info(f"Разработано командой {EPOSConfig.DEVELOPER}")

    def setup_devices(self):
        """Настройка симулируемых устройств"""
        from config.equipment_profiles import EquipmentManager

        for i in range(self.num_devices):
            # Выбор типа оборудования
            eq_type = self.equipment_types[i % len(self.equipment_types)]

            # Получение профиля оборудования
            try:
                profile = EquipmentManager.get_profile(f"{eq_type}_{self._get_subtype(eq_type)}")
            except ValueError:
                # Создание простого профиля
                from config.equipment_profiles import EquipmentProfile
                profile = EquipmentProfile(
                    name=f"{eq_type.capitalize()} #{i + 1}",
                    equipment_type=eq_type,
                    power_nominal=100,
                    power_min=20,
                    power_max=200,
                    min_on_time=2,
                    min_off_time=1,
                    startup_time=0.5,
                    shutdown_time=0.25,
                    efficiency=0.85
                )

            # Создание устройства
            device_id = f"DEV_{eq_type.upper()}_{i + 1:03d}"
            device = IoTDevice(
                device_id=device_id,
                equipment_profile=profile,
                location=f"Цех {i % 3 + 1}",
                manufacturer="TechLitCode IoT"
            )

            # Случайный начальный статус
            statuses = ["OFFLINE", "STANDBY", "RUNNING"]
            device.set_status(random.choice(statuses))

            self.devices[device_id] = device
            log_info(f"Добавлено устройство: {device_id} ({profile.name})")

    def _get_subtype(self, eq_type: str) -> str:
        """Получение подтипа оборудования"""
        subtypes = {
            "compressor": "air",
            "pump": "water_cooling",
            "oven": "induction",
            "ventilation": "main",
            "conveyor": "belt"
        }
        return subtypes.get(eq_type, "air")

    def start_simulation(self, interval: float = 1.0):
        """
        Запуск симуляции

        Args:
            interval: Интервал между опросами датчиков в секундах
        """
        if self.running:
            log_warning("Симуляция уже запущена")
            return

        self.running = True
        self.simulation_thread = threading.Thread(
            target=self._simulation_loop,
            args=(interval,),
            daemon=True
        )
        self.simulation_thread.start()

        log_info(f"Симуляция запущена (интервал: {interval} сек)")

    def _simulation_loop(self, interval: float):
        """Основной цикл симуляции"""
        log_info("Цикл симуляции запущен")

        try:
            while self.running:
                # Опрос всех устройств
                for device_id, device in self.devices.items():
                    # Случайное изменение статуса оборудования (10% вероятность)
                    if random.random() < 0.1:
                        new_status = random.choice(["OFFLINE", "STANDBY", "RUNNING", "FAULT"])
                        device.set_status(new_status)

                    # Чтение всех датчиков устройства
                    readings = device.read_all_sensors()

                    # Помещение данных в очередь
                    for reading in readings:
                        self.data_queue.put(reading)

                # Пауза между циклами
                time.sleep(interval)

        except Exception as e:
            log_error(f"Ошибка в цикле симуляции: {e}")
            self.running = False

    def stop_simulation(self):
        """Остановка симуляции"""
        self.running = False

        if self.simulation_thread:
            self.simulation_thread.join(timeout=2.0)

        log_info("Симуляция остановлена")

    def get_latest_data(self, max_items: int = 100) -> List[SensorReading]:
        """
        Получение последних данных из очереди

        Args:
            max_items: Максимальное количество элементов

        Returns:
            Список последних показаний
        """
        data = []

        try:
            while not self.data_queue.empty() and len(data) < max_items:
                data.append(self.data_queue.get_nowait())
        except queue.Empty:
            pass

        return data

    def stream_data(self,
                    callback: Callable[[SensorReading], None],
                    interval: float = 1.0):
        """
        Потоковая передача данных через callback

        Args:
            callback: Функция для обработки данных
            interval: Интервал проверки новых данных
        """

        def stream_loop():
            while self.running:
                data = self.get_latest_data()
                for reading in data:
                    callback(reading)
                time.sleep(interval)

        stream_thread = threading.Thread(target=stream_loop, daemon=True)
        stream_thread.start()

        log_info(f"Потоковая передача данных запущена (интервал: {interval} сек)")

    def save_device_data(self,
                         device_id: str,
                         time_window: Optional[timedelta] = None) -> Path:
        """
        Сохранение данных устройства в файл

        Args:
            device_id: ID устройства
            time_window: Временное окно

        Returns:
            Путь к сохраненному файлу
        """
        if device_id not in self.devices:
            raise ValueError(f"Устройство не найдено: {device_id}")

        device = self.devices[device_id]

        # Сбор данных
        all_readings = []
        for sensor_id in device.sensors:
            sensor_history = device.get_sensor_history(sensor_id, time_window)
            all_readings.extend([r.to_dict() for r in sensor_history])

        # Сохранение в JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iot_data_{device_id}_{timestamp}.json"

        output_dir = EPOSConfig.OUTPUTS_DIR / "json" / "iot"
        output_dir.mkdir(parents=True, exist_ok=True)

        filepath = output_dir / filename

        data = {
            'metadata': {
                'device_id': device_id,
                'exported_at': datetime.now().isoformat(),
                'readings_count': len(all_readings),
                'time_window': str(time_window) if time_window else 'all',
                'system': EPOSConfig.SYSTEM_NAME
            },
            'device_info': device.to_dict(),
            'readings': all_readings
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        log_info(f"Данные устройства сохранены: {filepath}")
        return filepath

    def generate_anomalies(self,
                           anomaly_type: str = "spike",
                           intensity: float = 0.1):
        """
        Генерация аномалий в данных

        Args:
            anomaly_type: Тип аномалии (spike, drift, noise, failure)
            intensity: Интенсивность аномалии (0-1)
        """
        log_info(f"Генерация аномалий типа '{anomaly_type}' с интенсивностью {intensity}")

        for device in self.devices.values():
            for sensor in device.sensors.values():
                if random.random() < intensity:
                    if anomaly_type == "spike":
                        # Временный всплеск значений
                        sensor.max_value *= 1.5
                        log_debug(f"Аномалия: всплеск на датчике {sensor.sensor_id}")

                    elif anomaly_type == "drift":
                        # Ускоренный дрейф
                        sensor.drift_rate *= 5.0
                        log_debug(f"Аномалия: дрейф на датчике {sensor.sensor_id}")

                    elif anomaly_type == "noise":
                        # Увеличение шума
                        sensor.noise_level = min(0.5, sensor.noise_level * 3)
                        log_debug(f"Аномалия: шум на датчике {sensor.sensor_id}")

                    elif anomaly_type == "failure":
                        # Отказ датчика
                        sensor.failed = True
                        log_debug(f"Аномалия: отказ датчика {sensor.sensor_id}")


# Утилитарные функции для быстрого доступа
def create_demo_simulator(num_devices: int = 3) -> IoTSimulator:
    """Создание демонстрационного симулятора"""
    simulator = IoTSimulator(num_devices=num_devices)
    simulator.setup_devices()
    return simulator


def simulate_real_time(duration_seconds: int = 60,
                       interval: float = 1.0,
                       callback: Optional[Callable] = None):
    """
    Запуск реальной симуляции

    Args:
        duration_seconds: Длительность симуляции в секундах
        interval: Интервал между опросами
        callback: Функция для обработки данных
    """
    simulator = create_demo_simulator(2)
    simulator.start_simulation(interval)

    log_info(f"Запущена реальная симуляция на {duration_seconds} секунд")

    try:
        if callback:
            simulator.stream_data(callback, interval)

        time.sleep(duration_seconds)

    finally:
        simulator.stop_simulation()
        log_info("Симуляция завершена")


def print_sensor_data(reading: SensorReading):
    """Функция для печати данных датчиков"""
    print(f"[{reading.timestamp.strftime('%H:%M:%S')}] "
          f"{reading.equipment_id}.{reading.sensor_id}: "
          f"{reading.value:.2f} {reading.unit} "
          f"(качество: {reading.quality:.2f})")