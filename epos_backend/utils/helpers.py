"""
Вспомогательные функции для системы ЭПОС
Разработано командой TechLitCode
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union
import json
import pickle
import hashlib
from pathlib import Path
import warnings
from config.settings import EPOSConfig
from utils.logger import log_info, log_error, log_warning


def generate_timestamps(hours: int = 24,
                        start_date: Optional[datetime] = None,
                        interval: float = 1.0) -> List[datetime]:
    """
    Генерация временных меток

    Args:
        hours: Количество часов
        start_date: Начальная дата-время
        interval: Интервал в часах

    Returns:
        Список datetime объектов
    """
    if start_date is None:
        start_date = datetime.now().replace(minute=0, second=0, microsecond=0)

    timestamps = []
    for i in range(hours):
        timestamp = start_date + timedelta(hours=i * interval)
        timestamps.append(timestamp)

    return timestamps


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Расчет статистики для данных

    Args:
        data: Список числовых значений

    Returns:
        Словарь со статистикой
    """
    if not data:
        return {}

    arr = np.array(data)

    stats = {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
        "q25": float(np.percentile(arr, 25)),
        "q75": float(np.percentile(arr, 75)),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25))
    }

    return stats


def normalize_data(data: List[float],
                   method: str = "minmax") -> List[float]:
    """
    Нормализация данных

    Args:
        data: Входные данные
        method: Метод нормализации (minmax, zscore, robust)

    Returns:
        Нормализованные данные
    """
    if not data:
        return []

    arr = np.array(data)

    if method == "minmax":
        if np.max(arr) == np.min(arr):
            return [0.5] * len(data)
        normalized = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    elif method == "zscore":
        if np.std(arr) == 0:
            return [0] * len(data)
        normalized = (arr - np.mean(arr)) / np.std(arr)

    elif method == "robust":
        q75, q25 = np.percentile(arr, [75, 25])
        if q75 == q25:
            return [0] * len(data)
        normalized = (arr - np.median(arr)) / (q75 - q25)

    else:
        log_warning(f"Неизвестный метод нормализации: {method}. Используется minmax.")
        if np.max(arr) == np.min(arr):
            return [0.5] * len(data)
        normalized = (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

    return normalized.tolist()


def add_noise(data: List[float],
              noise_level: float = 0.05,
              noise_type: str = "gaussian") -> List[float]:
    """
    Добавление шума к данным

    Args:
        data: Исходные данные
        noise_level: Уровень шума (стандартное отклонение для гауссовского)
        noise_type: Тип шума (gaussian, uniform, percentage)

    Returns:
        Данные с шумом
    """
    if not data:
        return []

    arr = np.array(data)

    if noise_type == "gaussian":
        noise = np.random.normal(0, noise_level * np.std(arr), len(arr))
        noisy_data = arr + noise

    elif noise_type == "uniform":
        noise = np.random.uniform(-noise_level, noise_level, len(arr)) * np.std(arr)
        noisy_data = arr + noise

    elif noise_type == "percentage":
        noise = np.random.uniform(-noise_level, noise_level, len(arr)) * arr
        noisy_data = arr + noise

    else:
        log_warning(f"Неизвестный тип шума: {noise_type}. Используется gaussian.")
        noise = np.random.normal(0, noise_level * np.std(arr), len(arr))
        noisy_data = arr + noise

    # Защита от отрицательных значений
    noisy_data = np.maximum(noisy_data, 0)

    return noisy_data.tolist()


def detect_peaks(data: List[float],
                 threshold: float = 0.8) -> List[int]:
    """
    Обнаружение пиков в данных

    Args:
        data: Входные данные
        threshold: Порог для обнаружения пиков (0-1)

    Returns:
        Индексы пиков
    """
    if len(data) < 3:
        return []

    normalized = normalize_data(data, "minmax")
    mean_val = np.mean(normalized)
    std_val = np.std(normalized)

    peaks = []
    for i in range(1, len(data) - 1):
        if (normalized[i] > normalized[i - 1] and
                normalized[i] > normalized[i + 1] and
                normalized[i] > mean_val + threshold * std_val):
            peaks.append(i)

    return peaks


def save_to_csv(data: Dict[str, Any],
                filename: str,
                directory: Optional[Path] = None) -> Path:
    """
    Сохранение данных в CSV

    Args:
        data: Данные для сохранения
        filename: Имя файла
        directory: Директория для сохранения

    Returns:
        Путь к сохраненному файлу
    """
    if directory is None:
        directory = EPOSConfig.REPORTS_DIR / "csv"

    directory.mkdir(parents=True, exist_ok=True)

    if not filename.endswith('.csv'):
        filename += '.csv'

    filepath = directory / filename

    try:
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        log_info(f"Данные сохранены в CSV: {filepath}")
        return filepath
    except Exception as e:
        log_error(f"Ошибка сохранения в CSV: {e}")
        raise


def save_to_json(data: Dict[str, Any],
                 filename: str,
                 directory: Optional[Path] = None) -> Path:
    """
    Сохранение данных в JSON

    Args:
        data: Данные для сохранения
        filename: Имя файла
        directory: Директория для сохранения

    Returns:
        Путь к сохраненному файлу
    """
    if directory is None:
        directory = EPOSConfig.REPORTS_DIR / "json"

    directory.mkdir(parents=True, exist_ok=True)

    if not filename.endswith('.json'):
        filename += '.json'

    filepath = directory / filename

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log_info(f"Данные сохранены в JSON: {filepath}")
        return filepath
    except Exception as e:
        log_error(f"Ошибка сохранения в JSON: {e}")
        raise


def load_from_json(filename: str,
                   directory: Optional[Path] = None) -> Dict[str, Any]:
    """
    Загрузка данных из JSON

    Args:
        filename: Имя файла
        directory: Директория

    Returns:
        Загруженные данные
    """
    if directory is None:
        directory = EPOSConfig.REPORTS_DIR / "json"

    filepath = directory / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        log_info(f"Данные загружены из JSON: {filepath}")
        return data
    except Exception as e:
        log_error(f"Ошибка загрузки из JSON: {e}")
        raise


def format_currency(value: float) -> str:
    """
    Форматирование валюты

    Args:
        value: Числовое значение

    Returns:
        Отформатированная строка
    """
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f} млн руб."
    elif abs(value) >= 1_000:
        return f"{value / 1_000:.2f} тыс. руб."
    else:
        return f"{value:.2f} руб."


def format_power(value: float) -> str:
    """
    Форматирование мощности

    Args:
        value: Мощность в кВт

    Returns:
        Отформатированная строка
    """
    if value >= 1000:
        return f"{value / 1000:.2f} МВт"
    else:
        return f"{value:.2f} кВт"


def format_energy(value: float) -> str:
    """
    Форматирование энергии

    Args:
        value: Энергия в кВт*ч

    Returns:
        Отформатированная строка
    """
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f} ГВт*ч"
    elif value >= 1_000:
        return f"{value / 1_000:.2f} МВт*ч"
    else:
        return f"{value:.2f} кВт*ч"


def calculate_hash(data: Any) -> str:
    """
    Расчет хэша данных

    Args:
        data: Данные для хэширования

    Returns:
        Хэш-строка
    """
    data_str = str(data).encode('utf-8')
    return hashlib.md5(data_str).hexdigest()


def validate_time_series(data: List[float],
                         expected_length: int = 24) -> List[str]:
    """
    Валидация временного ряда

    Args:
        data: Временной ряд
        expected_length: Ожидаемая длина

    Returns:
        Список ошибок
    """
    errors = []

    if len(data) != expected_length:
        errors.append(f"Некорректная длина: {len(data)} (ожидается {expected_length})")

    if any(np.isnan(x) for x in data):
        errors.append("Обнаружены NaN значения")

    if any(np.isinf(x) for x in data):
        errors.append("Обнаружены бесконечные значения")

    if any(x < 0 for x in data):
        errors.append("Обнаружены отрицательные значения")

    return errors


def smooth_data(data: List[float],
                window_size: int = 3) -> List[float]:
    """
    Сглаживание данных скользящим средним

    Args:
        data: Исходные данные
        window_size: Размер окна

    Returns:
        Сглаженные данные
    """
    if len(data) < window_size:
        return data

    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size // 2)
        end = min(len(data), i + window_size // 2 + 1)
        window = data[start:end]
        smoothed.append(np.mean(window))

    return smoothed


def print_table(headers: List[str],
                rows: List[List[Any]],
                title: str = None):
    """
    Красивая печать таблицы в консоль

    Args:
        headers: Заголовки столбцов
        rows: Строки данных
        title: Заголовок таблицы
    """
    if title:
        print(f"\n{title}")
        print("=" * 60)

    # Определяем ширину столбцов
    col_widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))

    # Печать заголовков
    header_str = " | ".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))

    # Печать строк
    for row in rows:
        row_str = " | ".join(str(cell).ljust(w) for cell, w in zip(row, col_widths))
        print(row_str)


def get_timestamp_string() -> str:
    """
    Получение строки с временной меткой

    Returns:
        Строка с временной меткой
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def check_disk_space(min_gb: float = 1.0) -> bool:
    """
    Проверка свободного места на диске

    Args:
        min_gb: Минимальное необходимое место в GB

    Returns:
        True если места достаточно
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free / (2 ** 30)  # Convert to GB

        if free_gb < min_gb:
            log_warning(f"Мало свободного места: {free_gb:.1f} GB (требуется {min_gb} GB)")
            return False
        else:
            log_info(f"Свободное место на диске: {free_gb:.1f} GB")
            return True
    except Exception as e:
        log_error(f"Ошибка проверки дискового пространства: {e}")
        return True  # Продолжаем работу даже при ошибке


def backup_data(data: Any,
                description: str = "") -> Path:
    """
    Создание резервной копии данных

    Args:
        data: Данные для резервного копирования
        description: Описание данных

    Returns:
        Путь к файлу резервной копии
    """
    timestamp = get_timestamp_string()
    filename = f"backup_{timestamp}_{hashlib.md5(description.encode()).hexdigest()[:8]}.pkl"
    filepath = EPOSConfig.OUTPUTS_DIR / filename

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        log_info(f"Резервная копия создана: {filepath} ({description})")
        return filepath
    except Exception as e:
        log_error(f"Ошибка создания резервной копии: {e}")
        raise