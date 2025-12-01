"""
Конфигурационные настройки проекта ЭПОС
Разработано командой TechLitCode
"""

import os
from pathlib import Path
from datetime import datetime
import json


class EPOSConfig:
    """Класс конфигурации системы ЭПОС"""

    # Версия системы
    VERSION = "1.0.0"
    SYSTEM_NAME = "ЭПОС (Энерго-Производственный Оптимизационный Синтез)"
    DEVELOPER = "TechLitCode"

    # Пути
    BASE_DIR = Path(__file__).resolve().parent.parent
    OUTPUTS_DIR = BASE_DIR / "outputs"
    LOGS_DIR = OUTPUTS_DIR / "logs"
    REPORTS_DIR = OUTPUTS_DIR / "reports"
    PLOTS_DIR = OUTPUTS_DIR / "plots"

    # Создание директорий при инициализации
    @classmethod
    def init_directories(cls):
        """Инициализация структуры директорий"""
        directories = [
            cls.OUTPUTS_DIR,
            cls.LOGS_DIR,
            cls.REPORTS_DIR / "csv",
            cls.REPORTS_DIR / "json",
            cls.REPORTS_DIR / "pdf",
            cls.PLOTS_DIR
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            cls._log(f"Директория создана: {directory}")

    # Настройки логирования
    LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    LOG_TO_FILE = True
    LOG_TO_CONSOLE = True
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

    # Настройки оптимизации
    OPTIMIZATION_HORIZON = 24  # Часов для оптимизации
    TIME_INTERVAL = 1  # Часовой интервал
    SOLVER = "PULP"  # PULP или CVXPY
    OPTIMIZATION_TIMEOUT = 60  # Секунд на решение

    # Настройки данных
    USE_REAL_PRICES = True
    PRICE_DATA_SOURCE = "ATS"  # ATS или SPB
    DEFAULT_EQUIPMENT = "compressor"  # По умолчанию компрессор
    DATA_SAMPLING_RATE = 300  # Секунд между замерами

    # Настройки оборудования
    EQUIPMENT_PROFILES = {
        "compressor": {
            "name": "Промышленный компрессор",
            "power_nominal": 200,  # кВт
            "power_min": 50,
            "power_max": 200,
            "min_on_time": 2,  # Минимальное время работы, часов
            "min_off_time": 1,  # Минимальное время простоя
            "startup_time": 0.5,
            "shutdown_time": 0.25,
            "efficiency": 0.85
        },
        "pump": {
            "name": "Водяной насос",
            "power_nominal": 150,
            "power_min": 30,
            "power_max": 150,
            "min_on_time": 1,
            "min_off_time": 0.5,
            "startup_time": 0.25,
            "shutdown_time": 0.25,
            "efficiency": 0.88
        },
        "oven": {
            "name": "Промышленная печь",
            "power_nominal": 500,
            "power_min": 100,
            "power_max": 500,
            "min_on_time": 4,
            "min_off_time": 2,
            "startup_time": 1,
            "shutdown_time": 0.5,
            "efficiency": 0.75
        }
    }

    # Настройки симуляции
    SIMULATION_DAYS = 7
    PRICE_VARIABILITY = 0.2  # Волатильность цен (20%)
    LOAD_NOISE_LEVEL = 0.05  # Уровень шума в данных нагрузки (5%)

    # Настройки отчетов
    REPORT_FORMATS = ["csv", "json", "pdf"]
    GENERATE_PLOTS = True
    PLOT_FORMAT = "png"
    PLOT_DPI = 150

    # Стили брендинга TechLitCode
    BRANDING = {
        "name": "TechLitCode",
        "colors": {
            "primary": "#1B2B4E",  # Темно-синий
            "accent": "#F5C32C",  # Золотой
            "secondary": "#f5f5f7",  # Светло-серый
            "text": "#1B2B4E"
        },
        "logo": "╔══════════════════════════════╗\n"
                "║     ЭПОС v1.0.0              ║\n"
                "║  TechLitCode • Энерго-система║\n"
                "╚══════════════════════════════╝"
    }

    @classmethod
    def get_branding_header(cls):
        """Получить заголовок с брендингом"""
        return f"""
{'=' * 60}
{cls.BRANDING['logo']}
{cls.SYSTEM_NAME}
Разработано командой {cls.DEVELOPER}
Версия: {cls.VERSION}
{'=' * 60}
        """

    @classmethod
    def _log(cls, message):
        """Внутреннее логирование инициализации"""
        print(f"[EPOS Config] {message}")

    @classmethod
    def save_config(cls, filename="epos_config.json"):
        """Сохранить конфигурацию в файл"""
        config_dict = {
            k: v for k, v in cls.__dict__.items()
            if not k.startswith('_') and not callable(v)
        }

        # Убираем несериализуемые объекты
        config_dict.pop('BASE_DIR', None)
        config_dict.pop('OUTPUTS_DIR', None)
        config_dict.pop('LOGS_DIR', None)
        config_dict.pop('REPORTS_DIR', None)
        config_dict.pop('PLOTS_DIR', None)

        config_path = cls.OUTPUTS_DIR / filename
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        cls._log(f"Конфигурация сохранена: {config_path}")
        return config_path


# Инициализация директорий при импорте
EPOSConfig.init_directories()