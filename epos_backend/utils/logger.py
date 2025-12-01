"""
Кастомный логгер для системы ЭПОС
Разработано командой TechLitCode
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from config.settings import EPOSConfig
import traceback
import json


class EPOSLogger:
    """Кастомный логгер для проекта ЭПОС"""

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EPOSLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.setup_logging()

    def setup_logging(self):
        """Настройка системы логирования"""
        # Создаем логгер
        self.logger = logging.getLogger("EPOS")
        self.logger.setLevel(getattr(logging, EPOSConfig.LOG_LEVEL))

        # Очищаем существующие обработчики
        self.logger.handlers.clear()

        # Форматтер
        formatter = logging.Formatter(
            EPOSConfig.LOG_FORMAT,
            datefmt=EPOSConfig.LOG_DATE_FORMAT
        )

        # Обработчик для консоли
        if EPOSConfig.LOG_TO_CONSOLE:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, EPOSConfig.LOG_LEVEL))
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # Обработчик для файла
        if EPOSConfig.LOG_TO_FILE:
            log_file = EPOSConfig.LOGS_DIR / f"epos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            self.log_file = log_file
            self.logger.info(f"Лог-файл создан: {log_file}")

        # Перехват непойманных исключений
        sys.excepthook = self.handle_uncaught_exception

        self.logger.info("=" * 60)
        self.logger.info("Система логирования ЭПОС инициализирована")
        self.logger.info(f"Уровень логирования: {EPOSConfig.LOG_LEVEL}")
        self.logger.info(f"Логи в файл: {EPOSConfig.LOG_TO_FILE}")
        self.logger.info(f"Логи в консоль: {EPOSConfig.LOG_TO_CONSOLE}")
        self.logger.info("=" * 60)

    def handle_uncaught_exception(self, exc_type, exc_value, exc_traceback):
        """Обработка непойманных исключений"""
        self.logger.critical(
            "Непойманное исключение:",
            exc_info=(exc_type, exc_value, exc_traceback)
        )

        # Сохраняем дополнительную информацию об ошибке
        error_info = {
            "timestamp": datetime.now().isoformat(),
            "exception_type": str(exc_type.__name__),
            "exception_message": str(exc_value),
            "traceback": traceback.format_exception(exc_type, exc_value, exc_traceback)
        }

        error_file = EPOSConfig.LOGS_DIR / f"error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_info, f, indent=2, ensure_ascii=False)

        self.logger.error(f"Детали ошибки сохранены в: {error_file}")

    def log_system_start(self):
        """Логирование запуска системы"""
        self.logger.info(EPOSConfig.get_branding_header())
        self.logger.info(f"Запуск системы ЭПОС v{EPOSConfig.VERSION}")
        self.logger.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Рабочая директория: {EPOSConfig.BASE_DIR}")

    def log_optimization_start(self, equipment_name: str, horizon: int):
        """Логирование начала оптимизации"""
        self.logger.info(f"{'=' * 40}")
        self.logger.info(f"НАЧАЛО ОПТИМИЗАЦИИ")
        self.logger.info(f"Оборудование: {equipment_name}")
        self.logger.info(f"Горизонт планирования: {horizon} часов")
        self.logger.info(f"Время: {datetime.now().strftime('%H:%M:%S')}")
        self.logger.info(f"{'=' * 40}")

    def log_optimization_result(self, result: dict):
        """Логирование результатов оптимизации"""
        self.logger.info(f"{'=' * 40}")
        self.logger.info(f"РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")

        if result.get("success"):
            self.logger.info(f"✓ Оптимизация завершена успешно")
            self.logger.info(f"  Статус: {result.get('status', 'N/A')}")
            self.logger.info(f"  Время решения: {result.get('solve_time', 0):.2f} сек")

            if "savings" in result:
                savings = result["savings"]
                self.logger.info(f"  Экономия: {savings.get('absolute', 0):.2f} руб.")
                self.logger.info(f"  Процент экономии: {savings.get('percent', 0):.2f}%")
        else:
            self.logger.error(f"✗ Оптимизация не удалась")
            self.logger.error(f"  Ошибка: {result.get('error', 'Неизвестная ошибка')}")
            self.logger.error(f"  Статус: {result.get('status', 'N/A')}")

        self.logger.info(f"{'=' * 40}")

    def log_data_generation(self, data_type: str, params: dict):
        """Логирование генерации данных"""
        self.logger.info(f"Генерация данных типа: {data_type}")
        for key, value in params.items():
            self.logger.info(f"  {key}: {value}")

    def log_scenario_start(self, scenario_name: str, description: str):
        """Логирование запуска сценария"""
        self.logger.info(f"\n{'#' * 60}")
        self.logger.info(f"СЦЕНАРИЙ: {scenario_name}")
        self.logger.info(f"Описание: {description}")
        self.logger.info(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"{'#' * 60}\n")

    def performance_log(self, operation: str, duration: float):
        """Логирование производительности"""
        if duration > 1.0:
            self.logger.warning(f"Медленная операция: {operation} заняла {duration:.2f} сек")
        else:
            self.logger.debug(f"Операция: {operation} заняла {duration:.2f} сек")

    def get_logger(self):
        """Получить объект логгера"""
        return self.logger


# Создаем глобальный экземпляр логгера
logger = EPOSLogger().get_logger()


# Удобные функции для быстрого доступа
def log_info(message: str):
    logger.info(message)


def log_error(message: str, exc_info=False):
    logger.error(message, exc_info=exc_info)


def log_warning(message: str):
    logger.warning(message)


def log_debug(message: str):
    logger.debug(message)


def log_critical(message: str):
    logger.critical(message)