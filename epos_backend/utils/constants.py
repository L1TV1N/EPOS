"""
Константы системы ЭПОС
Разработано командой TechLitCode
"""

from enum import Enum

class TimeIntervals(Enum):
    """Временные интервалы"""
    HOURLY = 1
    HALF_HOUR = 0.5
    QUARTER_HOUR = 0.25
    TEN_MINUTES = 0.1667
    FIVE_MINUTES = 0.0833


class EquipmentTypes(Enum):
    """Типы оборудования"""
    COMPRESSOR = "compressor"
    PUMP = "pump"
    OVEN = "oven"
    VENTILATION = "ventilation"
    CONVEYOR = "conveyor"
    LIGHTING = "lighting"
    HVAC = "hvac"  # Отопление, вентиляция, кондиционирование


class OptimizationObjectives(Enum):
    """Целевые функции оптимизации"""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_PEAK = "minimize_peak"
    MAXIMIZE_PROFIT = "maximize_profit"
    MINIMIZE_EMISSIONS = "minimize_emissions"
    BALANCED = "balanced"


class PriceTypes(Enum):
    """Типы цен на энергию"""
    DAY_AHEAD = "day_ahead"      # На сутки вперед
    REAL_TIME = "real_time"      # В реальном времени
    PEAK = "peak"                # Пиковые цены
    OFF_PEAK = "off_peak"        # Внепиковые цены
    NIGHT = "night"              # Ночной тариф


# Цветовая схема TechLitCode для графиков
COLOR_SCHEME = {
    "primary": "#1B2B4E",      # Темно-синий
    "accent": "#F5C32C",       # Золотой
    "secondary": "#2C3E50",    # Серо-синий
    "success": "#27AE60",      # Зеленый
    "warning": "#E67E22",      # Оранжевый
    "danger": "#E74C3C",       # Красный
    "info": "#3498DB",         # Голубой
    "light": "#ECF0F1",        # Светло-серый
    "dark": "#2C3E50",         # Темно-серый
    "grid": "#BDC3C7"          # Цвет сетки
}

# Константы для расчетов
SECONDS_IN_HOUR = 3600
HOURS_IN_DAY = 24
DAYS_IN_WEEK = 7
MONTHS_IN_YEAR = 12

# Коэффициенты преобразования
RUB_PER_KWH = 4.0  # Средняя цена, будет переопределена реальными данными
CO2_PER_KWH = 0.5  # кг CO2 на кВт*ч (среднее значение для РФ)

# Параметры по умолчанию
DEFAULT_OPTIMIZATION_PARAMS = {
    "horizon": 24,
    "interval": 1,
    "objective": "minimize_cost",
    "consider_startup": True,
    "consider_ramping": True,
    "allow_curtailment": False,
    "storage_enabled": False,
    "renewables_enabled": False
}

# Сообщения об ошибках
ERROR_MESSAGES = {
    "NO_DATA": "Отсутствуют данные для анализа",
    "INVALID_PROFILE": "Неверный профиль оборудования",
    "OPTIMIZATION_FAILED": "Оптимизация не удалась",
    "PRICE_DATA_MISSING": "Отсутствуют данные о ценах",
    "CONSTRAINT_VIOLATION": "Нарушение ограничений оборудования",
    "SOLVER_ERROR": "Ошибка решателя оптимизации",
    "MEMORY_ERROR": "Недостаточно памяти для обработки",
    "TIMEOUT": "Превышено время выполнения"
}

# Статусы выполнения
STATUS_CODES = {
    "SUCCESS": 0,
    "WARNING": 1,
    "ERROR": 2,
    "CRITICAL": 3
}

# Форматы файлов
FILE_FORMATS = {
    "CSV": ".csv",
    "JSON": ".json",
    "EXCEL": ".xlsx",
    "PDF": ".pdf",
    "PNG": ".png",
    "SVG": ".svg"
}

# Единицы измерения
UNITS = {
    "power": "кВт",
    "energy": "кВт*ч",
    "cost": "руб.",
    "time": "час",
    "temperature": "°C",
    "pressure": "бар",
    "flow": "м³/ч",
    "efficiency": "%"
}

# Константы для энергорынка
MARKET_CONSTANTS = {
    "PEAK_HOURS": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    "NIGHT_HOURS": [0, 1, 2, 3, 4, 5, 6],
    "WEEKEND_DISCOUNT": 0.15,  # Скидка в выходные
    "SEASONAL_MULTIPLIER": {
        "winter": 1.2,
        "spring": 1.0,
        "summer": 1.1,
        "autumn": 1.0
    }
}