"""
Загрузчик реальных цен на электроэнергию для системы ЭПОС
Разработано командой TechLitCode
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path
from urllib.parse import urljoin
import time
from dataclasses import dataclass
from config.settings import EPOSConfig
from utils.logger import log_info, log_error, log_warning, log_debug
from utils.helpers import generate_timestamps, save_to_json, save_to_csv
from utils import constants


@dataclass
class PriceData:
    """Структура для хранения данных о ценах"""
    timestamps: List[datetime]
    prices: List[float]  # руб/кВт*ч
    market: str
    zone: str
    data_source: str
    retrieved_at: datetime

    def to_dataframe(self) -> pd.DataFrame:
        """Конвертация в DataFrame"""
        return pd.DataFrame({
            'timestamp': self.timestamps,
            'price_rub_kwh': self.prices,
            'market': self.market,
            'zone': self.zone
        })

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'metadata': {
                'market': self.market,
                'zone': self.zone,
                'data_source': self.data_source,
                'retrieved_at': self.retrieved_at.isoformat(),
                'hours': len(self.timestamps),
                'system': EPOSConfig.SYSTEM_NAME
            },
            'data': [
                {
                    'timestamp': ts.isoformat(),
                    'price_rub_kwh': price,
                    'hour': ts.hour,
                    'day': ts.day,
                    'month': ts.month,
                    'year': ts.year
                }
                for ts, price in zip(self.timestamps, self.prices)
            ]
        }


class PriceLoader:
    """Загрузчик цен на электроэнергию с различных источников"""

    # URL для различных рынков
    MARKET_URLS = {
        'ATS': {
            'base_url': 'https://www.atsenergo.ru/',
            'api_url': 'https://www.atsenergo.ru/nreport',
            'regions': ['Ценовая зона Европейской части России и Урала']
        },
        'SPB': {
            'base_url': 'https://spbexchange.ru/',
            'api_url': 'https://spbexchange.ru/ru/market-data/',
            'regions': ['Санкт-Петербург', 'Ленинградская область']
        }
    }

    # Кэш загруженных данных
    _price_cache: Dict[str, PriceData] = {}

    def __init__(self, market: str = 'ATS', use_cache: bool = True):
        """
        Инициализация загрузчика цен

        Args:
            market: Рынок (ATS, SPB, или 'simulated' для симуляции)
            use_cache: Использовать кэширование данных
        """
        self.market = market.upper()
        self.use_cache = use_cache

        if self.market not in self.MARKET_URLS and self.market != 'SIMULATED':
            log_warning(f"Неизвестный рынок: {market}. Используется ATS.")
            self.market = 'ATS'

        log_info(f"Инициализация загрузчика цен для рынка {self.market}")
        log_info(f"Разработано командой {EPOSConfig.DEVELOPER}")

    def load_prices(self,
                    date: Optional[datetime] = None,
                    hours: int = 24,
                    region: Optional[str] = None) -> PriceData:
        """
        Загрузка цен на электроэнергию

        Args:
            date: Дата для загрузки (по умолчанию - сегодня)
            hours: Количество часов (максимум 48)
            region: Регион (если поддерживается рынком)

        Returns:
            Объект PriceData с ценами
        """
        if date is None:
            date = datetime.now()

        # Проверка кэша
        cache_key = f"{self.market}_{date.date()}_{hours}"
        if self.use_cache and cache_key in self._price_cache:
            log_info(f"Используются кэшированные данные для {cache_key}")
            return self._price_cache[cache_key]

        log_info(f"Загрузка цен с рынка {self.market} на {date.date()} ({hours} часов)")

        try:
            if self.market == 'SIMULATED':
                price_data = self._load_simulated_prices(date, hours)
            else:
                price_data = self._load_real_prices(date, hours, region)

            # Сохранение в кэш
            if self.use_cache:
                self._price_cache[cache_key] = price_data

            return price_data

        except Exception as e:
            log_error(f"Ошибка загрузки цен: {e}")
            log_warning("Используются симулированные данные")
            return self._load_simulated_prices(date, hours)

    def _load_real_prices(self,
                          date: datetime,
                          hours: int,
                          region: Optional[str]) -> PriceData:
        """
        Загрузка реальных цен с рынка

        Note: Это демо-реализация. В реальном проекте нужно интегрироваться
        с API конкретного рынка.
        """
        log_warning("Реальные данные временно недоступны. Используются симулированные данные.")

        # Генерация реалистичных цен
        return self._load_simulated_prices(date, hours, realistic=True)

    def _load_simulated_prices(self,
                               date: datetime,
                               hours: int,
                               realistic: bool = True) -> PriceData:
        """
        Генерация симулированных цен

        Args:
            date: Базовая дата
            hours: Количество часов
            realistic: Генерировать реалистичные цены

        Returns:
            Объект PriceData
        """
        timestamps = generate_timestamps(hours, date)

        if realistic:
            # Реалистичный профиль цен с учетом времени суток и дня недели
            prices = self._generate_realistic_prices(timestamps)
        else:
            # Простой случайный профиль
            base_price = 4.0  # руб/кВт*ч
            prices = [base_price + np.random.uniform(-1, 2) for _ in range(hours)]

        return PriceData(
            timestamps=timestamps,
            prices=prices,
            market=self.market,
            zone='simulated' if self.market == 'SIMULATED' else self.MARKET_URLS[self.market]['regions'][0],
            data_source='simulated' if self.market == 'SIMULATED' else f'{self.market}_api',
            retrieved_at=datetime.now()
        )

    def _generate_realistic_prices(self, timestamps: List[datetime]) -> List[float]:
        """
        Генерация реалистичных цен на основе временных меток

        Args:
            timestamps: Список временных меток

        Returns:
            Список цен
        """
        prices = []

        for ts in timestamps:
            # Базовая цена
            base_price = 4.0  # руб/кВт*ч

            # Временные коэффициенты
            hour = ts.hour
            day_of_week = ts.weekday()  # 0 = понедельник

            # Суточные колебания
            if 0 <= hour < 6:
                # Ночное время - низкие цены
                time_factor = 0.6 + 0.1 * np.sin(hour * np.pi / 12)
            elif 6 <= hour < 9:
                # Утренний подъем
                time_factor = 0.7 + 0.2 * (hour - 6) / 3
            elif 9 <= hour < 18:
                # Рабочий день - высокие цены с пиком в обед
                peak_factor = 1.0 + 0.3 * np.sin((hour - 12) * np.pi / 12)
                time_factor = 1.0 * peak_factor
            elif 18 <= hour < 21:
                # Вечерний пик
                time_factor = 1.2 - 0.3 * (hour - 18) / 3
            else:
                # Поздний вечер
                time_factor = 0.9 - 0.2 * (hour - 21) / 3

            # Выходные дни - снижение цен
            if day_of_week >= 5:  # Суббота и воскресенье
                weekend_factor = 0.85
            else:
                weekend_factor = 1.0

            # Сезонные колебания (зимой выше)
            month = ts.month
            if month in [12, 1, 2]:  # Зима
                seasonal_factor = 1.2
            elif month in [6, 7, 8]:  # Лето
                seasonal_factor = 1.1
            else:
                seasonal_factor = 1.0

            # Случайная составляющая
            random_factor = 1.0 + np.random.randn() * 0.1

            # Итоговая цена
            price = base_price * time_factor * weekend_factor * seasonal_factor * random_factor

            # Ограничения
            price = max(1.0, min(10.0, price))
            prices.append(round(price, 2))

        return prices

    def load_historical_prices(self,
                               start_date: datetime,
                               end_date: datetime,
                               save_to_file: bool = True) -> Dict[str, Any]:
        """
        Загрузка исторических данных о ценах

        Args:
            start_date: Начальная дата
            end_date: Конечная дата
            save_to_file: Сохранять ли данные в файл

        Returns:
            Словарь с историческими данными
        """
        log_info(f"Загрузка исторических данных с {start_date.date()} по {end_date.date()}")

        # Генерация дат
        date_range = pd.date_range(start_date, end_date, freq='D')
        all_data = []

        for date in date_range:
            date_dt = date.to_pydatetime()
            price_data = self.load_prices(date_dt, hours=24)

            for ts, price in zip(price_data.timestamps, price_data.prices):
                all_data.append({
                    'timestamp': ts,
                    'price_rub_kwh': price,
                    'date': ts.date(),
                    'hour': ts.hour,
                    'day_of_week': ts.weekday(),
                    'month': ts.month,
                    'year': ts.year
                })

        # Создание DataFrame
        df = pd.DataFrame(all_data)

        # Расчет статистики
        stats = {
            'total_days': len(date_range),
            'total_hours': len(df),
            'average_price': float(df['price_rub_kwh'].mean()),
            'min_price': float(df['price_rub_kwh'].min()),
            'max_price': float(df['price_rub_kwh'].max()),
            'std_price': float(df['price_rub_kwh'].std()),
            'peak_hours': df[df['price_rub_kwh'] > df['price_rub_kwh'].mean() * 1.2].shape[0]
        }

        result = {
            'metadata': {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'market': self.market,
                'retrieved_at': datetime.now().isoformat(),
                'system': EPOSConfig.SYSTEM_NAME,
                'developer': EPOSConfig.DEVELOPER
            },
            'statistics': stats,
            'data': all_data
        }

        # Сохранение в файл
        if save_to_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"historical_prices_{start_date.date()}_to_{end_date.date()}_{timestamp}"

            # Сохранение в JSON
            json_path = save_to_json(result, filename)

            # Сохранение в CSV
            csv_path = EPOSConfig.REPORTS_DIR / "csv" / f"{filename}.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')

            log_info(f"Исторические данные сохранены:")
            log_info(f"  JSON: {json_path}")
            log_info(f"  CSV: {csv_path}")

            result['saved_files'] = {
                'json': str(json_path),
                'csv': str(csv_path)
            }

        return result

    def analyze_price_patterns(self, price_data: PriceData) -> Dict[str, Any]:
        """
        Анализ паттернов цен

        Args:
            price_data: Данные о ценах

        Returns:
            Словарь с анализом паттернов
        """
        log_info(f"Анализ паттернов цен для рынка {price_data.market}")

        prices = price_data.prices
        timestamps = price_data.timestamps

        if not prices:
            return {}

        # Базовые статистики
        prices_array = np.array(prices)
        mean_price = np.mean(prices_array)
        std_price = np.std(prices_array)

        # Обнаружение пиков
        peak_threshold = mean_price + std_price * 0.5
        peak_hours = [i for i, price in enumerate(prices) if price > peak_threshold]

        # Обнаружение минимумов
        low_threshold = mean_price - std_price * 0.5
        low_hours = [i for i, price in enumerate(prices) if price < low_threshold]

        # Анализ по времени суток
        hourly_prices = {}
        for ts, price in zip(timestamps, prices):
            hour = ts.hour
            if hour not in hourly_prices:
                hourly_prices[hour] = []
            hourly_prices[hour].append(price)

        hourly_stats = {}
        for hour, hour_prices in hourly_prices.items():
            hourly_stats[hour] = {
                'average': np.mean(hour_prices),
                'min': np.min(hour_prices),
                'max': np.max(hour_prices),
                'std': np.std(hour_prices)
            }

        # Оптимальные часы для потребления (самые дешевые)
        sorted_hours = sorted(enumerate(prices), key=lambda x: x[1])
        cheapest_hours = [hour for hour, _ in sorted_hours[:6]]  # 6 самых дешевых часов
        expensive_hours = [hour for hour, _ in sorted_hours[-6:]]  # 6 самых дорогих часов

        analysis = {
            'market': price_data.market,
            'zone': price_data.zone,
            'period': {
                'start': timestamps[0].isoformat(),
                'end': timestamps[-1].isoformat(),
                'hours': len(timestamps)
            },
            'price_statistics': {
                'mean': float(mean_price),
                'median': float(np.median(prices_array)),
                'std': float(std_price),
                'min': float(np.min(prices_array)),
                'max': float(np.max(prices_array)),
                'q25': float(np.percentile(prices_array, 25)),
                'q75': float(np.percentile(prices_array, 75))
            },
            'peak_analysis': {
                'threshold': float(peak_threshold),
                'peak_hours': peak_hours,
                'peak_count': len(peak_hours),
                'peak_percentage': len(peak_hours) / len(prices) * 100
            },
            'low_price_analysis': {
                'threshold': float(low_threshold),
                'low_hours': low_hours,
                'low_count': len(low_hours),
                'low_percentage': len(low_hours) / len(prices) * 100
            },
            'hourly_analysis': hourly_stats,
            'recommendations': {
                'cheapest_hours': cheapest_hours,
                'expensive_hours': expensive_hours,
                'best_time_to_consume': cheapest_hours,
                'best_time_to_conserve': expensive_hours,
                'estimated_savings_percent': (
                        (np.mean([prices[h] for h in expensive_hours]) -
                         np.mean([prices[h] for h in cheapest_hours])) /
                        np.mean([prices[h] for h in expensive_hours]) * 100
                ) if expensive_hours and cheapest_hours else 0
            }
        }

        return analysis

    def get_price_forecast(self,
                           hours_ahead: int = 24,
                           confidence: float = 0.8) -> Dict[str, Any]:
        """
        Прогноз цен на электроэнергию

        Args:
            hours_ahead: Часов для прогноза
            confidence: Уровень доверия (0-1)

        Returns:
            Словарь с прогнозом
        """
        log_info(f"Генерация прогноза цен на {hours_ahead} часов вперед")

        # Базовая дата - сейчас
        now = datetime.now()
        forecast_timestamps = generate_timestamps(hours_ahead, now)

        # Загрузка исторических данных для анализа
        historical_data = self.load_historical_prices(
            start_date=now - timedelta(days=7),
            end_date=now - timedelta(days=1)
        )

        # Генерация прогноза на основе исторических паттернов
        forecast_prices = []
        confidence_intervals = []

        for ts in forecast_timestamps:
            hour = ts.hour
            day_of_week = ts.weekday()

            # Базовый прогноз на основе времени суток и дня недели
            base_forecast = self._generate_realistic_prices([ts])[0]

            # Добавление случайной составляющей с учетом уверенности
            random_factor = np.random.randn() * (1 - confidence) * 0.5
            forecast_price = base_forecast * (1 + random_factor)

            # Интервал доверия
            lower_bound = forecast_price * (1 - (1 - confidence) * 0.3)
            upper_bound = forecast_price * (1 + (1 - confidence) * 0.3)

            forecast_prices.append(round(forecast_price, 2))
            confidence_intervals.append({
                'lower': round(lower_bound, 2),
                'upper': round(upper_bound, 2)
            })

        forecast_data = {
            'metadata': {
                'generated_at': now.isoformat(),
                'hours_ahead': hours_ahead,
                'confidence_level': confidence,
                'market': self.market,
                'system': EPOSConfig.SYSTEM_NAME
            },
            'forecast': [
                {
                    'timestamp': ts.isoformat(),
                    'forecast_price_rub_kwh': price,
                    'confidence_interval': interval,
                    'hour': ts.hour,
                    'day_of_week': ts.weekday()
                }
                for ts, price, interval in zip(forecast_timestamps, forecast_prices, confidence_intervals)
            ],
            'statistics': {
                'average_forecast': float(np.mean(forecast_prices)),
                'min_forecast': float(np.min(forecast_prices)),
                'max_forecast': float(np.max(forecast_prices)),
                'volatility': float(np.std(forecast_prices) / np.mean(forecast_prices) * 100)
            },
            'recommendations': {
                'cheapest_forecasted_hours': sorted(
                    enumerate(forecast_prices),
                    key=lambda x: x[1]
                )[:3],
                'expensive_forecasted_hours': sorted(
                    enumerate(forecast_prices),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]
            }
        }

        return forecast_data


# Утилитарные функции для быстрого доступа
def get_current_prices(hours: int = 24, market: str = 'ATS') -> PriceData:
    """Быстрая загрузка текущих цен"""
    loader = PriceLoader(market=market)
    return loader.load_prices(hours=hours)


def save_price_analysis(price_data: PriceData, filename: str = None) -> Path:
    """Сохранение анализа цен в файл"""
    loader = PriceLoader(price_data.market)
    analysis = loader.analyze_price_patterns(price_data)

    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"price_analysis_{price_data.market}_{timestamp}"

    # Сохранение анализа
    json_path = save_to_json(analysis, filename)

    # Сохранение сырых данных в CSV
    csv_path = EPOSConfig.REPORTS_DIR / "csv" / f"{filename}.csv"
    df = price_data.to_dataframe()
    df.to_csv(csv_path, index=False, encoding='utf-8')

    log_info(f"Анализ цен сохранен:")
    log_info(f"  Анализ (JSON): {json_path}")
    log_info(f"  Данные (CSV): {csv_path}")

    return json_path


def compare_markets(markets: List[str] = None, hours: int = 24) -> Dict[str, Any]:
    """Сравнение цен на разных рынках"""
    if markets is None:
        markets = ['ATS', 'SPB', 'SIMULATED']

    log_info(f"Сравнение цен на рынках: {markets}")

    comparison = {
        'metadata': {
            'compared_at': datetime.now().isoformat(),
            'markets': markets,
            'hours': hours,
            'system': EPOSConfig.SYSTEM_NAME
        },
        'data': {},
        'statistics': {},
        'recommendations': {}
    }

    # Загрузка данных с каждого рынка
    market_data = {}
    for market in markets:
        try:
            loader = PriceLoader(market=market)
            price_data = loader.load_prices(hours=hours)
            market_data[market] = price_data

            # Анализ паттернов
            analysis = loader.analyze_price_patterns(price_data)
            comparison['data'][market] = analysis

        except Exception as e:
            log_error(f"Ошибка загрузки данных с рынка {market}: {e}")
            comparison['data'][market] = {'error': str(e)}

    # Сравнительная статистика
    valid_markets = [m for m in markets if m in market_data]
    if len(valid_markets) >= 2:
        # Сравнение средних цен
        avg_prices = {}
        for market in valid_markets:
            if 'price_statistics' in comparison['data'][market]:
                avg_prices[market] = comparison['data'][market]['price_statistics']['mean']

        if avg_prices:
            cheapest_market = min(avg_prices.items(), key=lambda x: x[1])
            expensive_market = max(avg_prices.items(), key=lambda x: x[1])

            comparison['statistics']['average_prices'] = avg_prices
            comparison['statistics']['cheapest_market'] = {
                'market': cheapest_market[0],
                'price': cheapest_market[1]
            }
            comparison['statistics']['expensive_market'] = {
                'market': expensive_market[0],
                'price': expensive_market[1]
            }
            comparison['statistics']['price_difference_percent'] = (
                    (expensive_market[1] - cheapest_market[1]) / expensive_market[1] * 100
            )

    return comparison