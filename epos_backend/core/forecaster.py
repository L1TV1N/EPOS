"""
Прогнозные модели системы ЭПОС
Разработано командой TechLitCode
"""
from pathlib import Path
import json
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

from epos_backend.config.settings import EPOSConfig
from epos_backend.utils.logger import log_info, log_error, log_warning, log_debug
from epos_backend.utils.helpers import generate_timestamps, calculate_statistics, normalize_data
from epos_backend.utils import constants


@dataclass
class ForecastResult:
    """Результат прогнозирования"""
    success: bool
    model_name: str
    predictions: List[float]
    actuals: Optional[List[float]] = None
    horizon: int = 24
    metrics: Optional[Dict[str, float]] = None
    confidence_intervals: Optional[List[Tuple[float, float]]] = None
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'success': self.success,
            'model_name': self.model_name,
            'horizon': self.horizon,
            'predictions': self.predictions,
            'actuals': self.actuals,
            'metrics': self.metrics,
            'confidence_intervals': self.confidence_intervals,
            'feature_importance': self.feature_importance,
            'metadata': self.metadata or {}
        }

    def print_summary(self):
        """Печать сводки результатов"""
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ")
        print("=" * 60)
        print(f"Модель: {self.model_name}")
        print(f"Горизонт: {self.horizon} часов")

        if self.metrics:
            print(f"\nМетрики качества:")
            for metric, value in self.metrics.items():
                print(f"  {metric}: {value:.4f}")

        if self.predictions:
            stats = calculate_statistics(self.predictions)
            print(f"\nСтатистика прогноза:")
            print(f"  Среднее: {stats['mean']:.2f}")
            print(f"  Минимум: {stats['min']:.2f}")
            print(f"  Максимум: {stats['max']:.2f}")


class EPOSForecaster:
    """Система прогнозирования для ЭПОС"""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Инициализация системы прогнозирования

        Args:
            model_type: Тип модели (random_forest, gradient_boosting, linear, ensemble)
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

        self._initialize_model()

        log_info(f"Инициализация системы прогнозирования")
        log_info(f"Тип модели: {model_type}")
        log_info(f"Разработано командой {EPOSConfig.DEVELOPER}")

    def _initialize_model(self):
        """Инициализация модели машинного обучения"""
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(alpha=1.0)
        elif self.model_type == 'ensemble':
            # Ансамбль моделей
            self.model = {
                'rf': RandomForestRegressor(n_estimators=50, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=50, random_state=42),
                'linear': LinearRegression()
            }
        else:
            log_warning(f"Неизвестный тип модели: {self.model_type}. Используется Random Forest.")
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def prepare_features(self,
                         historical_data: pd.DataFrame,
                         horizon: int = 24) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Подготовка признаков для прогнозирования

        Args:
            historical_data: Исторические данные
            horizon: Горизонт прогнозирования

        Returns:
            Матрица признаков, целевая переменная, имена признаков
        """
        log_info(f"Подготовка признаков для прогнозирования на {horizon} часов")

        # Создание копии данных
        df = historical_data.copy()

        # Базовые временные признаки
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['day_of_month'] = df['timestamp'].dt.day
            df['month'] = df['timestamp'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

        # Лаговые признаки (значения в предыдущие часы)
        target_col = 'load_kw' if 'load_kw' in df.columns else df.columns[0]

        for lag in [1, 2, 3, 24, 48, 168]:  # 1ч, 2ч, 3ч, сутки, двое суток, неделя
            if len(df) > lag:
                df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)

        # Скользящие статистики
        for window in [3, 6, 12, 24]:
            if len(df) > window:
                df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
                df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()

        # Сезонные признаки
        if 'hour' in df.columns:
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        if 'day_of_week' in df.columns:
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # Взаимодействия
        if 'hour' in df.columns and 'is_weekend' in df.columns:
            df['hour_weekend_interaction'] = df['hour'] * df['is_weekend']

        # Целевая переменная (сдвинутая на горизонт)
        df['target'] = df[target_col].shift(-horizon)

        # Удаление строк с NaN
        df_clean = df.dropna()

        if df_clean.empty:
            raise ValueError("Недостаточно данных для создания признаков")

        # Разделение на признаки и целевую переменную
        feature_cols = [col for col in df_clean.columns if col not in ['timestamp', 'target', target_col]]
        X = df_clean[feature_cols].values
        y = df_clean['target'].values

        # Нормализация признаков
        X_scaled = self.scaler.fit_transform(X)

        self.feature_names = feature_cols
        log_debug(f"Создано {len(feature_cols)} признаков")
        log_debug(f"Размерность данных: X={X_scaled.shape}, y={y.shape}")

        return X_scaled, y, feature_cols

    def train(self,
              historical_data: pd.DataFrame,
              horizon: int = 24,
              test_size: float = 0.2) -> ForecastResult:
        """
        Обучение модели прогнозирования

        Args:
            historical_data: Исторические данные
            horizon: Горизонт прогнозирования
            test_size: Доля тестовых данных

        Returns:
            Результат обучения
        """
        log_info(f"Обучение модели {self.model_type} на исторических данных")

        try:
            # Подготовка данных
            X, y, feature_names = self.prepare_features(historical_data, horizon)

            if len(X) < 100:
                log_warning(f"Мало данных для обучения: {len(X)} образцов")

            # Разделение на train/test с учетом временного ряда
            n_samples = len(X)
            n_train = int(n_samples * (1 - test_size))

            X_train, X_test = X[:n_train], X[n_train:]
            y_train, y_test = y[:n_train], y[n_train:]

            # Обучение модели
            if self.model_type == 'ensemble':
                # Обучение ансамбля
                predictions = []
                for name, model in self.model.items():
                    log_debug(f"Обучение {name}...")
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    predictions.append(pred)

                # Усреднение прогнозов
                y_pred = np.mean(predictions, axis=0)
            else:
                # Обучение одиночной модели
                self.model.fit(X_train, y_train)
                y_pred = self.model.predict(X_test)

            # Расчет метрик
            metrics = self._calculate_metrics(y_test, y_pred)

            # Важность признаков
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                importance = self.model.feature_importances_
                feature_importance = dict(zip(feature_names, importance))
            elif self.model_type == 'ensemble':
                # Средняя важность по ансамблю
                importance_sum = np.zeros(len(feature_names))
                for model in self.model.values():
                    if hasattr(model, 'feature_importances_'):
                        importance_sum += model.feature_importances_
                feature_importance = dict(zip(feature_names, importance_sum / len(self.model)))

            self.is_trained = True

            result = ForecastResult(
                success=True,
                model_name=self.model_type,
                predictions=y_pred.tolist(),
                actuals=y_test.tolist(),
                horizon=horizon,
                metrics=metrics,
                feature_importance=feature_importance,
                metadata={
                    'train_samples': len(X_train),
                    'test_samples': len(X_test),
                    'feature_count': len(feature_names),
                    'training_date': datetime.now().isoformat()
                }
            )

            log_info(f"Модель обучена успешно")
            log_info(f"Метрики: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, R²={metrics['r2']:.3f}")

            return result

        except Exception as e:
            log_error(f"Ошибка обучения модели: {e}")
            return ForecastResult(
                success=False,
                model_name=self.model_type,
                predictions=[],
                horizon=horizon,
                metadata={'error': str(e)}
            )

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Расчет метрик качества прогноза"""
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': self._calculate_mape(y_true, y_pred)
        }
        return metrics

    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Расчет Mean Absolute Percentage Error"""
        # Избегаем деления на ноль
        mask = y_true != 0
        if np.sum(mask) > 0:
            return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return 0.0

    def forecast(self,
                 historical_data: pd.DataFrame,
                 horizon: int = 24,
                 n_steps: int = 1) -> ForecastResult:
        """
        Прогнозирование на несколько шагов вперед

        Args:
            historical_data: Исторические данные
            horizon: Горизонт прогнозирования
            n_steps: Количество шагов прогноза

        Returns:
            Результат прогнозирования
        """
        if not self.is_trained:
            log_warning("Модель не обучена. Выполняется обучение...")
            self.train(historical_data, horizon)

        log_info(f"Прогнозирование на {n_steps} шагов вперед")

        try:
            # Подготовка последних известных данных
            X_last, _, feature_names = self.prepare_features(historical_data, horizon)

            if len(X_last) == 0:
                raise ValueError("Недостаточно данных для прогнозирования")

            # Прогнозирование
            predictions = []
            confidence_intervals = []

            # Используем последнюю точку данных
            X_current = X_last[-1:].copy()

            for step in range(n_steps):
                # Прогноз на один шаг
                if self.model_type == 'ensemble':
                    step_predictions = []
                    for model in self.model.values():
                        pred = model.predict(X_current)
                        step_predictions.append(pred[0])
                    y_pred = np.mean(step_predictions)
                    y_std = np.std(step_predictions)
                else:
                    y_pred = self.model.predict(X_current)[0]
                    y_std = self._estimate_uncertainty(X_current)

                predictions.append(float(y_pred))

                # Доверительный интервал (95%)
                lower = y_pred - 1.96 * y_std
                upper = y_pred + 1.96 * y_std
                confidence_intervals.append((float(lower), float(upper)))

                # Обновление признаков для следующего шага
                # (в реальной системе нужно обновить исторические данные)

            result = ForecastResult(
                success=True,
                model_name=self.model_type,
                predictions=predictions,
                horizon=horizon,
                confidence_intervals=confidence_intervals,
                metadata={
                    'forecast_steps': n_steps,
                    'forecast_date': datetime.now().isoformat(),
                    'last_data_point': historical_data.index[-1] if hasattr(historical_data, 'index') else 'unknown'
                }
            )

            log_info(f"Прогноз успешно сгенерирован")
            return result

        except Exception as e:
            log_error(f"Ошибка прогнозирования: {e}")
            return ForecastResult(
                success=False,
                model_name=self.model_type,
                predictions=[],
                horizon=horizon,
                metadata={'error': str(e)}
            )

    def _estimate_uncertainty(self, X: np.ndarray) -> float:
        """Оценка неопределенности прогноза"""
        # Упрощенная оценка неопределенности
        # В реальной системе можно использовать bootstrap или другие методы

        if hasattr(self.model, 'estimators_'):
            # Для Random Forest используем std предсказаний деревьев
            tree_predictions = []
            for estimator in self.model.estimators_:
                tree_predictions.append(estimator.predict(X)[0])
            return np.std(tree_predictions)
        else:
            # Для других моделей - эвристическая оценка
            return 0.1 * np.mean(X) if np.mean(X) > 0 else 1.0

    def forecast_load(self,
                      equipment_type: str,
                      hours: int = 24,
                      include_weather: bool = True) -> ForecastResult:
        """
        Прогнозирование нагрузки оборудования

        Args:
            equipment_type: Тип оборудования
            hours: Горизонт прогнозирования
            include_weather: Включать ли погодные данные

        Returns:
            Результат прогноза нагрузки
        """
        log_info(f"Прогнозирование нагрузки для {equipment_type} на {hours} часов")

        try:
            # Генерация исторических данных
            from data.generator import DataGenerator
            generator = DataGenerator()

            # Генерация данных за последнюю неделю
            historical_hours = 168  # 7 дней
            dataset = generator.generate_complete_dataset(
                equipment_type=equipment_type,
                hours=historical_hours,
                save_to_file=False
            )

            # Создание DataFrame
            df = pd.DataFrame({
                'timestamp': pd.date_range(end=datetime.now(), periods=historical_hours, freq='H'),
                'load_kw': dataset['load_profile_kw'],
                'price': dataset['price_profile_rub_kwh'],
                'temperature': dataset['weather']['temperature_c'],
                'humidity': dataset['weather']['humidity_percent']
            })

            # Обучение модели
            result = self.train(df, horizon=hours)

            if result.success:
                # Прогнозирование
                forecast_result = self.forecast(df, horizon=hours, n_steps=hours)

                # Добавление контекстной информации
                if forecast_result.metadata:
                    forecast_result.metadata.update({
                        'equipment_type': equipment_type,
                        'include_weather': include_weather,
                        'historical_hours': historical_hours
                    })

                return forecast_result
            else:
                return result

        except Exception as e:
            log_error(f"Ошибка прогнозирования нагрузки: {e}")
            return ForecastResult(
                success=False,
                model_name=self.model_type,
                predictions=[],
                horizon=hours,
                metadata={'error': str(e)}
            )

    def forecast_prices(self,
                        market: str = 'ATS',
                        hours: int = 24) -> ForecastResult:
        """
        Прогнозирование цен на электроэнергию

        Args:
            market: Рынок
            hours: Горизонт прогнозирования

        Returns:
            Результат прогноза цен
        """
        log_info(f"Прогнозирование цен на рынке {market} на {hours} часов")

        try:
            # Загрузка исторических данных
            from data.price_loader import PriceLoader
            loader = PriceLoader(market=market)

            # Загрузка данных за последний месяц
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)

            historical_data = loader.load_historical_prices(start_date, end_date, save_to_file=False)

            if 'data' not in historical_data:
                raise ValueError("Не удалось загрузить исторические данные")

            # Создание DataFrame
            price_data = historical_data['data']
            df = pd.DataFrame(price_data)

            # Переименование столбцов
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.rename(columns={'price_rub_kwh': 'price'})
            df = df[['timestamp', 'price']].sort_values('timestamp')

            # Обучение модели
            result = self.train(df, horizon=hours)

            if result.success:
                # Прогнозирование
                forecast_result = self.forecast(df, horizon=hours, n_steps=hours)

                # Добавление контекстной информации
                if forecast_result.metadata:
                    forecast_result.metadata.update({
                        'market': market,
                        'forecast_type': 'price'
                    })

                return forecast_result
            else:
                return result

        except Exception as e:
            log_error(f"Ошибка прогнозирования цен: {e}")
            return ForecastResult(
                success=False,
                model_name=self.model_type,
                predictions=[],
                horizon=hours,
                metadata={'error': str(e)}
            )

    def save_forecast_report(self,
                             forecast_result: ForecastResult,
                             filename: str = None) -> Dict[str, Path]:
        """
        Сохранение отчета о прогнозировании

        Args:
            forecast_result: Результат прогнозирования
            filename: Имя файла

        Returns:
            Словарь с путями к сохраненным файлам
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = forecast_result.model_name
            filename = f"forecast_{model_name}_{timestamp}"

        # Сохранение в JSON
        report_data = forecast_result.to_dict()
        json_path = save_to_json(report_data, f"{filename}.json")

        # Сохранение прогноза в CSV
        if forecast_result.predictions:
            csv_data = []
            for i, pred in enumerate(forecast_result.predictions):
                row = {'hour': i, 'forecast': pred}

                if forecast_result.actuals and i < len(forecast_result.actuals):
                    row['actual'] = forecast_result.actuals[i]
                    row['error'] = forecast_result.actuals[i] - pred

                if forecast_result.confidence_intervals and i < len(forecast_result.confidence_intervals):
                    row['confidence_lower'] = forecast_result.confidence_intervals[i][0]
                    row['confidence_upper'] = forecast_result.confidence_intervals[i][1]

                csv_data.append(row)

            csv_path = save_to_csv({'forecast': csv_data}, f"{filename}.csv")
        else:
            csv_path = None

        # Сохранение важности признаков
        if forecast_result.feature_importance:
            importance_data = [
                {'feature': feat, 'importance': imp}
                for feat, imp in sorted(forecast_result.feature_importance.items(),
                                        key=lambda x: x[1], reverse=True)[:20]  # Топ-20 признаков
            ]
            importance_path = save_to_csv({'feature_importance': importance_data},
                                          f"{filename}_importance.csv")
        else:
            importance_path = None

        log_info(f"Отчет о прогнозировании сохранен:")
        log_info(f"  JSON: {json_path}")
        if csv_path:
            log_info(f"  CSV прогноз: {csv_path}")
        if importance_path:
            log_info(f"  CSV важность признаков: {importance_path}")

        return {
            'json': json_path,
            'forecast_csv': csv_path,
            'importance_csv': importance_path
        }


# Утилитарные функции для быстрого доступа
def quick_load_forecast(equipment_type: str = "compressor",
                        hours: int = 24) -> ForecastResult:
    """Быстрый прогноз нагрузки"""
    forecaster = EPOSForecaster(model_type='random_forest')
    return forecaster.forecast_load(equipment_type, hours)


def compare_forecast_models(historical_data: pd.DataFrame,
                            horizon: int = 24,
                            models: List[str] = None) -> Dict[str, ForecastResult]:
    """Сравнение различных моделей прогнозирования"""
    if models is None:
        models = ['random_forest', 'gradient_boosting', 'linear', 'ridge']

    results = {}

    for model_type in models:
        log_info(f"Тестирование модели: {model_type}")

        forecaster = EPOSForecaster(model_type=model_type)
        result = forecaster.train(historical_data, horizon=horizon)

        results[model_type] = result

    # Анализ сравнения
    comparison = {
        'best_model': None,
        'best_mae': float('inf'),
        'model_rankings': []
    }

    for model_name, result in results.items():
        if result.success and result.metrics:
            mae = result.metrics['mae']
            comparison['model_rankings'].append({
                'model': model_name,
                'mae': mae,
                'r2': result.metrics['r2']
            })

            if mae < comparison['best_mae']:
                comparison['best_mae'] = mae
                comparison['best_model'] = model_name

    # Сортировка по качеству
    comparison['model_rankings'].sort(key=lambda x: x['mae'])

    return results, comparison


def create_baseline_forecast(historical_data: pd.DataFrame,
                             horizon: int = 24,
                             method: str = 'naive') -> List[float]:
    """
    Создание базового прогноза (baseline)

    Args:
        historical_data: Исторические данные
        horizon: Горизонт прогнозирования
        method: Метод (naive, seasonal, moving_average)

    Returns:
        Базовый прогноз
    """
    if 'load_kw' in historical_data.columns:
        series = historical_data['load_kw'].values
    else:
        series = historical_data.iloc[:, 0].values

    if method == 'naive':
        # Наивный прогноз: последнее значение
        forecast = [series[-1]] * horizon

    elif method == 'seasonal':
        # Сезонный прогноз: среднее за тот же час в предыдущие дни
        forecast = []
        if len(series) >= 24:
            for h in range(horizon):
                hour_of_day = h % 24
                # Берем значения за тот же час в последние 7 дней
                same_hour_values = []
                for day in range(1, 8):
                    idx = len(series) - (day * 24) + hour_of_day
                    if idx >= 0:
                        same_hour_values.append(series[idx])
                if same_hour_values:
                    forecast.append(np.mean(same_hour_values))
                else:
                    forecast.append(np.mean(series))
        else:
            forecast = [np.mean(series)] * horizon

    elif method == 'moving_average':
        # Скользящее среднее
        window = min(24, len(series))
        last_avg = np.mean(series[-window:])
        forecast = [last_avg] * horizon

    else:
        raise ValueError(f"Неизвестный метод: {method}")

    return forecast