"""
Оптимизационный движок системы ЭПОС
Разработано командой TechLitCode
"""

import pulp
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from pathlib import Path
import json
from config.settings import EPOSConfig
from config.equipment_profiles import EquipmentProfile
from utils.logger import log_info, log_error, log_warning, log_debug
from utils.helpers import calculate_statistics, format_currency, format_power, save_to_json, save_to_csv
from utils import constants


class OptimizationStatus(Enum):
    """Статусы оптимизации"""
    OPTIMAL = "Оптимальное решение найдено"
    INFEASIBLE = "Задача неразрешима"
    UNBOUNDED = "Задача неограничена"
    NOT_SOLVED = "Не решена"
    ERROR = "Ошибка"


class ObjectiveType(Enum):
    """Типы целевых функций"""
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_PEAK = "minimize_peak"
    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_PROFIT = "maximize_profit"
    BALANCED = "balanced"


@dataclass
class OptimizationResult:
    """Результат оптимизации"""
    success: bool
    status: OptimizationStatus
    objective_value: float
    solution: Dict[str, List[float]]
    solve_time: float
    iterations: int
    constraints_count: int
    variables_count: int
    equipment_name: str
    optimization_horizon: int
    savings: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Конвертация в словарь"""
        return {
            'success': self.success,
            'status': self.status.value,
            'objective_value': self.objective_value,
            'solve_time': self.solve_time,
            'iterations': self.iterations,
            'constraints_count': self.constraints_count,
            'variables_count': self.variables_count,
            'equipment_name': self.equipment_name,
            'optimization_horizon': self.optimization_horizon,
            'savings': self.savings,
            'metadata': self.metadata or {},
            'solution_summary': {
                'total_energy_kwh': sum(sum(values) for values in self.solution.values()),
                'average_power_kw': np.mean(list(self.solution.values())[0]) if self.solution else 0
            }
        }

    def print_summary(self):
        """Печать сводки результатов"""
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ ЭПОС")
        print("=" * 60)
        print(f"Оборудование: {self.equipment_name}")
        print(f"Статус: {self.status.value}")
        print(f"Время решения: {self.solve_time:.2f} сек")
        print(f"Итерации: {self.iterations}")
        print(f"Целевая функция: {self.objective_value:.2f}")

        if self.savings:
            print(f"\nЭкономический эффект:")
            print(f"  Экономия: {format_currency(self.savings.get('absolute', 0))}")
            print(f"  Процент экономии: {self.savings.get('percent', 0):.1f}%")

        if self.solution and 'power' in self.solution:
            powers = self.solution['power']
            print(f"\nРешение (первые 12 часов):")
            for i in range(min(12, len(powers))):
                print(f"  Час {i:2d}: {format_power(powers[i])}")


class EPOSOptimizer:
    """Основной оптимизатор системы ЭПОС"""

    def __init__(self, solver: str = None, verbose: bool = True):
        """
        Инициализация оптимизатора

        Args:
            solver: Решатель (PULP, CBC, GLPK, Gurobi, CPLEX)
            verbose: Подробный вывод
        """
        self.solver = solver or EPOSConfig.SOLVER
        self.verbose = verbose
        self.problem = None
        self.result = None

        # Доступные решатели
        self.available_solvers = self._detect_available_solvers()

        log_info(f"Инициализация оптимизатора ЭПОС")
        log_info(f"Решатель: {self.solver}")
        log_info(f"Доступные решатели: {list(self.available_solvers.keys())}")
        log_info(f"Разработано командой {EPOSConfig.DEVELOPER}")

    def _detect_available_solvers(self) -> Dict[str, bool]:
        """Определение доступных решателей"""
        solvers = {}

        # Проверка PuLP
        try:
            pulp.getSolver('PULP_CBC_CMD')
            solvers['PULP'] = True
            log_debug("Решатель PULP доступен")
        except:
            solvers['PULP'] = False
            log_warning("Решатель PULP недоступен")

        # Проверка CBC
        try:
            pulp.getSolver('CBC_CMD')
            solvers['CBC'] = True
            log_debug("Решатель CBC доступен")
        except:
            solvers['CBC'] = False

        # Проверка GLPK
        try:
            pulp.getSolver('GLPK_CMD')
            solvers['GLPK'] = True
            log_debug("Решатель GLPK доступен")
        except:
            solvers['GLPK'] = False

        return solvers

    def optimize_equipment_schedule(self,
                                    equipment_profile: EquipmentProfile,
                                    prices: List[float],
                                    initial_state: str = 'off',
                                    objective: str = 'minimize_cost',
                                    horizon: int = 24,
                                    production_target: Optional[float] = None,
                                    consider_startup: bool = True,
                                    consider_ramping: bool = True,
                                    allow_curtailment: bool = False) -> OptimizationResult:
        """
        Оптимизация графика работы оборудования

        Args:
            equipment_profile: Профиль оборудования
            prices: Цены на электроэнергию по часам
            initial_state: Начальное состояние ('on' или 'off')
            objective: Целевая функция
            horizon: Горизонт планирования (часов)
            production_target: Целевой объем производства
            consider_startup: Учитывать затраты на запуск
            consider_ramping: Учитывать ограничения на изменение нагрузки
            allow_curtailment: Разрешить снижение нагрузки ниже минимума

        Returns:
            Результат оптимизации
        """
        start_time = time.time()
        log_info(f"Начало оптимизации для {equipment_profile.name}")
        log_info(f"Горизонт: {horizon} часов, Цель: {objective}")

        try:
            # Создание задачи оптимизации
            self._create_optimization_problem(
                equipment_profile, prices, horizon, objective
            )

            # Добавление переменных
            variables = self._add_variables(equipment_profile, horizon)

            # Добавление ограничений
            constraints_count = self._add_constraints(
                variables, equipment_profile, horizon, initial_state,
                consider_startup, consider_ramping, allow_curtailment,
                production_target
            )

            # Добавление целевой функции
            self._add_objective(objective, variables, prices, equipment_profile)

            # Решение задачи
            solution_status = self._solve_problem()

            # Извлечение решения
            solution = self._extract_solution(variables, horizon)

            # Расчет времени решения
            solve_time = time.time() - start_time

            # Формирование результата
            self.result = OptimizationResult(
                success=solution_status == OptimizationStatus.OPTIMAL,
                status=solution_status,
                objective_value=self.problem.objective.value() if self.problem else 0,
                solution=solution,
                solve_time=solve_time,
                iterations=self._get_iterations_count(),
                constraints_count=constraints_count,
                variables_count=len(variables) * horizon,
                equipment_name=equipment_profile.name,
                optimization_horizon=horizon,
                metadata={
                    'objective': objective,
                    'initial_state': initial_state,
                    'production_target': production_target,
                    'consider_startup': consider_startup,
                    'consider_ramping': consider_ramping,
                    'prices_mean': np.mean(prices)
                }
            )

            log_info(f"Оптимизация завершена за {solve_time:.2f} сек")
            log_info(f"Статус: {solution_status.value}")

            if self.verbose:
                self.result.print_summary()

            return self.result

        except Exception as e:
            log_error(f"Ошибка оптимизации: {e}")

            return OptimizationResult(
                success=False,
                status=OptimizationStatus.ERROR,
                objective_value=0,
                solution={},
                solve_time=time.time() - start_time,
                iterations=0,
                constraints_count=0,
                variables_count=0,
                equipment_name=equipment_profile.name,
                optimization_horizon=horizon,
                metadata={'error': str(e)}
            )

    def _create_optimization_problem(self,
                                     equipment_profile: EquipmentProfile,
                                     prices: List[float],
                                     horizon: int,
                                     objective: str):
        """Создание задачи оптимизации"""
        problem_name = f"EPOS_{equipment_profile.name.replace(' ', '_')}_{horizon}h"

        # Определение типа задачи (минимизация или максимизация)
        if objective == 'maximize_profit':
            sense = pulp.LpMaximize
        else:
            sense = pulp.LpMinimize

        self.problem = pulp.LpProblem(problem_name, sense)
        log_debug(f"Создана задача оптимизации: {problem_name}")

    def _add_variables(self,
                       equipment_profile: EquipmentProfile,
                       horizon: int) -> Dict[str, List]:
        """Добавление переменных оптимизации"""
        variables = {}

        # Основные переменные
        # 1. Двоичная переменная: работает ли оборудование в час t
        variables['on'] = [
            pulp.LpVariable(f"on_{t}", cat='Binary')
            for t in range(horizon)
        ]

        # 2. Мощность оборудования в час t (непрерывная)
        variables['power'] = [
            pulp.LpVariable(
                f"power_{t}",
                lowBound=0,
                upBound=equipment_profile.power_max,
                cat='Continuous'
            )
            for t in range(horizon)
        ]

        # 3. Двоичные переменные для учета запуска и остановки
        variables['startup'] = [
            pulp.LpVariable(f"startup_{t}", cat='Binary')
            for t in range(horizon)
        ]

        variables['shutdown'] = [
            pulp.LpVariable(f"shutdown_{t}", cat='Binary')
            for t in range(horizon)
        ]

        # 4. Переменная для пиковой мощности
        variables['peak_power'] = pulp.LpVariable("peak_power", lowBound=0, cat='Continuous')

        log_debug(f"Добавлено {sum(len(v) for v in variables.values())} переменных")
        return variables

    def _add_constraints(self,
                         variables: Dict[str, List],
                         equipment_profile: EquipmentProfile,
                         horizon: int,
                         initial_state: str,
                         consider_startup: bool,
                         consider_ramping: bool,
                         allow_curtailment: bool,
                         production_target: Optional[float]) -> int:
        """Добавление ограничений оптимизации"""
        constraints_count = 0
        on = variables['on']
        power = variables['power']
        startup = variables['startup']
        shutdown = variables['shutdown']
        peak_power = variables['peak_power']

        # 1. Связь между состоянием и мощностью
        for t in range(horizon):
            # Если оборудование выключено, мощность = 0
            self.problem += power[t] <= equipment_profile.power_max * on[t]

            # Если оборудование включено, мощность >= минимальной
            if not allow_curtailment:
                self.problem += power[t] >= equipment_profile.power_min * on[t]

            constraints_count += 2

        # 2. Минимальное время работы
        if equipment_profile.min_on_time > 1:
            for t in range(horizon - equipment_profile.min_on_time + 1):
                # Если оборудование включилось в час t, оно должно работать min_on_time часов
                self.problem += pulp.lpSum(on[t:i] for i in range(t, t + equipment_profile.min_on_time)) \
                                >= equipment_profile.min_on_time * startup[t]
                constraints_count += 1

        # 3. Минимальное время простоя
        if equipment_profile.min_off_time > 1:
            for t in range(horizon - equipment_profile.min_off_time + 1):
                self.problem += pulp.lpSum([on[i] for i in range(t, t + equipment_profile.min_off_time)]) \
                                <= equipment_profile.min_off_time * (1 - shutdown[t])
                constraints_count += 1

        # 4. Ограничения на запуск и остановку
        for t in range(1, horizon):
            # Запуск: on[t] = 1 и on[t-1] = 0
            self.problem += startup[t] >= on[t] - on[t - 1]
            self.problem += startup[t] <= on[t]
            self.problem += startup[t] <= 1 - on[t - 1]

            # Остановка: on[t] = 0 и on[t-1] = 1
            self.problem += shutdown[t] >= on[t - 1] - on[t]
            self.problem += shutdown[t] <= on[t - 1]
            self.problem += shutdown[t] <= 1 - on[t]

            constraints_count += 6

        # 5. Начальное состояние
        if initial_state == 'on':
            self.problem += on[0] == 1
            self.problem += startup[0] == 0
        else:
            self.problem += on[0] == 0
            self.problem += shutdown[0] == 0
        constraints_count += 2

        # 6. Ограничения на изменение нагрузки (ramping)
        if consider_ramping and equipment_profile.ramp_rate < 100:
            ramp_limit = equipment_profile.ramp_rate / 100 * equipment_profile.power_max

            for t in range(1, horizon):
                self.problem += power[t] - power[t - 1] <= ramp_limit
                self.problem += power[t - 1] - power[t] <= ramp_limit
                constraints_count += 2

        # 7. Целевое производство
        if production_target:
            total_energy = pulp.lpSum(power)  # кВт*ч
            total_production = total_energy * equipment_profile.efficiency

            self.problem += total_production >= production_target
            constraints_count += 1

        # 8. Пиковая мощность
        for t in range(horizon):
            self.problem += peak_power >= power[t]
            constraints_count += 1

        log_debug(f"Добавлено {constraints_count} ограничений")
        return constraints_count

    def _add_objective(self,
                       objective: str,
                       variables: Dict[str, List],
                       prices: List[float],
                       equipment_profile: EquipmentProfile):
        """Добавление целевой функции"""
        on = variables['on']
        power = variables['power']
        startup = variables['startup']
        peak_power = variables['peak_power']
        horizon = len(power)

        # Стоимость энергии
        energy_cost = pulp.lpSum([power[t] * prices[t] for t in range(horizon)])

        # Стоимость запуска (упрощенно)
        startup_cost = pulp.lpSum([startup[t] * equipment_profile.power_max * 0.1
                                   for t in range(horizon)])

        if objective == 'minimize_cost':
            # Минимизация общей стоимости
            self.problem += energy_cost + startup_cost

        elif objective == 'minimize_peak':
            # Минимизация пиковой мощности
            peak_cost = peak_power * np.mean(prices) * 10  # Штраф за пик
            self.problem += energy_cost + startup_cost + peak_cost

        elif objective == 'minimize_energy':
            # Минимизация потребления энергии
            total_energy = pulp.lpSum(power)
            self.problem += total_energy

        elif objective == 'maximize_profit':
            # Максимизация прибыли (если есть доход от производства)
            if hasattr(equipment_profile, 'production_rate') and equipment_profile.production_rate:
                revenue = pulp.lpSum([power[t] * equipment_profile.production_rate * 100
                                      for t in range(horizon)])  # Упрощенный расчет
                self.problem += revenue - energy_cost - startup_cost
            else:
                log_warning("Невозможно максимизировать прибыль: не указана производственная ставка")
                self.problem += -energy_cost - startup_cost  # Минимизация затрат

        elif objective == 'balanced':
            # Сбалансированная цель: стоимость + пик
            peak_cost = peak_power * np.mean(prices) * 5
            self.problem += energy_cost + startup_cost + peak_cost

        else:
            log_warning(f"Неизвестная целевая функция: {objective}. Используется minimize_cost")
            self.problem += energy_cost + startup_cost

        log_debug(f"Целевая функция: {objective}")

    def _solve_problem(self) -> OptimizationStatus:
        """Решение задачи оптимизации"""
        if not self.problem:
            return OptimizationStatus.NOT_SOLVED

        try:
            # Выбор решателя
            if self.solver == 'PULP' and 'PULP' in self.available_solvers:
                solver = pulp.getSolver('PULP_CBC_CMD', msg=False)
            elif self.solver == 'CBC' and 'CBC' in self.available_solvers:
                solver = pulp.getSolver('CBC_CMD', msg=False)
            elif self.solver == 'GLPK' and 'GLPK' in self.available_solvers:
                solver = pulp.getSolver('GLPK_CMD', msg=False)
            else:
                # Использовать решатель по умолчанию
                solver = None

            # Решение задачи
            self.problem.solve(solver)

            # Определение статуса
            status_map = {
                pulp.LpStatusOptimal: OptimizationStatus.OPTIMAL,
                pulp.LpStatusInfeasible: OptimizationStatus.INFEASIBLE,
                pulp.LpStatusUnbounded: OptimizationStatus.UNBOUNDED,
                pulp.LpStatusNotSolved: OptimizationStatus.NOT_SOLVED
            }

            pulp_status = pulp.LpStatus[self.problem.status]
            return status_map.get(pulp_status, OptimizationStatus.ERROR)

        except Exception as e:
            log_error(f"Ошибка решения задачи: {e}")
            return OptimizationStatus.ERROR

    def _extract_solution(self, variables: Dict[str, List], horizon: int) -> Dict[str, List[float]]:
        """Извлечение решения из переменных"""
        solution = {}

        for var_name, var_list in variables.items():
            if var_name == 'peak_power':
                # Одиночная переменная
                solution[var_name] = [pulp.value(var_list)]
            else:
                # Список переменных
                solution[var_name] = [pulp.value(var) for var in var_list]

        # Дополнительные расчеты
        if 'power' in solution and 'on' in solution:
            power = solution['power']
            on = solution['on']

            # Расчет нагрузки в процентах
            solution['load_factor'] = []
            for p, o in zip(power, on):
                if o > 0.5:  # Оборудование включено
                    load_factor = p / max(p, 1)  # Избегаем деления на 0
                    solution['load_factor'].append(min(1.0, load_factor))
                else:
                    solution['load_factor'].append(0.0)

            # Расчет стоимости по часам
            if 'price' in solution:
                solution['hourly_cost'] = [
                    p * solution['price'][i] for i, p in enumerate(power)
                ]

        return solution

    def _get_iterations_count(self) -> int:
        """Получение количества итераций (упрощенная версия)"""
        # В PuLP нет прямого доступа к количеству итераций
        # Можно добавить логирование из решателя в будущем
        return 0

    def calculate_savings(self,
                          baseline_schedule: List[float],
                          optimized_schedule: List[float],
                          prices: List[float]) -> Dict[str, float]:
        """
        Расчет экономии от оптимизации

        Args:
            baseline_schedule: Базовый график нагрузки
            optimized_schedule: Оптимизированный график
            prices: Цены на электроэнергию

        Returns:
            Словарь с расчетом экономии
        """
        baseline_cost = sum(p * l for p, l in zip(prices, baseline_schedule))
        optimized_cost = sum(p * l for p, l in zip(prices, optimized_schedule))

        absolute_saving = baseline_cost - optimized_cost
        percent_saving = (absolute_saving / baseline_cost * 100) if baseline_cost > 0 else 0

        # Расчет снижения пиковой нагрузки
        baseline_peak = max(baseline_schedule) if baseline_schedule else 0
        optimized_peak = max(optimized_schedule) if optimized_schedule else 0
        peak_reduction = baseline_peak - optimized_peak
        peak_reduction_percent = (peak_reduction / baseline_peak * 100) if baseline_peak > 0 else 0

        return {
            'baseline_cost': round(baseline_cost, 2),
            'optimized_cost': round(optimized_cost, 2),
            'absolute': round(absolute_saving, 2),
            'percent': round(percent_saving, 2),
            'baseline_peak': round(baseline_peak, 2),
            'optimized_peak': round(optimized_peak, 2),
            'peak_reduction': round(peak_reduction, 2),
            'peak_reduction_percent': round(peak_reduction_percent, 2)
        }

    def save_optimization_report(self,
                                 result: OptimizationResult,
                                 prices: List[float],
                                 baseline_schedule: Optional[List[float]] = None,
                                 filename: str = None) -> Dict[str, Path]:
        """
        Сохранение отчета об оптимизации

        Args:
            result: Результат оптимизации
            prices: Цены на электроэнергию
            baseline_schedule: Базовый график нагрузки
            filename: Имя файла

        Returns:
            Словарь с путями к сохраненным файлам
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_{result.equipment_name}_{timestamp}"

        # Подготовка данных для сохранения
        report_data = result.to_dict()

        # Добавление детального решения по часам
        if 'power' in result.solution:
            detailed_solution = []
            for hour in range(result.optimization_horizon):
                hour_data = {
                    'hour': hour,
                    'on_status': result.solution.get('on', [])[hour] if 'on' in result.solution else 0,
                    'power_kw': result.solution.get('power', [])[hour] if 'power' in result.solution else 0,
                    'price_rub_kwh': prices[hour] if hour < len(prices) else 0,
                    'hourly_cost_rub': result.solution.get('power', [])[hour] * prices[hour]
                    if hour < len(prices) and 'power' in result.solution else 0,
                    'load_factor': result.solution.get('load_factor', [])[hour]
                    if 'load_factor' in result.solution else 0
                }
                detailed_solution.append(hour_data)

            report_data['detailed_solution'] = detailed_solution

        # Расчет экономии если есть базовый график
        if baseline_schedule and 'power' in result.solution:
            savings = self.calculate_savings(baseline_schedule, result.solution['power'], prices)
            report_data['savings'] = savings

            # Обновление результата
            result.savings = savings

        # Сохранение в JSON
        json_path = save_to_json(report_data, f"{filename}.json")

        # Сохранение в CSV
        if 'detailed_solution' in report_data:
            csv_data = []
            for hour_data in report_data['detailed_solution']:
                csv_data.append({
                    'hour': hour_data['hour'],
                    'on_status': hour_data['on_status'],
                    'power_kw': hour_data['power_kw'],
                    'price_rub_kwh': hour_data['price_rub_kwh'],
                    'hourly_cost_rub': hour_data['hourly_cost_rub'],
                    'load_factor': hour_data['load_factor']
                })

            csv_path = save_to_csv({'optimization_schedule': csv_data}, f"{filename}.csv")
        else:
            csv_path = None

        # Сохранение сводки в текстовый файл
        summary_path = EPOSConfig.REPORTS_DIR / "pdf" / f"{filename}_summary.txt"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"ОТЧЕТ ОБ ОПТИМИЗАЦИИ ЭПОС\n")
            f.write("=" * 60 + "\n")
            f.write(f"Оборудование: {result.equipment_name}\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Статус: {result.status.value}\n")
            f.write(f"Время решения: {result.solve_time:.2f} сек\n")
            f.write(f"Целевая функция: {result.objective_value:.2f}\n")

            if result.savings:
                f.write(f"\nЭКОНОМИЧЕСКИЙ ЭФФЕКТ:\n")
                f.write(f"  Стоимость до оптимизации: {format_currency(result.savings['baseline_cost'])}\n")
                f.write(f"  Стоимость после оптимизации: {format_currency(result.savings['optimized_cost'])}\n")
                f.write(f"  Экономия: {format_currency(result.savings['absolute'])}\n")
                f.write(f"  Процент экономии: {result.savings['percent']:.1f}%\n")
                f.write(f"  Снижение пиковой нагрузки: {result.savings['peak_reduction_percent']:.1f}%\n")

            f.write(f"\nРАЗРАБОТАНО СИСТЕМОЙ ЭПОС\n")
            f.write(f"TechLitCode © {datetime.now().year}\n")

        log_info(f"Отчет об оптимизации сохранен:")
        log_info(f"  JSON: {json_path}")
        log_info(f"  CSV: {csv_path}")
        log_info(f"  Сводка: {summary_path}")

        return {
            'json': json_path,
            'csv': csv_path,
            'summary': summary_path
        }


# Утилитарные функции для быстрого доступа
def optimize_quick(equipment_type: str = "compressor",
                   hours: int = 24,
                   objective: str = "minimize_cost") -> OptimizationResult:
    """Быстрая оптимизация с генерацией данных"""
    from config.equipment_profiles import EquipmentManager
    from data.generator import DataGenerator

    # Получение профиля оборудования
    profile = EquipmentManager.get_profile(f"{equipment_type}_air")

    # Генерация данных
    generator = DataGenerator()
    prices = generator.generate_price_profile(hours=hours)

    # Оптимизация
    optimizer = EPOSOptimizer()
    result = optimizer.optimize_equipment_schedule(
        equipment_profile=profile,
        prices=prices,
        objective=objective,
        horizon=hours
    )

    return result


def compare_optimization_strategies(equipment_profile: EquipmentProfile,
                                    prices: List[float],
                                    strategies: List[str] = None) -> Dict[str, OptimizationResult]:
    """Сравнение различных стратегий оптимизации"""
    if strategies is None:
        strategies = ['minimize_cost', 'minimize_peak', 'balanced']

    results = {}

    for strategy in strategies:
        log_info(f"Тестирование стратегии: {strategy}")

        optimizer = EPOSOptimizer(verbose=False)
        result = optimizer.optimize_equipment_schedule(
            equipment_profile=equipment_profile,
            prices=prices,
            objective=strategy,
            horizon=len(prices)
        )

        results[strategy] = result

    # Анализ сравнения
    comparison = {
        'best_cost': None,
        'best_peak': None,
        'recommendation': None
    }

    # Поиск лучшей стратегии по стоимости
    valid_results = {k: v for k, v in results.items() if v.success}
    if valid_results:
        best_cost_strategy = min(valid_results.items(),
                                 key=lambda x: x[1].objective_value)
        comparison['best_cost'] = {
            'strategy': best_cost_strategy[0],
            'cost': best_cost_strategy[1].objective_value
        }

    return results, comparison


def validate_solution(solution: Dict[str, List[float]],
                      equipment_profile: EquipmentProfile) -> List[str]:
    """Валидация решения оптимизации"""
    errors = []

    if 'power' not in solution:
        errors.append("Отсутствуют данные о мощности")
        return errors

    powers = solution['power']
    horizon = len(powers)

    # Проверка минимального времени работы
    if equipment_profile.min_on_time > 1:
        for i in range(horizon - equipment_profile.min_on_time + 1):
            # Проверяем, что если оборудование включилось, оно работает min_on_time часов
            if i > 0 and 'on' in solution:
                if solution['on'][i] == 1 and solution['on'][i - 1] == 0:
                    # Запуск в час i, проверяем следующие min_on_time часов
                    should_be_on = all(solution['on'][j] == 1
                                       for j in range(i, min(i + equipment_profile.min_on_time, horizon)))
                    if not should_be_on:
                        errors.append(f"Нарушение минимального времени работы в час {i}")

    # Проверка ограничений мощности
    for i, power in enumerate(powers):
        if power < 0:
            errors.append(f"Отрицательная мощность в час {i}")
        elif power > equipment_profile.power_max:
            errors.append(f"Превышение максимальной мощности в час {i}: {power} > {equipment_profile.power_max}")
        elif 'on' in solution and solution['on'][i] == 0 and power > 0.1:
            errors.append(f"Мощность > 0 при выключенном оборудовании в час {i}")

    return errors