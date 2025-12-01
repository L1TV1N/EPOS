#!/usr/bin/env python3
"""
Главный модуль системы ЭПОС (Энерго-Производственный Оптимизационный Синтез)
Точка входа командной строки
Разработано командой TechLitCode
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Добавление пути к модулям проекта
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import EPOSConfig
from config.equipment_profiles import EquipmentProfile, EquipmentManager
from data.generator import DataGenerator, generate_quick_dataset
from data.price_loader import PriceLoader, get_current_prices
from data.iot_simulator import create_demo_simulator, simulate_real_time
from core.optimizer import EPOSOptimizer, optimize_quick, compare_optimization_strategies
from core.forecaster import EPOSForecaster, quick_load_forecast, compare_forecast_models
from core.validator import EPOSValidator, quick_validate_equipment, validate_data_file
from utils.logger import log_info, log_error, log_warning, EPOSLogger
from utils.helpers import print_table, format_currency, format_power, save_to_json, save_to_csv


class EPOSCLI:
    """Командный интерфейс системы ЭПОС"""

    def __init__(self):
        """Инициализация CLI"""
        self.logger = EPOSLogger().get_logger()
        self.setup_argparse()

    def setup_argparse(self):
        """Настройка парсера аргументов командной строки"""
        self.parser = argparse.ArgumentParser(
            description=f"{EPOSConfig.SYSTEM_NAME} - Система оптимизации энергопотребления",
            epilog=f"Разработано командой {EPOSConfig.DEVELOPER}",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        # Основные команды
        subparsers = self.parser.add_subparsers(dest='command', help='Доступные команды')

        # Команда: информация о системе
        info_parser = subparsers.add_parser('info', help='Информация о системе')

        # Команда: генерация данных
        generate_parser = subparsers.add_parser('generate', help='Генерация данных')
        generate_parser.add_argument('--type', type=str, default='compressor',
                                     choices=['compressor', 'pump', 'oven', 'ventilation', 'conveyor'],
                                     help='Тип оборудования')
        generate_parser.add_argument('--hours', type=int, default=24,
                                     help='Количество часов данных')
        generate_parser.add_argument('--output', type=str,
                                     help='Путь для сохранения данных')

        # Команда: загрузка цен
        prices_parser = subparsers.add_parser('prices', help='Работа с ценами')
        prices_parser.add_argument('--market', type=str, default='ATS',
                                   choices=['ATS', 'SPB', 'SIMULATED'],
                                   help='Рынок электроэнергии')
        prices_parser.add_argument('--hours', type=int, default=24,
                                   help='Количество часов')
        prices_parser.add_argument('--forecast', action='store_true',
                                   help='Получить прогноз цен')

        # Команда: оптимизация
        optimize_parser = subparsers.add_parser('optimize', help='Оптимизация графика')
        optimize_parser.add_argument('--equipment', type=str, default='compressor',
                                     help='Тип оборудования')
        optimize_parser.add_argument('--hours', type=int, default=24,
                                     help='Горизонт оптимизации')
        optimize_parser.add_argument('--objective', type=str, default='minimize_cost',
                                     choices=['minimize_cost', 'minimize_peak', 'maximize_profit', 'balanced'],
                                     help='Целевая функция')
        optimize_parser.add_argument('--output', type=str,
                                     help='Путь для сохранения отчета')

        # Команда: прогнозирование
        forecast_parser = subparsers.add_parser('forecast', help='Прогнозирование')
        forecast_parser.add_argument('--type', type=str, default='load',
                                     choices=['load', 'prices'],
                                     help='Тип прогноза')
        forecast_parser.add_argument('--equipment', type=str, default='compressor',
                                     help='Тип оборудования (для прогноза нагрузки)')
        forecast_parser.add_argument('--hours', type=int, default=24,
                                     help='Горизонт прогнозирования')

        # Команда: валидация
        validate_parser = subparsers.add_parser('validate', help='Валидация данных')
        validate_parser.add_argument('--type', type=str, default='equipment',
                                     choices=['equipment', 'data', 'solution'],
                                     help='Тип валидации')
        validate_parser.add_argument('--input', type=str,
                                     help='Путь к файлу для валидации')
        validate_parser.add_argument('--equipment', type=str, default='compressor',
                                     help='Тип оборудования')

        # Команда: симуляция IoT
        iot_parser = subparsers.add_parser('iot', help='Симуляция IoT-датчиков')
        iot_parser.add_argument('--devices', type=int, default=3,
                                help='Количество устройств')
        iot_parser.add_argument('--duration', type=int, default=30,
                                help='Длительность симуляции (секунд)')
        iot_parser.add_argument('--output', type=str,
                                help='Путь для сохранения данных')

        # Команда: сценарий
        scenario_parser = subparsers.add_parser('scenario', help='Запуск сценария')
        scenario_parser.add_argument('number', type=int, choices=[1, 2, 3, 4, 5],
                                     help='Номер сценария (1-5)')
        scenario_parser.add_argument('--output', type=str,
                                     help='Путь для сохранения результатов')

        # Команда: отчет
        report_parser = subparsers.add_parser('report', help='Генерация отчета')
        report_parser.add_argument('--type', type=str, default='summary',
                                   choices=['summary', 'comparison', 'detailed'],
                                   help='Тип отчета')

    def print_banner(self):
        """Печать баннера системы"""
        banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                    ЭПОС v{EPOSConfig.VERSION}                           ║
║      Энерго-Производственный Оптимизационный Синтез          ║
║                                                              ║
║          Динамический цифровой двойник энергосистемы         ║
║                    Разработано командой                      ║
║                        {EPOSConfig.DEVELOPER}                        ║
╚══════════════════════════════════════════════════════════════╝
        """
        print(banner)

    def run(self):
        """Запуск CLI"""
        if len(sys.argv) == 1:
            self.print_banner()
            self.parser.print_help()
            return

        args = self.parser.parse_args()

        # Логирование запуска команды
        self.logger.info(f"Запуск команды: {args.command}")
        self.logger.info(f"Аргументы: {vars(args)}")

        # Обработка команд
        if args.command == 'info':
            self.cmd_info()
        elif args.command == 'generate':
            self.cmd_generate(args)
        elif args.command == 'prices':
            self.cmd_prices(args)
        elif args.command == 'optimize':
            self.cmd_optimize(args)
        elif args.command == 'forecast':
            self.cmd_forecast(args)
        elif args.command == 'validate':
            self.cmd_validate(args)
        elif args.command == 'iot':
            self.cmd_iot(args)
        elif args.command == 'scenario':
            self.cmd_scenario(args)
        elif args.command == 'report':
            self.cmd_report(args)
        else:
            log_error(f"Неизвестная команда: {args.command}")
            self.parser.print_help()

    def cmd_info(self):
        """Команда: информация о системе"""
        self.print_banner()

        print("\n" + "=" * 60)
        print("ИНФОРМАЦИЯ О СИСТЕМЕ")
        print("=" * 60)

        print(f"\nСистема: {EPOSConfig.SYSTEM_NAME}")
        print(f"Версия: {EPOSConfig.VERSION}")
        print(f"Разработчик: {EPOSConfig.DEVELOPER}")
        print(f"Дата сборки: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\nКонфигурация:")
        print(f"  Горизонт оптимизации: {EPOSConfig.OPTIMIZATION_HORIZON} часов")
        print(f"  Решатель: {EPOSConfig.SOLVER}")
        print(f"  Использовать реальные цены: {EPOSConfig.USE_REAL_PRICES}")

        print(f"\nПрофили оборудования:")
        from config.equipment_profiles import EquipmentManager
        profiles = EquipmentManager.list_profiles()
        for name, profile in list(profiles.items())[:5]:  # Показываем первые 5
            print(f"  • {name}: {profile['name']} ({profile['power_nominal']} кВт)")

        if len(profiles) > 5:
            print(f"  ... и еще {len(profiles) - 5} профилей")

        print(f"\nДиректории:")
        print(f"  Корневая: {EPOSConfig.BASE_DIR}")
        print(f"  Логи: {EPOSConfig.LOGS_DIR}")
        print(f"  Отчеты: {EPOSConfig.REPORTS_DIR}")
        print(f"  Графики: {EPOSConfig.PLOTS_DIR}")

        print(f"\nИспользование:")
        print("  python main.py [команда] --help  # Справка по команде")
        print("  python main.py generate --type compressor --hours 24")
        print("  python main.py optimize --equipment pump --objective minimize_cost")
        print("  python main.py scenario 1  # Запуск демонстрационного сценария")

    def cmd_generate(self, args):
        """Команда: генерация данных"""
        print(f"\nГенерация данных для оборудования: {args.type}")
        print(f"Горизонт: {args.hours} часов")

        generator = DataGenerator(seed=42)

        # Генерация полного набора данных
        dataset = generator.generate_complete_dataset(
            equipment_type=args.type,
            hours=args.hours,
            save_to_file=True
        )

        # Сохранение в указанный файл
        if args.output:
            output_path = Path(args.output)
            if output_path.suffix == '.json':
                save_to_json(dataset, output_path.stem)
            elif output_path.suffix == '.csv':
                # Конвертация в DataFrame и сохранение
                import pandas as pd
                df_data = {
                    'hour': list(range(args.hours)),
                    'load_kw': dataset['load_profile_kw'],
                    'price_rub_kwh': dataset['price_profile_rub_kwh'],
                    'temperature_c': dataset['weather']['temperature_c']
                }
                df = pd.DataFrame(df_data)
                df.to_csv(output_path, index=False, encoding='utf-8')
                print(f"Данные сохранены в CSV: {output_path}")

        # Печать статистики
        load_stats = dataset['statistics']['load']
        price_stats = dataset['statistics']['price']

        print("\nСтатистика сгенерированных данных:")
        print(f"Нагрузка: среднее={load_stats['mean']:.1f} кВт, "
              f"макс={load_stats['max']:.1f} кВт, мин={load_stats['min']:.1f} кВт")
        print(f"Цены: среднее={price_stats['mean']:.2f} руб/кВт*ч, "
              f"макс={price_stats['max']:.2f} руб, мин={price_stats['min']:.2f} руб")

        if 'saved_to' in dataset:
            print(f"\nДанные сохранены в: {dataset['saved_to']}")

    def cmd_prices(self, args):
        """Команда: работа с ценами"""
        print(f"\nРабота с ценами на рынке: {args.market}")

        loader = PriceLoader(market=args.market)

        if args.forecast:
            # Прогноз цен
            print(f"Получение прогноза цен на {args.hours} часов...")
            forecast = loader.get_price_forecast(hours_ahead=args.hours)

            if forecast:
                print(f"\nПрогноз цен на {args.hours} часов:")
                for i, item in enumerate(forecast['forecast'][:12]):  # Показываем первые 12 часов
                    print(f"  Час {item['hour']:2d}: {item['forecast_price_rub_kwh']:.2f} руб/кВт*ч "
                          f"({item['confidence_interval']['lower']:.2f}-{item['confidence_interval']['upper']:.2f})")

                stats = forecast['statistics']
                print(f"\nСтатистика прогноза:")
                print(f"  Средняя цена: {stats['average_forecast']:.2f} руб/кВт*ч")
                print(f"  Волатильность: {stats['volatility']:.1f}%")

                # Сохранение прогноза
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"price_forecast_{args.market}_{timestamp}"
                save_to_json(forecast, filename)
                print(f"\nПрогноз сохранен в: {EPOSConfig.REPORTS_DIR}/json/{filename}.json")
        else:
            # Текущие цены
            print(f"Загрузка текущих цен на {args.hours} часов...")
            price_data = loader.load_prices(hours=args.hours)

            print(f"\nЦены на рынке {price_data.market} ({price_data.zone}):")
            for i, (ts, price) in enumerate(zip(price_data.timestamps, price_data.prices)):
                if i < 12:  # Показываем первые 12 часов
                    print(f"  {ts.strftime('%H:%M')}: {price:.2f} руб/кВт*ч")

            # Анализ паттернов
            analysis = loader.analyze_price_patterns(price_data)

            print(f"\nАнализ ценовых паттернов:")
            print(f"  Средняя цена: {analysis['price_statistics']['mean']:.2f} руб/кВт*ч")
            print(f"  Пиковые часы: {len(analysis['peak_analysis']['peak_hours'])} из {args.hours}")
            print(f"  Самые дешевые часы: {analysis['recommendations']['cheapest_hours']}")

            # Сохранение данных
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"prices_{args.market}_{timestamp}"

            # Сохранение в JSON
            data_dict = price_data.to_dict()
            save_to_json(data_dict, filename)

            # Сохранение анализа
            save_to_json(analysis, f"{filename}_analysis")

            print(f"\nДанные сохранены в: {EPOSConfig.REPORTS_DIR}/json/")

    def cmd_optimize(self, args):
        """Команда: оптимизация графика"""
        print(f"\nОптимизация графика для оборудования: {args.equipment}")
        print(f"Горизонт: {args.hours} часов")
        print(f"Целевая функция: {args.objective}")

        # Получение профиля оборудования
        try:
            from config.equipment_profiles import EquipmentManager
            profile = EquipmentManager.get_profile(f"{args.equipment}_air")
        except ValueError:
            print(f"Профиль оборудования {args.equipment} не найден. Используется компрессор.")
            profile = EquipmentManager.get_profile("compressor_air")

        # Генерация данных
        from data.generator import DataGenerator
        generator = DataGenerator()
        prices = generator.generate_price_profile(hours=args.hours)

        # Оптимизация
        optimizer = EPOSOptimizer(verbose=True)
        result = optimizer.optimize_equipment_schedule(
            equipment_profile=profile,
            prices=prices,
            objective=args.objective,
            horizon=args.hours
        )

        # Печать результатов
        result.print_summary()

        # Сохранение отчета
        if args.output or result.success:
            if args.output:
                filename = Path(args.output).stem
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"optimization_{args.equipment}_{args.objective}_{timestamp}"

            # Генерация базового графика для сравнения
            baseline = generator.generate_load_profile(
                equipment_type=args.equipment,
                hours=args.hours,
                pattern="industrial"
            )

            # Сохранение отчета
            report_paths = optimizer.save_optimization_report(
                result=result,
                prices=prices,
                baseline_schedule=baseline,
                filename=filename
            )

            print(f"\nОтчеты сохранены:")
            for report_type, path in report_paths.items():
                if path:
                    print(f"  {report_type}: {path}")

    def cmd_forecast(self, args):
        """Команда: прогнозирование"""
        print(f"\nПрогнозирование: {args.type}")
        print(f"Горизонт: {args.hours} часов")

        forecaster = EPOSForecaster(model_type='random_forest')

        if args.type == 'load':
            # Прогноз нагрузки оборудования
            print(f"Оборудование: {args.equipment}")
            result = forecaster.forecast_load(
                equipment_type=args.equipment,
                hours=args.hours,
                include_weather=True
            )
        else:
            # Прогноз цен
            result = forecaster.forecast_prices(
                market='ATS',
                hours=args.hours
            )

        # Печать результатов
        result.print_summary()

        # Сохранение отчета
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"forecast_{args.type}_{args.equipment if args.type == 'load' else 'price'}_{timestamp}"

        report_paths = forecaster.save_forecast_report(result, filename)

        print(f"\nОтчеты сохранены:")
        for report_type, path in report_paths.items():
            if path:
                print(f"  {report_type}: {path}")

    def cmd_validate(self, args):
        """Команда: валидация данных"""
        print(f"\nВалидация: {args.type}")

        validator = EPOSValidator(level='full')

        if args.type == 'equipment':
            # Валидация профиля оборудования
            from config.equipment_profiles import EquipmentManager
            profile = EquipmentManager.get_profile(f"{args.equipment}_air")
            result = validator.validate_equipment_profile(profile)

        elif args.type == 'data' and args.input:
            # Валидация данных из файла
            filepath = Path(args.input)
            result = validate_data_file(filepath)

        elif args.type == 'solution':
            # Валидация решения оптимизации
            print("Для валидации решения требуется файл решения.")
            print("Используйте: python main.py validate --type data --input <файл>")
            return

        else:
            print("Не указан входной файл для валидации данных.")
            print("Используйте: python main.py validate --type data --input <файл>")
            return

        # Печать результатов
        result.print_summary()

        # Сохранение отчета
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"validation_{args.type}_{timestamp}"

        report_path = validator.save_validation_report(result, filename)
        print(f"\nОтчет о валидации сохранен: {report_path}")

    def cmd_iot(self, args):
        """Команда: симуляция IoT-датчиков"""
        print(f"\nСимуляция IoT-датчиков")
        print(f"Количество устройств: {args.devices}")
        print(f"Длительность: {args.duration} секунд")

        # Создание симулятора
        simulator = create_demo_simulator(num_devices=args.devices)

        # Запуск симуляции
        print("\nЗапуск симуляции... (нажмите Ctrl+C для остановки)")
        print("Чтение данных с датчиков:")

        def print_callback(reading):
            """Callback для печати данных датчиков"""
            print(f"[{reading.timestamp.strftime('%H:%M:%S')}] "
                  f"{reading.equipment_id}.{reading.sensor_id}: "
                  f"{reading.value:.1f} {reading.unit}")

        try:
            simulator.start_simulation(interval=1.0)
            simulator.stream_data(print_callback, interval=2.0)

            # Ожидание указанной длительности
            import time
            time.sleep(args.duration)

        except KeyboardInterrupt:
            print("\nСимуляция остановлена пользователем")
        finally:
            simulator.stop_simulation()

        # Сохранение данных
        if args.output or args.devices > 0:
            print(f"\nСохранение данных устройств...")

            for device_id in list(simulator.devices.keys())[:3]:  # Сохраняем первые 3 устройства
                try:
                    filepath = simulator.save_device_data(
                        device_id=device_id,
                        time_window=timedelta(minutes=5)
                    )
                    print(f"  Данные сохранены: {filepath}")
                except Exception as e:
                    print(f"  Ошибка сохранения данных {device_id}: {e}")

    def cmd_scenario(self, args):
        """Команда: запуск сценария"""
        print(f"\nЗапуск сценария {args.number}")

        # Импорт сценария
        try:
            if args.number == 1:
                from scenarios.scenario_1 import run_scenario_1
                scenario_func = run_scenario_1
                scenario_name = "Компрессор с пиковыми ценами"
            elif args.number == 2:
                from scenarios.scenario_2 import run_scenario_2
                scenario_func = run_scenario_2
                scenario_name = "Насос с солнечной генерацией"
            elif args.number == 3:
                from scenarios.scenario_3 import run_scenario_3
                scenario_func = run_scenario_3
                scenario_name = "Промышленная печь"
            elif args.number == 4:
                from scenarios.scenario_4 import run_scenario_4
                scenario_func = run_scenario_4
                scenario_name = "Система вентиляции"
            elif args.number == 5:
                from scenarios.scenario_5 import run_scenario_5
                scenario_func = run_scenario_5
                scenario_name = "Полный цех с накопителями"
            else:
                print(f"Сценарий {args.number} не найден")
                return
        except ImportError as e:
            print(f"Ошибка загрузки сценария: {e}")
            print("Убедитесь, что файлы сценариев существуют в директории scenarios/")
            return

        print(f"Название: {scenario_name}")

        # Запуск сценария
        try:
            output_dir = Path(args.output) if args.output else EPOSConfig.OUTPUTS_DIR / "scenarios"
            results = scenario_func(output_dir=output_dir)

            print(f"\nСценарий завершен успешно!")
            print(f"Результаты сохранены в: {output_dir}")

            # Краткий отчет
            if 'summary' in results:
                print(f"\nКраткий отчет:")
                for key, value in results['summary'].items():
                    print(f"  {key}: {value}")

        except Exception as e:
            print(f"Ошибка выполнения сценария: {e}")
            import traceback
            traceback.print_exc()

    def cmd_report(self, args):
        """Команда: генерация отчета"""
        print(f"\nГенерация отчета: {args.type}")

        if args.type == 'summary':
            # Сводный отчет по системе
            self._generate_summary_report()
        elif args.type == 'comparison':
            # Сравнительный отчет
            self._generate_comparison_report()
        elif args.type == 'detailed':
            # Детальный отчет
            self._generate_detailed_report()

    def _generate_summary_report(self):
        """Генерация сводного отчета"""
        print("\n" + "=" * 60)
        print("СВОДНЫЙ ОТЧЕТ СИСТЕМЫ ЭПОС")
        print("=" * 60)

        # Информация о системе
        print(f"\nСистемная информация:")
        print(f"  Версия: {EPOSConfig.VERSION}")
        print(f"  Запусков: 1")  # Можно добавить счетчик запусков
        print(f"  Последний запуск: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Статистика файлов
        import os
        report_files = list(EPOSConfig.REPORTS_DIR.glob("**/*.json"))
        csv_files = list(EPOSConfig.REPORTS_DIR.glob("**/*.csv"))

        print(f"\nФайлы отчетов:")
        print(f"  JSON отчеты: {len(report_files)} файлов")
        print(f"  CSV данные: {len(csv_files)} файлов")
        print(f"  Логи: {len(list(EPOSConfig.LOGS_DIR.glob('*.log')))} файлов")

        # Профили оборудования
        from config.equipment_profiles import EquipmentManager
        profiles = EquipmentManager.list_profiles()

        print(f"\nПрофили оборудования:")
        print(f"  Всего профилей: {len(profiles)}")
        for eq_type in ['compressor', 'pump', 'oven', 'ventilation', 'conveyor']:
            count = len([p for p in profiles.keys() if eq_type in p])
            if count > 0:
                print(f"  {eq_type}: {count} профилей")

        # Сохранение отчета
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"system_summary_{timestamp}"

        summary = {
            'system': {
                'name': EPOSConfig.SYSTEM_NAME,
                'version': EPOSConfig.VERSION,
                'developer': EPOSConfig.DEVELOPER,
                'generated_at': datetime.now().isoformat()
            },
            'files': {
                'json_reports': len(report_files),
                'csv_data': len(csv_files),
                'log_files': len(list(EPOSConfig.LOGS_DIR.glob('*.log')))
            },
            'equipment_profiles': {
                'total': len(profiles),
                'by_type': {
                    eq_type: len([p for p in profiles.keys() if eq_type in p])
                    for eq_type in ['compressor', 'pump', 'oven', 'ventilation', 'conveyor']
                }
            }
        }

        report_path = save_to_json(summary, filename)
        print(f"\nОтчет сохранен: {report_path}")

    def _generate_comparison_report(self):
        """Генерация сравнительного отчета"""
        print("\nСравнительный отчет (демонстрационный)")

        # Сравнение стратегий оптимизации
        from config.equipment_profiles import EquipmentManager
        from data.generator import DataGenerator

        profile = EquipmentManager.get_profile("compressor_air")
        generator = DataGenerator()
        prices = generator.generate_price_profile(hours=24)

        print("\nСравнение стратегий оптимизации для компрессора:")
        print("=" * 60)

        strategies = ['minimize_cost', 'minimize_peak', 'balanced']
        results = []

        for strategy in strategies:
            optimizer = EPOSOptimizer(verbose=False)
            result = optimizer.optimize_equipment_schedule(
                equipment_profile=profile,
                prices=prices,
                objective=strategy,
                horizon=24
            )

            if result.success:
                total_cost = sum(result.solution['power'][i] * prices[i] for i in range(24))
                peak_power = max(result.solution['power'])
                runtime = result.solve_time

                results.append({
                    'strategy': strategy,
                    'cost': total_cost,
                    'peak': peak_power,
                    'time': runtime,
                    'status': 'success'
                })
            else:
                results.append({
                    'strategy': strategy,
                    'cost': 0,
                    'peak': 0,
                    'time': 0,
                    'status': 'failed'
                })

        # Печать таблицы сравнения
        headers = ['Стратегия', 'Стоимость, руб', 'Пик, кВт', 'Время, сек', 'Статус']
        rows = []

        for r in results:
            if r['status'] == 'success':
                rows.append([
                    r['strategy'],
                    f"{r['cost']:.2f}",
                    f"{r['peak']:.1f}",
                    f"{r['time']:.2f}",
                    "✓"
                ])
            else:
                rows.append([r['strategy'], '—', '—', '—', '✗'])

        print_table(headers, rows, "Результаты оптимизации")

        # Определение лучшей стратегии
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            best_cost = min(successful, key=lambda x: x['cost'])
            best_peak = min(successful, key=lambda x: x['peak'])

            print(f"\nЛучшая стратегия по стоимости: {best_cost['strategy']} ({best_cost['cost']:.2f} руб)")
            print(f"Лучшая стратегия по пиковой нагрузке: {best_peak['strategy']} ({best_peak['peak']:.1f} кВт)")

    def _generate_detailed_report(self):
        """Генерация детального отчета"""
        print("\nДетальный отчет системы")

        # Сбор информации о всех модулях
        modules_info = {
            'config': {
                'settings': 'Настройки системы',
                'equipment_profiles': 'Профили оборудования'
            },
            'core': {
                'optimizer': 'Оптимизационный движок',
                'forecaster': 'Система прогнозирования',
                'validator': 'Валидатор данных'
            },
            'data': {
                'generator': 'Генератор данных',
                'price_loader': 'Загрузчик цен',
                'iot_simulator': 'Симулятор IoT'
            },
            'utils': {
                'logger': 'Система логирования',
                'helpers': 'Вспомогательные функции',
                'constants': 'Константы системы'
            }
        }

        print("\nМодули системы:")
        for module, submodules in modules_info.items():
            print(f"\n  {module.upper()}:")
            for submodule, description in submodules.items():
                print(f"    • {submodule}: {description}")

        # Проверка доступности модулей
        print("\nПроверка модулей:")

        import importlib
        available = []
        missing = []

        for module in ['pulp', 'numpy', 'pandas', 'sklearn']:
            try:
                importlib.import_module(module)
                available.append(module)
            except ImportError:
                missing.append(module)

        print(f"  Доступны: {', '.join(available)}")
        if missing:
            print(f"  Отсутствуют: {', '.join(missing)}")
            print(f"  Установите: pip install {' '.join(missing)}")

        # Сохранение детального отчета
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"detailed_report_{timestamp}"

        report = {
            'system': {
                'name': EPOSConfig.SYSTEM_NAME,
                'version': EPOSConfig.VERSION,
                'config': {
                    'optimization_horizon': EPOSConfig.OPTIMIZATION_HORIZON,
                    'solver': EPOSConfig.SOLVER,
                    'use_real_prices': EPOSConfig.USE_REAL_PRICES
                }
            },
            'modules': modules_info,
            'dependencies': {
                'available': available,
                'missing': missing
            },
            'directories': {
                'base': str(EPOSConfig.BASE_DIR),
                'outputs': str(EPOSConfig.OUTPUTS_DIR),
                'logs': str(EPOSConfig.LOGS_DIR),
                'reports': str(EPOSConfig.REPORTS_DIR)
            },
            'generated_at': datetime.now().isoformat()
        }

        report_path = save_to_json(report, filename)
        print(f"\nДетальный отчет сохранен: {report_path}")


def main():
    """Главная функция"""
    try:
        cli = EPOSCLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n\nПрограмма остановлена пользователем")
        sys.exit(0)
    except Exception as e:
        log_error(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()