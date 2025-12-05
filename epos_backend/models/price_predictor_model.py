"""
Модель прогнозирования цен на электроэнергию для системы ЭПОС
Разработано командой TechLitCode

Эта модель использует машинное обучение для прогнозирования цен на электроэнергию
на основе исторических данных, сезонных паттернов и рыночных факторов.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
import pickle
import warnings
import json
from pathlib import Path

warnings.filterwarnings('ignore')

from epos_backend.config.settings import EPOSConfig
from epos_backend.utils.logger import log_info, log_error, log_warning, log_debug
from epos_backend.utils.helpers import calculate_statistics, normalize_data, save_to_json, save_to_csv


