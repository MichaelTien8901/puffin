"""Tree ensemble models for trading."""

from .random_forest import RandomForestTrader
from .long_short import EnsembleLongShort

# Optional imports with graceful fallback
try:
    from .xgboost_model import XGBoostTrader
except ImportError:
    XGBoostTrader = None

try:
    from .lightgbm_model import LightGBMTrader
except ImportError:
    LightGBMTrader = None

try:
    from .catboost_model import CatBoostTrader
except ImportError:
    CatBoostTrader = None

try:
    from .interpretation import ModelInterpreter
except ImportError:
    ModelInterpreter = None

__all__ = [
    "RandomForestTrader",
    "XGBoostTrader",
    "LightGBMTrader",
    "CatBoostTrader",
    "ModelInterpreter",
    "EnsembleLongShort",
]
