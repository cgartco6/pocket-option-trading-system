# Pocket Option Trading System Configuration

# Trading parameters
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'EURUSD', 'GBPJPY', 'AUDCAD', 'GBPUSD']
TIMEFRAMES = ['5m', '15m', '1h', '4h']
MODEL_PATHS = {
    'BTCUSDT': 'data/models/btc_model.h5',
    'ETHUSDT': 'data/models/eth_model.h5'
}

# Signal generation parameters
PO_SIGNAL_CONFIG = {
    'RSI_PERIOD': 14,
    'RSI_BUY_THRESHOLD': 30,
    'RSI_SELL_THRESHOLD': 70,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'BOLLINGER_PERIOD': 20,
    'BOLLINGER_BUY_THRESHOLD': 0.05,
    'BOLLINGER_SELL_THRESHOLD': 0.05,
    'STOCH_K_PERIOD': 14,
    'STOCH_D_PERIOD': 3,
    'VOLUME_SMA_PERIOD': 20,
    'VOLUME_BUY_MULTIPLIER': 1.5,
    'VOLUME_SELL_MULTIPLIER': 1.5,
    'MIN_BUY_CONDITIONS': 3,
    'MIN_SELL_CONDITIONS': 3
}

SIGNAL_VALIDATION_CONFIG = {
    'MIN_PRICE_CHANGE': 0.001,
    'MIN_VOLATILITY_THRESHOLD': 0.005,
    'FAILED_SIGNAL_REVERSAL_THRESHOLD': 0.003,
    'MIN_VOLUME_RATIO': 0.8,
    'MIN_PROFITABLE_CHANGE': 0.003,
    'PURE_SIGNAL_CONFIDENCE_THRESHOLD': 0.85,
    'MAX_PRICE_CHANGE_FACTOR': 0.02,
    'VOLATILITY_CONFIDENCE_FACTOR': 0.015
}

# AI model parameters
PREDICTION_THRESHOLDS = {
    'HIGH_CONFIDENCE': 0.75,
    'MEDIUM_CONFIDENCE': 0.65
}

# Retraining configuration
RETRAIN_INTERVAL = {
    'daily': '03:00',
    'weekly': '04:00'
}

DATA_LOOKBACK = {
    'daily': 1000,
    'weekly': 5000,
    'performance': 2000
}

PERFORMANCE_THRESHOLDS = {
    'min_accuracy': 0.65,
    'min_precision': 0.60,
    'min_recall': 0.55
}

SEQUENCE_LENGTH = 60

# Data cleaning configuration
DATA_CLEANING_CONFIG = {
    'VOLUME_ZSCORE_THRESHOLD': 4.0,
    'WINSORIZE_LIMITS': [0.01, 0.01],
    'MIN_VALID_COMPLETENESS': 0.95,
    'TIME_TOLERANCE_PCT': 0.2
}

# API credentials (use environment variables in production)
API_KEYS = {
    'BINANCE_API_KEY': 'your_binance_api_key',
    'BINANCE_SECRET': 'your_binance_secret',
    'PO_API_KEY': 'your_pocket_option_api_key',
    'PO_API_SECRET': 'your_pocket_option_secret'
}

# Military-grade security
ENCRYPTION_SALT = 'military-grade-salt-value'
