SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'EURUSD', 'GBPJPY']
TIMEFRAMES = ['5m', '15m', '1h']
MODEL_PATHS = {
    'BTCUSDT': 'data/models/btc_model_v1.h5',
    'ETHUSDT': 'data/models/eth_model_v1.h5'
}

SIGNAL_THRESHOLDS = {
    'PO_SIGNAL': {
        'RSI_BUY': 30,
        'RSI_SELL': 70,
        'MACD_CONFIRMATION': 2
    },
    'PO_AI_SIGNAL': {
        'CONFIDENCE_THRESHOLD': 0.75,
        'VOLATILITY_FILTER': 0.8
    }
}

RETRAIN_SCHEDULE = {
    'daily': '03:00',
    'weekly': 'sunday 04:00'
}

DATA_CLEANING_CONFIG = {
    'VOLUME_ZSCORE_THRESHOLD': 4.0,       # Z-score threshold for volume anomalies
    'WINSORIZE_LIMITS': [0.01, 0.01],     # Trim top and bottom 1% of values
    'MIN_VALID_COMPLETENESS': 0.95,       # Minimum required data completeness
    'TIME_TOLERANCE_PCT': 0.2,            # 20% tolerance for time intervals
}

# PO Signal Configuration
PO_SIGNAL_CONFIG = {
    'RSI_PERIOD': 14,
    'RSI_BUY_THRESHOLD': 30,
    'RSI_SELL_THRESHOLD': 70,
    'MACD_FAST': 12,
    'MACD_SLOW': 26,
    'MACD_SIGNAL': 9,
    'BOLLINGER_PERIOD': 20,
    'BOLLINGER_BUY_THRESHOLD': 0.05,  # 5% from lower band
    'BOLLINGER_SELL_THRESHOLD': 0.05,  # 5% from upper band
    'STOCH_K_PERIOD': 14,
    'STOCH_D_PERIOD': 3,
    'VOLUME_SMA_PERIOD': 20,
    'VOLUME_BUY_MULTIPLIER': 1.5,
    'VOLUME_SELL_MULTIPLIER': 1.5,
    'MIN_BUY_CONDITIONS': 3,  # At least 3 conditions must be met
    'MIN_SELL_CONDITIONS': 3
}

# Signal Validation Configuration
SIGNAL_VALIDATION_CONFIG = {
    'MIN_PRICE_CHANGE': 0.001,  # 0.1% minimum change to be valid
    'MIN_VOLATILITY_THRESHOLD': 0.005,  # 0.5% minimum candle range
    'FAILED_SIGNAL_REVERSAL_THRESHOLD': 0.003,  # 0.3% reversal to classify as noise
    'MIN_VOLUME_RATIO': 0.8,  # Volume must be at least 80% of SMA
    'MIN_PROFITABLE_CHANGE': 0.003,  # 0.3% minimum profitable change
    'PURE_SIGNAL_CONFIDENCE_THRESHOLD': 0.85,  # 85% confidence for pure signals
    'MAX_PRICE_CHANGE_FACTOR': 0.02,  # 2% price change = +30% confidence
    'VOLATILITY_CONFIDENCE_FACTOR': 0.015,  # 1.5% volatility = +20% confidence
}
