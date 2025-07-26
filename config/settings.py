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
