import pandas as pd
import numpy as np
import talib as ta
from typing import List, Dict, Any
import logging

logger = logging.getLogger('FeatureEngineer')

class FeatureEngineer:
    def __init__(self):
        self.technical_indicators = [
            'RSI', 'MACD', 'STOCH', 'BBANDS', 'ATR', 
            'ADX', 'OBV', 'CCI', 'EMA', 'WILLR'
        ]
        self.pattern_indicators = [
            'CDLENGULFING', 'CDLHAMMER', 'CDLSHOOTINGSTAR',
            'CDLMORNINGSTAR', 'CDLEVENINGSTAR'
        ]
        logger.info("Initialized FeatureEngineer")

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical features to dataframe"""
        # Price transformations
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['close_ema_10'] = df['close'].ewm(span=10, adjust=False).mean()
        df['close_ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        
        # Volatility features
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['natr'] = ta.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['volatility'] = df['close'].rolling(20).std()
        
        # Momentum features
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = ta.MACD(df['close'])
        df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
        df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        df['willr'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Volume features
        df['obv'] = ta.OBV(df['close'], df['volume'])
        df['volume_pct_change'] = df['volume'].pct_change()
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        
        # Cycle features
        df['ht_dcperiod'] = ta.HT_DCPERIOD(df['close'])
        df['ht_phasor'], _ = ta.HT_PHASOR(df['close'])
        
        # Pattern recognition
        df['CDLENGULFING'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['CDLHAMMER'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['CDLSHOOTINGSTAR'] = ta.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        df['CDLEVENINGSTAR'] = ta.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Statistical features
        df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = ta.BBANDS(
            df['close'], timeperiod=20)
        df['kurtosis'] = df['close'].rolling(50).kurt()
        
        # Drop initial NaN values
        df = df.dropna()
        logger.info(f"Added {len(df.columns)} features to dataset")
        return df

    def get_feature_list(self) -> List[str]:
        """Get list of all generated features"""
        return [
            'returns', 'log_returns', 'close_ema_10', 'close_ema_50', 'atr', 'natr', 'volatility',
            'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'cci', 'adx', 'willr',
            'obv', 'volume_pct_change', 'volume_sma_20', 'ht_dcperiod', 'ht_phasor',
            'CDLENGULFING', 'CDLHAMMER', 'CDLSHOOTINGSTAR', 'CDLMORNINGSTAR', 'CDLEVENINGSTAR',
            'z_score', 'bollinger_upper', 'bollinger_middle', 'bollinger_lower', 'kurtosis'
        ]
