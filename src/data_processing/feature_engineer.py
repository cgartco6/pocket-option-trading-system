import talib as ta
import numpy as np
import pandas as pd

class FeatureEngineer:
    @staticmethod
    def add_features(df):
        # Price transformations
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['natr'] = ta.NATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['volatility'] = df['close'].rolling(20).std()
        
        # Momentum
        df['rsi'] = ta.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = ta.MACD(df['close'])
        df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
        df['cci'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=20)
        
        # Cycle
        df['ht_dcperiod'] = ta.HT_DCPERIOD(df['close'])
        df['ht_phasor'], _ = ta.HT_PHASOR(df['close'])
        
        # Pattern recognition
        df['CDLENGULFING'] = ta.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
        df['CDLHAMMER'] = ta.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
        df['CDLMORNINGSTAR'] = ta.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
        
        # Statistical features
        df['z_score'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = ta.BBANDS(
            df['close'], timeperiod=20)
        
        # Drop initial NaN values
        return df.dropna()
