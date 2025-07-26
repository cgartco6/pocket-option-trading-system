import pandas as pd
import numpy as np
import talib as ta
import logging
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('PO_Signal')

class POSignalGenerator:
    def __init__(self, symbol: str, timeframe: str):
        """
        Initialize PO Signal Generator
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            timeframe (str): Timeframe (e.g., '5m')
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = settings.PO_SIGNAL_CONFIG
        logger.info(f"Initialized PO Signal Generator for {symbol} {timeframe}")
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for signal generation
        
        Args:
            df (DataFrame): Cleaned market data
            
        Returns:
            DataFrame: Data with technical indicators
        """
        # RSI
        df['rsi'] = ta.RSI(df['close'], timeperiod=self.config['RSI_PERIOD'])
        
        # MACD
        df['macd'], df['macd_signal'], _ = ta.MACD(
            df['close'],
            fastperiod=self.config['MACD_FAST'],
            slowperiod=self.config['MACD_SLOW'],
            signalperiod=self.config['MACD_SIGNAL']
        )
        
        # Bollinger Bands
        df['bollinger_upper'], df['bollinger_middle'], df['bollinger_lower'] = ta.BBANDS(
            df['close'],
            timeperiod=self.config['BOLLINGER_PERIOD']
        )
        
        # Stochastic Oscillator
        df['stoch_k'], df['stoch_d'] = ta.STOCH(
            df['high'], df['low'], df['close'],
            fastk_period=self.config['STOCH_K_PERIOD'],
            slowk_period=self.config['STOCH_D_PERIOD'],
            slowk_matype=0,
            slowd_period=self.config['STOCH_D_PERIOD'],
            slowd_matype=0
        )
        
        # Volume analysis
        df['volume_sma'] = df['volume'].rolling(self.config['VOLUME_SMA_PERIOD']).mean()
        
        return df.dropna()
    
    def _generate_buy_signals(self, row: pd.Series, prev_row: pd.Series) -> bool:
        """
        Generate BUY signals based on technical indicators
        
        Args:
            row (Series): Current candle
            prev_row (Series): Previous candle
            
        Returns:
            bool: True if BUY signal generated
        """
        # RSI oversold condition
        rsi_condition = row['rsi'] < self.config['RSI_BUY_THRESHOLD']
        
        # MACD crossover
        macd_condition = (row['macd'] > row['macd_signal']) and (prev_row['macd'] <= prev_row['macd_signal'])
        
        # Price near Bollinger lower band
        bollinger_condition = row['close'] < row['bollinger_lower'] * (1 + self.config['BOLLINGER_BUY_THRESHOLD'])
        
        # Stochastic crossover
        stoch_condition = (row['stoch_k'] > row['stoch_d']) and (prev_row['stoch_k'] <= prev_row['stoch_d'])
        
        # Volume spike
        volume_condition = row['volume'] > row['volume_sma'] * self.config['VOLUME_BUY_MULTIPLIER']
        
        # Minimum number of conditions met
        conditions = [rsi_condition, macd_condition, bollinger_condition, stoch_condition, volume_condition]
        return sum(conditions) >= self.config['MIN_BUY_CONDITIONS']
    
    def _generate_sell_signals(self, row: pd.Series, prev_row: pd.Series) -> bool:
        """
        Generate SELL signals based on technical indicators
        
        Args:
            row (Series): Current candle
            prev_row (Series): Previous candle
            
        Returns:
            bool: True if SELL signal generated
        """
        # RSI overbought condition
        rsi_condition = row['rsi'] > self.config['RSI_SELL_THRESHOLD']
        
        # MACD crossunder
        macd_condition = (row['macd'] < row['macd_signal']) and (prev_row['macd'] >= prev_row['macd_signal'])
        
        # Price near Bollinger upper band
        bollinger_condition = row['close'] > row['bollinger_upper'] * (1 - self.config['BOLLINGER_SELL_THRESHOLD'])
        
        # Stochastic crossunder
        stoch_condition = (row['stoch_k'] < row['stoch_d']) and (prev_row['stoch_k'] >= prev_row['stoch_d'])
        
        # Volume spike
        volume_condition = row['volume'] > row['volume_sma'] * self.config['VOLUME_SELL_MULTIPLIER']
        
        # Minimum number of conditions met
        conditions = [rsi_condition, macd_condition, bollinger_condition, stoch_condition, volume_condition]
        return sum(conditions) >= self.config['MIN_SELL_CONDITIONS']
    
    def generate_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate PO trading signals
        
        Args:
            df (DataFrame): Cleaned market data with OHLCV
            
        Returns:
            List: Trading signals with metadata
        """
        logger.info(f"Generating PO signals for {self.symbol} {self.timeframe}")
        
        try:
            # Calculate indicators
            df = self._calculate_indicators(df.copy())
            
            # Generate signals
            signals = []
            for i in range(1, len(df)):
                current = df.iloc[i]
                previous = df.iloc[i-1]
                
                signal = None
                signal_type = None
                
                if self._generate_buy_signals(current, previous):
                    signal = 'BUY'
                    signal_type = 'PO_Signal'
                elif self._generate_sell_signals(current, previous):
                    signal = 'SELL'
                    signal_type = 'PO_Signal'
                
                if signal:
                    signals.append({
                        'signal_type': signal_type,
                        'symbol': self.symbol,
                        'timeframe': self.timeframe,
                        'timestamp': current['timestamp'],
                        'direction': signal,
                        'price': current['close'],
                        'indicators': {
                            'rsi': current['rsi'],
                            'macd': current['macd'],
                            'macd_signal': current['macd_signal'],
                            'bollinger_upper': current['bollinger_upper'],
                            'bollinger_lower': current['bollinger_lower'],
                            'stoch_k': current['stoch_k'],
                            'stoch_d': current['stoch_d'],
                            'volume': current['volume'],
                            'volume_sma': current['volume_sma']
                        }
                    })
            
            logger.info(f"Generated {len(signals)} PO signals")
            return signals
        
        except Exception as e:
            logger.error(f"Signal generation failed: {str(e)}")
            return []
