import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SignalValidator')

class SignalValidator:
    def __init__(self):
        """
        Initialize Signal Validator
        """
        self.config = settings.SIGNAL_VALIDATION_CONFIG
        logger.info("Initialized Signal Validator")
    
    def _validate_single_signal(self, signal: Dict[str, Any], next_candle: pd.Series) -> Dict[str, Any]:
        """
        Validate a single signal against the next candle
        
        Args:
            signal (dict): Trading signal
            next_candle (Series): Next candle data
            
        Returns:
            dict: Validated signal with results
        """
        validated_signal = signal.copy()
        
        # Determine success based on signal direction
        if signal['direction'] == 'BUY':
            success = next_candle['close'] > signal['price']
        elif signal['direction'] == 'SELL':
            success = next_candle['close'] < signal['price']
        else:
            success = False
        
        # Add validation results
        validated_signal['validation'] = {
            'timestamp': datetime.utcnow(),
            'next_candle_open': next_candle['open'],
            'next_candle_high': next_candle['high'],
            'next_candle_low': next_candle['low'],
            'next_candle_close': next_candle['close'],
            'success': success,
            'price_change': (next_candle['close'] - signal['price']) / signal['price'],
            'color': 'green' if success else 'red'
        }
        
        # Add noise classification
        validated_signal['is_noise'] = self._classify_as_noise(validated_signal, next_candle)
        
        return validated_signal
    
    def _classify_as_noise(self, signal: Dict[str, Any], next_candle: pd.Series) -> bool:
        """
        Classify if signal is noise based on market conditions
        
        Args:
            signal (dict): Trading signal
            next_candle (Series): Next candle data
            
        Returns:
            bool: True if signal is classified as noise
        """
        # Small price movement
        price_change = abs(signal['validation']['price_change'])
        if price_change < self.config['MIN_PRICE_CHANGE']:
            return True
        
        # Volatility too low
        candle_range = next_candle['high'] - next_candle['low']
        if candle_range < signal['price'] * self.config['MIN_VOLATILITY_THRESHOLD']:
            return True
        
        # Failed signal with small reversal
        if not signal['validation']['success']:
            reversal_size = abs(next_candle['close'] - signal['price'])
            if reversal_size < signal['price'] * self.config['FAILED_SIGNAL_REVERSAL_THRESHOLD']:
                return True
        
        # Volume too low
        if 'indicators' in signal and 'volume' in signal['indicators']:
            volume = signal['indicators']['volume']
            volume_sma = signal['indicators']['volume_sma']
            if volume < volume_sma * self.config['MIN_VOLUME_RATIO']:
                return True
        
        return False
    
    def validate_signals(self, signals: List[Dict[str, Any]], df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Validate signals against the next candle
        
        Args:
            signals (list): List of trading signals
            df (DataFrame): Market data with OHLCV
            
        Returns:
            list: Validated signals with results
        """
        logger.info(f"Validating {len(signals)} signals")
        
        validated_signals = []
        df = df.set_index('timestamp')
        
        for signal in signals:
            signal_time = signal['timestamp']
            
            try:
                # Find next candle
                if signal_time not in df.index:
                    logger.warning(f"Signal timestamp {signal_time} not found in data")
                    continue
                
                signal_index = df.index.get_loc(signal_time)
                if signal_index >= len(df) - 1:
                    logger.warning(f"No next candle available for signal at {signal_time}")
                    continue
                
                next_candle = df.iloc[signal_index + 1]
                
                # Validate signal
                validated = self._validate_single_signal(signal, next_candle)
                validated_signals.append(validated)
                
                logger.debug(f"Validated signal at {signal_time}: " 
                             f"{'SUCCESS' if validated['validation']['success'] else 'FAIL'} "
                             f"{'NOISE' if validated['is_noise'] else ''}")
            
            except Exception as e:
                logger.error(f"Validation failed for signal at {signal_time}: {str(e)}")
        
        logger.info(f"Validated {len(validated_signals)} signals")
        return validated_signals
    
    def filter_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filter signals to remove noise and failed signals
        
        Args:
            signals (list): List of validated signals
            
        Returns:
            list: Filtered signals
        """
        logger.info(f"Filtering {len(signals)} signals")
        
        # Filter out noise and failed signals
        filtered = [
            s for s in signals 
            if not s['is_noise'] and s['validation']['success']
        ]
        
        # Filter out small movements
        filtered = [
            s for s in filtered
            if abs(s['validation']['price_change']) >= self.config['MIN_PROFITABLE_CHANGE']
        ]
        
        logger.info(f"Filtered to {len(filtered)} high-quality signals")
        return filtered
    
    def generate_pure_signals(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate pure surefire signals (PO AI Signal)
        
        Args:
            signals (list): List of validated and filtered signals
            
        Returns:
            list: Pure signals with AI confidence
        """
        logger.info(f"Generating pure signals from {len(signals)} inputs")
        
        pure_signals = []
        for signal in signals:
            # Calculate AI confidence score
            confidence = self._calculate_confidence(signal)
            
            if confidence >= self.config['PURE_SIGNAL_CONFIDENCE_THRESHOLD']:
                pure_signal = signal.copy()
                pure_signal['signal_type'] = 'PO_AI_Signal'
                pure_signal['ai_confidence'] = confidence
                pure_signals.append(pure_signal)
                
                logger.info(f"Generated PO AI Signal at {signal['timestamp']} "
                            f"with confidence {confidence:.2%}")
        
        logger.info(f"Generated {len(pure_signals)} PO AI Signals")
        return pure_signals
    
    def _calculate_confidence(self, signal: Dict[str, Any]) -> float:
        """
        Calculate AI confidence score for a signal
        
        Args:
            signal (dict): Validated signal
            
        Returns:
            float: Confidence score (0-1)
        """
        # Base confidence from validation success
        confidence = 1.0 if signal['validation']['success'] else 0.0
        
        # Add factors based on market conditions
        price_change = abs(signal['validation']['price_change'])
        confidence += min(price_change / self.config['MAX_PRICE_CHANGE_FACTOR'], 0.3)
        
        # Volume factor
        if 'indicators' in signal:
            volume_ratio = signal['indicators']['volume'] / signal['indicators']['volume_sma']
            confidence += min((volume_ratio - 1) * 0.2, 0.2)
        
        # Volatility factor
        candle_range = (signal['validation']['next_candle_high'] - 
                        signal['validation']['next_candle_low'])
        volatility_ratio = candle_range / signal['price']
        confidence += min(volatility_ratio / self.config['VOLATILITY_CONFIDENCE_FACTOR'], 0.2)
        
        # Cap confidence at 95%
        return min(confidence, 0.95)
