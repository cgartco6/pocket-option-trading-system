import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
import logging
from typing import Dict, Any, Optional
from config import settings

logger = logging.getLogger('POAISignal')

class POAISignalGenerator:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_path = f"data/models/{symbol}_{timeframe}_model.h5"
        self.scaler_path = f"data/models/{symbol}_{timeframe}_scaler.pkl"
        self.seq_length = settings.SEQUENCE_LENGTH
        self.thresholds = settings.PREDICTION_THRESHOLDS
        self.model = self._load_model()
        self.scaler = self._load_scaler()
        logger.info(f"Initialized PO AI Signal Generator for {symbol} {timeframe}")

    def _load_model(self) -> Optional[tf.keras.Model]:
        """Load trained LSTM model"""
        try:
            return tf.keras.models.load_model(self.model_path)
        except:
            logger.warning(f"Model not found at {self.model_path}")
            return None

    def _load_scaler(self) -> Optional[StandardScaler]:
        """Load feature scaler"""
        try:
            return joblib.load(self.scaler_path)
        except:
            logger.warning(f"Scaler not found at {self.scaler_path}")
            return None

    def generate_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate AI-enhanced trading signal"""
        if not self.model or not self.scaler:
            return None
        
        # Filter for volatility
        if 'atr' in df.columns:
            atr_ma = df['atr'].rolling(20).mean().iloc[-1]
            if df['atr'].iloc[-1] < atr_ma * 0.7:
                logger.debug("Skipping signal due to low volatility")
                return None
        
        # Prepare last sequence
        feature_cols = [col for col in df.columns if col != 'timestamp']
        latest_data = df[feature_cols].iloc[-self.seq_length:]
        
        try:
            # Scale data
            scaled_data = self.scaler.transform(latest_data)
            sequence = scaled_data.reshape(1, self.seq_length, -1)
            
            # Predict
            prediction = self.model.predict(sequence, verbose=0)[0][0]
            confidence = max(prediction, 1 - prediction)
            direction = "BUY" if prediction > 0.5 else "SELL"
            
            # Generate signal based on confidence
            if confidence > self.thresholds['HIGH_CONFIDENCE']:
                signal_strength = "HIGH"
            elif confidence > self.thresholds['MEDIUM_CONFIDENCE']:
                signal_strength = "MEDIUM"
            else:
                return None
            
            return {
                'signal_type': 'PO_AI_Signal',
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'timestamp': df['timestamp'].iloc[-1],
                'direction': direction,
                'confidence': float(confidence),
                'signal_strength': signal_strength,
                'price': df['close'].iloc[-1],
                'features': latest_data.iloc[-1].to_dict()
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return None
