import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import logging
from datetime import datetime, timedelta
from config import settings
from src.data_processing.feature_engineer import FeatureEngineer
from src.data_processing.data_cleaner import DataCleaner

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Predictor')

class MarketPredictor:
    def __init__(self, symbol, timeframe):
        """
        Initialize market predictor for specific asset and timeframe
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            timeframe (str): Timeframe (e.g., '5m')
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_path = f"data/models/{symbol}_{timeframe}_model.h5"
        self.scaler_path = f"data/models/{symbol}_{timeframe}_scaler.pkl"
        self.seq_length = settings.SEQUENCE_LENGTH
        self.thresholds = settings.PREDICTION_THRESHOLDS
        self.last_prediction = None
        self.prediction_history = []
        
        # Load model and scaler
        self.load_resources()
        logger.info(f"Initialized MarketPredictor for {symbol} {timeframe}")

    def load_resources(self):
        """Load model and scaler from disk"""
        try:
            self.model = load_model(self.model_path)
            logger.info(f"Loaded model from {self.model_path}")
            
            self.scaler = joblib.load(self.scaler_path)
            logger.info(f"Loaded scaler from {self.scaler_path}")
        except Exception as e:
            logger.error(f"Failed to load resources: {str(e)}")
            raise RuntimeError(f"Resource loading failed for {self.symbol}") from e

    def refresh_model(self):
        """Refresh model from disk (for hot reloading)"""
        try:
            self.model = load_model(self.model_path)
            logger.info(f"Refreshed model from {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Model refresh failed: {str(e)}")
            return False

    def preprocess_data(self, df):
        """
        Prepare data for prediction
        
        Args:
            df (DataFrame): Raw market data
            
        Returns:
            tuple: (processed data, cleaned dataframe)
        """
        try:
            # Clean data
            cleaner = DataCleaner(self.symbol, self.timeframe)
            clean_df = cleaner.clean(df)
            
            # Add features
            feature_engineer = FeatureEngineer()
            feature_df = feature_engineer.add_features(clean_df)
            
            # Scale data
            features_to_scale = [col for col in feature_df.columns if col not in ['timestamp']]
            scaled_data = self.scaler.transform(feature_df[features_to_scale])
            
            # Create sequence
            sequence = scaled_data[-self.seq_length:]
            return sequence.reshape(1, self.seq_length, -1), feature_df
        
        except Exception as e:
            logger.error(f"Data preprocessing failed: {str(e)}")
            raise RuntimeError("Data preparation error") from e

    def predict_next_candle(self, df):
        """
        Predict next candle direction and confidence
        
        Args:
            df (DataFrame): Historical market data including current candle
            
        Returns:
            dict: Prediction results
        """
        try:
            # Preprocess data
            sequence, processed_df = self.preprocess_data(df)
            
            # Make prediction
            raw_prediction = self.model.predict(sequence, verbose=0)[0][0]
            confidence = raw_prediction if raw_prediction > 0.5 else 1 - raw_prediction
            direction = "BUY" if raw_prediction > 0.5 else "SELL"
            
            # Generate signal based on confidence thresholds
            signal = None
            if confidence > self.thresholds['HIGH_CONFIDENCE']:
                signal = direction
            elif confidence > self.thresholds['MEDIUM_CONFIDENCE']:
                signal = f"{direction}_WEAK"
            
            # Prepare result
            result = {
                'timestamp': datetime.utcnow(),
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'direction': direction,
                'confidence': float(confidence),
                'signal': signal,
                'raw_prediction': float(raw_prediction),
                'current_close': float(processed_df['close'].iloc[-1]),
                'features': processed_df.iloc[-1].to_dict()
            }
            
            # Store prediction
            self.last_prediction = result
            self.prediction_history.append(result)
            
            logger.info(f"Prediction: {direction} with {confidence:.2%} confidence")
            return result
        
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow(),
                'symbol': self.symbol,
                'timeframe': self.timeframe
            }

    def validate_prediction(self, actual_df):
        """
        Validate previous prediction against actual market data
        
        Args:
            actual_df (DataFrame): New market data with next candle
            
        Returns:
            dict: Validation results
        """
        if not self.last_prediction:
            logger.warning("No prediction to validate")
            return None
        
        try:
            # Get actual next candle
            next_candle = actual_df.iloc[0]
            actual_direction = "BUY" if next_candle['close'] > next_candle['open'] else "SELL"
            
            # Compare with prediction
            prediction_correct = actual_direction == self.last_prediction['direction']
            price_change = (next_candle['close'] - next_candle['open']) / next_candle['open']
            
            # Calculate validation metrics
            result = {
                'prediction_id': len(self.prediction_history),
                'timestamp': datetime.utcnow(),
                'symbol': self.symbol,
                'timeframe': self.timeframe,
                'predicted_direction': self.last_prediction['direction'],
                'actual_direction': actual_direction,
                'prediction_correct': prediction_correct,
                'confidence': self.last_prediction['confidence'],
                'price_change': float(price_change),
                'signal_generated': self.last_prediction['signal'] is not None,
                'validation_quality': self.calculate_validation_quality(),
                'features': {
                    'predicted': self.last_prediction['features'],
                    'actual': next_candle.to_dict()
                }
            }
            
            logger.info(f"Validation: {'SUCCESS' if prediction_correct else 'FAIL'} "
                        f"(Predicted: {self.last_prediction['direction']}, "
                        f"Actual: {actual_direction})")
            return result
        
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.utcnow(),
                'symbol': self.symbol,
                'timeframe': self.timeframe
            }
    
    def calculate_validation_quality(self):
        """Calculate data quality metrics for validation"""
        # Placeholder for advanced quality metrics
        return {
            'completeness': 1.0,
            'volatility': 0.0,
            'anomaly_score': 0.0
        }
    
    def get_performance_stats(self, lookback=100):
        """
        Calculate prediction performance statistics
        
        Args:
            lookback (int): Number of predictions to consider
            
        Returns:
            dict: Performance metrics
        """
        if len(self.prediction_history) < 10:
            return {}
        
        # Get recent predictions with validation
        valid_predictions = [p for p in self.prediction_history if 'validation' in p]
        if not valid_predictions:
            return {}
        
        # Calculate metrics
        recent = valid_predictions[-lookback:]
        accuracy = sum(1 for p in recent if p['validation']['prediction_correct']) / len(recent)
        avg_confidence = sum(p['confidence'] for p in recent) / len(recent)
        
        # Calculate signal performance
        signals = [p for p in recent if p['signal']]
        signal_accuracy = sum(1 for p in signals if p['validation']['prediction_correct']) / len(signals) if signals else 0
        
        return {
            'total_predictions': len(self.prediction_history),
            'validated_predictions': len(valid_predictions),
            'accuracy': accuracy,
            'signal_accuracy': signal_accuracy,
            'avg_confidence': avg_confidence,
            'last_updated': datetime.utcnow()
        }
