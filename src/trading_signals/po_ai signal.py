import numpy as np
import tensorflow as tf
from config import settings

class POAISignalGenerator:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model = tf.keras.models.load_model(
            f"data/models/{symbol}_{timeframe}_model.h5")
        self.scaler = joblib.load(f"data/models/{symbol}_{timeframe}_scaler.pkl")
        self.seq_length = 60
        self.thresholds = settings.SIGNAL_THRESHOLDS['PO_AI_SIGNAL']
    
    def generate_signal(self, df):
        # Filter for volatility
        if df['atr'].iloc[-1] < df['atr'].rolling(20).mean().iloc[-1] * self.thresholds['VOLATILITY_FILTER']:
            return None
        
        # Prepare last sequence
        latest_data = df.iloc[-self.seq_length:]
        scaled_data = self.scaler.transform(
            latest_data.drop(columns=['timestamp']))
        sequence = scaled_data.reshape(1, self.seq_length, -1)
        
        # Predict
        prediction = self.model.predict(sequence)[0][0]
        
        # Generate signal
        if prediction > self.thresholds['CONFIDENCE_THRESHOLD']:
            return ('BUY', df['timestamp'].iloc[-1], self.timeframe, prediction)
        elif prediction < 1 - self.thresholds['CONFIDENCE_THRESHOLD']:
            return ('SELL', df['timestamp'].iloc[-1], self.timeframe, 1 - prediction)
        return None
