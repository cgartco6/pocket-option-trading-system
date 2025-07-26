import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os
from config import settings

class LSTMModelTrainer:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_path = f"data/models/{symbol}_{timeframe}_model.h5"
        self.scaler_path = f"data/models/{symbol}_{timeframe}_scaler.pkl"
        self.seq_length = 60
        
    def create_model(self, input_shape):
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
            Dropout(0.3),
            LSTM(64),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        return model
    
    def prepare_data(self, df):
        # Create target - next candle direction
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Scale features
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df.drop(columns=['timestamp', 'target']))
        joblib.dump(scaler, self.scaler_path)
        
        # Create sequences
        X, y = [], []
        for i in range(len(df) - self.seq_length - 1):
            X.append(scaled_data[i:i+self.seq_length])
            y.append(df['target'].iloc[i+self.seq_length])
        
        return np.array(X), np.array(y), scaler
    
    def train(self, df):
        X, y, scaler = self.prepare_data(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        model = self.create_model((X_train.shape[1], X_train.shape[2]))
        
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stop],
            verbose=1
        )
        
        model.save(self.model_path)
        return history.history
