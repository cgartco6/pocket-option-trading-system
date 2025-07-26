import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
import logging
from config import settings

logger = logging.getLogger('ModelTrainer')

class LSTMModelTrainer:
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.model_path = f"data/models/{symbol}_{timeframe}_model.h5"
        self.scaler_path = f"data/models/{symbol}_{timeframe}_scaler.pkl"
        self.seq_length = settings.SEQUENCE_LENGTH
        logger.info(f"Initialized ModelTrainer for {symbol} {timeframe}")

    def create_model(self, input_shape: tuple) -> Sequential:
        """Create LSTM model architecture"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ]
        )
        
        logger.info("Created LSTM model architecture")
        return model

    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare training data sequences"""
        # Create target - next candle direction
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Remove timestamp and target from features
        feature_cols = [col for col in df.columns if col not in ['timestamp', 'target']]
        X = df[feature_cols].values
        y = df['target'].values
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, self.scaler_path)
        
        # Create sequences
        X_seq, y_seq = [], []
        for i in range(len(df) - self.seq_length - 1):
            X_seq.append(X_scaled[i:i+self.seq_length])
            y_seq.append(y[i+self.seq_length])
        
        return np.array(X_seq), np.array(y_seq)

    def train(self, df: pd.DataFrame) -> dict:
        """Train and save the LSTM model"""
        logger.info(f"Starting model training with {len(df)} samples")
        
        try:
            # Prepare data
            X, y = self.prepare_data(df)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Create model
            model = self.create_model((X_train.shape[1], X_train.shape[2]))
            
            # Callbacks
            callbacks = [
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=self.model_path,
                    save_best_only=True,
                    monitor='val_accuracy',
                    mode='max'
                )
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=64,
                validation_data=(X_test, y_test),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            
            # Save final model
            model.save(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
            
            return {
                'status': 'success',
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'history': history.history
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def evaluate_model(self, df: pd.DataFrame) -> dict:
        """Evaluate model performance on new data"""
        try:
            # Load model
            model = tf.keras.models.load_model(self.model_path)
            
            # Prepare data
            X, y = self.prepare_data(df)
            
            # Make predictions
            y_pred = (model.predict(X) > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred)
            recall = recall_score(y, y_pred)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'total_samples': len(y)
            }
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            return {'error': str(e)}
