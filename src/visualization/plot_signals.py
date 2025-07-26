import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import logging

logger = logging.getLogger('SignalVisualizer')

class SignalVisualizer:
    @staticmethod
    def plot_candlestick(df: pd.DataFrame, signals: List[Dict[str, Any]], 
                         title: str = "Trading Signals") -> plt.Figure:
        """Plot candlestick chart with trading signals"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot candlesticks
        for i in range(len(df)):
            row = df.iloc[i]
            color = 'green' if row['close'] >= row['open'] else 'red'
            plt.plot([row['timestamp'], row['timestamp']], 
                     [row['low'], row['high']], color=color, linewidth=1)
            plt.plot([row['timestamp'], row['timestamp']], 
                     [row['open'], row['close']], color=color, linewidth=3)
        
        # Plot signals
        for signal in signals:
            timestamp = signal['timestamp']
            price = signal['price']
            direction = signal['direction']
            signal_type = signal.get('signal_type', 'PO_Signal')
            
            if signal_type == 'PO_Signal':
                marker = '^' if direction == 'BUY' else 'v'
                color = 'blue'
                size = 80
            else:  # PO_AI_Signal
                marker = 'P' if direction == 'BUY' else 'X'
                color = 'lime' if 'HIGH' in signal.get('signal_strength', '') else 'orange'
                size = 120
            
            # Add success indicator if available
            if 'validation' in signal:
                color = 'green' if signal['validation']['success'] else 'red'
            
            plt.scatter(timestamp, price, marker=marker, color=color, 
                        s=size, edgecolors='black', zorder=5)
        
        # Format plot
        plt.title(f"{title} - {signals[0]['symbol'] if signals else ''}")
        plt.xlabel('Time')
        plt.ylabel('Price')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.xticks(rotation=45)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        logger.info("Generated signal visualization")
        return fig

    @staticmethod
    def plot_performance(performance_data: Dict[str, Any]) -> plt.Figure:
        """Plot model performance metrics"""
        fig, ax = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Metrics')
        
        # Accuracy plot
        ax[0, 0].plot(performance_data['accuracy_history'], label='Accuracy')
        ax[0, 0].set_title('Accuracy Over Time')
        ax[0, 0].set_ylabel('Accuracy')
        ax[0, 0].grid(True)
        
        # Precision-Recall plot
        ax[0, 1].plot(performance_data['precision_history'], label='Precision')
        ax[0, 1].plot(performance_data['recall_history'], label='Recall')
        ax[0, 1].set_title('Precision & Recall')
        ax[0, 1].legend()
        ax[0, 1].grid(True)
        
        # Confusion matrix
        cm = performance_data.get('confusion_matrix', [[0, 0], [0, 0]])
        im = ax[1, 0].imshow(cm, cmap='Blues')
        ax[1, 0].set_title('Confusion Matrix')
        ax[1, 0].set_xticks([0, 1])
        ax[1, 0].set_xticklabels(['Down', 'Up'])
        ax[1, 0].set_yticks([0, 1])
        ax[1, 0].set_yticklabels(['Down', 'Up'])
        for i in range(2):
            for j in range(2):
                ax[1, 0].text(j, i, str(cm[i][j]), 
                             ha='center', va='center', color='white' if cm[i][j] > cm.max()/2 else 'black')
        
        # Feature importance
        if 'feature_importance' in performance_data:
            features = list(performance_data['feature_importance'].keys())
            importance = list(performance_data['feature_importance'].values())
            y_pos = np.arange(len(features))
            ax[1, 1].barh(y_pos, importance, align='center')
            ax[1, 1].set_yticks(y_pos)
            ax[1, 1].set_yticklabels(features)
            ax[1, 1].set_title('Feature Importance')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        logger.info("Generated performance visualization")
        return fig
