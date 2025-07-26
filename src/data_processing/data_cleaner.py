import pandas as pd
import numpy as np
import re
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer
from scipy import stats
import talib as ta
import logging
from config import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('DataCleaner')

class DataCleaner:
    def __init__(self, symbol, timeframe):
        """
        Initialize data cleaner with asset and timeframe
        
        Args:
            symbol (str): Trading symbol (e.g., 'BTCUSDT')
            timeframe (str): Timeframe (e.g., '5m')
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.config = settings.DATA_CLEANING_CONFIG
        logger.info(f"Initialized DataCleaner for {symbol} {timeframe}")
    
    def _detect_anomalies(self, df):
        """
        Detect anomalies using statistical methods
        """
        # Price anomaly detection
        df['price_zscore'] = np.abs(stats.zscore(df['close'], nan_policy='omit'))
        
        # Volume anomaly detection
        volume_zscore = np.abs(stats.zscore(df['volume'], nan_policy='omit'))
        df['volume_anomaly'] = (volume_zscore > self.config['VOLUME_ZSCORE_THRESHOLD'])
        
        # Candlestick anomaly detection
        df['candle_range'] = df['high'] - df['low']
        df['range_zscore'] = np.abs(stats.zscore(df['candle_range'], nan_policy='omit'))
        
        return df
    
    def _handle_missing_values(self, df):
        """
        Handle missing values using advanced imputation
        """
        # Forward fill for minor gaps
        df.ffill(inplace=True)
        
        # Time-based interpolation for larger gaps
        if df.isnull().sum().sum() > 0:
            logger.warning(f"Missing values detected: {df.isnull().sum()}")
            
            # Create imputation strategy dictionary
            impute_strategies = {
                'open': 'mean',
                'high': 'mean',
                'low': 'mean',
                'close': 'mean',
                'volume': 'constant'
            }
            
            for col, strategy in impute_strategies.items():
                if df[col].isnull().any():
                    imputer = SimpleImputer(strategy=strategy, fill_value=0 if strategy == 'constant' else None)
                    df[col] = imputer.fit_transform(df[[col]])
        
        return df
    
    def _handle_outliers(self, df):
        """
        Identify and treat outliers using robust methods
        """
        # 1. Winsorize extreme values
        for col in ['open', 'high', 'low', 'close']:
            df[col] = stats.mstats.winsorize(
                df[col], 
                limits=self.config['WINSORIZE_LIMITS']
            )
        
        # 2. Transform volume to reduce skewness
        if 'volume' in df.columns:
            volume_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
            df['volume'] = volume_transformer.fit_transform(df[['volume']])
        
        return df
    
    def _validate_candles(self, df):
        """
        Validate candle integrity and fix inconsistencies
        """
        # 1. Validate OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC candles")
            
            # Fix strategy: Use neighboring candles to reconstruct
            for idx in df[invalid_ohlc].index:
                prev_idx = max(df.index[0], idx - 1)
                next_idx = min(df.index[-1], idx + 1)
                
                df.loc[idx, 'high'] = max(
                    df.loc[prev_idx, 'close'], 
                    df.loc[next_idx, 'open'],
                    df.loc[idx, 'open'],
                    df.loc[idx, 'close']
                )
                
                df.loc[idx, 'low'] = min(
                    df.loc[prev_idx, 'close'], 
                    df.loc[next_idx, 'open'],
                    df.loc[idx, 'open'],
                    df.loc[idx, 'close']
                )
        
        # 2. Validate volume
        if 'volume' in df.columns:
            negative_volume = df['volume'] < 0
            if negative_volume.any():
                df.loc[negative_volume, 'volume'] = 0
        
        return df
    
    def _remove_duplicates(self, df):
        """Remove duplicate timestamps and consolidate data"""
        duplicates = df.index.duplicated(keep='first')
        if duplicates.any():
            logger.warning(f"Found {duplicates.sum()} duplicate timestamps")
            
            # Aggregate duplicates (e.g., when multiple sources)
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            df = df.groupby(df.index).agg(agg_rules)
        
        return df
    
    def _normalize_structure(self, df):
        """
        Ensure consistent data structure and columns
        """
        # Ensure standard columns
        expected_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        for col in expected_cols:
            if col not in df.columns:
                df[col] = np.nan
        
        # Convert timestamp to datetime index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        
        # Sort chronologically
        df.sort_index(inplace=True)
        
        return df[expected_cols]
    
    def _add_quality_metrics(self, df):
        """
        Add data quality metrics for monitoring
        """
        # Data completeness score
        df['completeness'] = 1 - df.isnull().mean(axis=1)
        
        # Volatility consistency
        df['range_ratio'] = (df['high'] - df['low']) / (df['close'].rolling(5).std() + 1e-8)
        
        # Volume consistency
        if 'volume' in df.columns:
            df['volume_z'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-8)
        
        return df
    
    def clean(self, raw_df):
        """
        Full data cleaning pipeline
        
        Args:
            raw_df (DataFrame): Raw input data
            
        Returns:
            DataFrame: Cleaned and validated data
        """
        logger.info(f"Starting cleaning for {self.symbol} {self.timeframe} - {len(raw_df)} rows")
        
        try:
            # 1. Normalize data structure
            df = self._normalize_structure(raw_df.copy())
            
            # 2. Handle duplicates
            df = self._remove_duplicates(df)
            
            # 3. Handle missing values
            df = self._handle_missing_values(df)
            
            # 4. Validate candle integrity
            df = self._validate_candles(df)
            
            # 5. Detect and handle anomalies
            df = self._detect_anomalies(df)
            
            # 6. Handle outliers
            df = self._handle_outliers(df)
            
            # 7. Add quality metrics
            df = self._add_quality_metrics(df)
            
            # 8. Final validation
            self._validate_output(df)
            
            logger.info(f"Cleaning complete - {len(df)} valid rows")
            return df
        
        except Exception as e:
            logger.error(f"Cleaning failed: {str(e)}")
            raise RuntimeError(f"Data cleaning failed for {self.symbol}") from e
    
    def _validate_output(self, df):
        """Validate cleaned data meets quality standards"""
        # Check for NaNs
        if df.isnull().sum().sum() > 0:
            missing_counts = df.isnull().sum()
            raise ValueError(f"Data contains missing values after cleaning:\n{missing_counts}")
        
        # Check OHLC relationships
        ohlc_valid = (
            (df['high'] >= df[['open', 'close', 'low']].max(axis=1)) &
            (df['low'] <= df[['open', 'close', 'high']].min(axis=1))
        ).all()
        
        if not ohlc_valid:
            invalid_count = (~ohlc_valid).sum()
            raise ValueError(f"{invalid_count} candles have invalid OHLC relationships")
        
        # Check volume
        if 'volume' in df.columns:
            if (df['volume'] < 0).any():
                negative_count = (df['volume'] < 0).sum()
                raise ValueError(f"{negative_count} negative volume values found")
        
        # Check timestamp consistency
        time_gaps = df.index.to_series().diff().dropna()
        expected_interval = pd.Timedelta(self.timeframe)
        tolerance = expected_interval * 0.2  # 20% tolerance
        
        irregular_intervals = time_gaps[(time_gaps > expected_interval + tolerance) | 
                                       (time_gaps < expected_interval - tolerance)]
        
        if not irregular_intervals.empty:
            gap_count = len(irregular_intervals)
            raise ValueError(f"Found {gap_count} irregular time intervals in data")
        
        logger.info("Output validation passed")

# Example configuration would be in config/settings.py
# settings.DATA_CLEANING_CONFIG = {
#     'VOLUME_ZSCORE_THRESHOLD': 4.0,
#     'WINSORIZE_LIMITS': [0.01, 0.01],  # Trim top and bottom 1%
#     'MIN_VALID_COMPLETENESS': 0.95
# }
