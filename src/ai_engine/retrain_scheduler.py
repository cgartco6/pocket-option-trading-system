import time
import schedule
import threading
import logging
from datetime import datetime, timedelta
from config import settings
from src.ai_engine.model_trainer import ModelTrainer
from src.data_processing.data_fetcher import DataFetcher
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.feature_engineer import FeatureEngineer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('RetrainScheduler')

class RetrainScheduler:
    def __init__(self):
        self.symbols = settings.SYMBOLS
        self.timeframes = settings.TIMEFRAMES
        self.retrain_interval = settings.RETRAIN_INTERVAL
        self.data_lookback = settings.DATA_LOOKBACK
        self.performance_thresholds = settings.PERFORMANCE_THRESHOLDS
        self.training_queue = []
        self.training_lock = threading.Lock()
        self.last_retrain = {}
        self.initialize_schedule()
        logger.info("Initialized RetrainScheduler")

    def initialize_schedule(self):
        """Setup scheduled retraining tasks"""
        # Daily retraining
        schedule.every().day.at(self.retrain_interval['daily']).do(
            self.schedule_retraining, 'daily'
        )
        
        # Weekly retraining
        schedule.every().sunday.at(self.retrain_interval['weekly']).do(
            self.schedule_retraining, 'weekly'
        )
        
        # Performance-based retraining checker
        schedule.every().hour.do(self.check_performance)

    def schedule_retraining(self, schedule_type='daily'):
        """
        Add models to retraining queue
        
        Args:
            schedule_type (str): 'daily' or 'weekly'
        """
        with self.training_lock:
            for symbol in self.symbols:
                for timeframe in self.timeframes:
                    # Check if recently trained
                    last_train = self.last_retrain.get((symbol, timeframe), None)
                    if last_train and (datetime.utcnow() - last_train) < timedelta(hours=6):
                        logger.info(f"Skipping {symbol}-{timeframe} - recently trained")
                        continue
                    
                    # Add to queue
                    self.training_queue.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'schedule_type': schedule_type,
                        'priority': 1 if schedule_type == 'daily' else 2
                    })
            logger.info(f"Scheduled {len(self.training_queue)} models for retraining ({schedule_type})")

    def check_performance(self):
        """
        Check model performance and schedule retraining if below thresholds
        """
        # This would integrate with performance monitoring system
        # Placeholder implementation
        logger.info("Checking model performance...")
        
        # In a real system, we would:
        # 1. Get recent performance metrics
        # 2. Identify underperforming models
        # 3. Add them to the retraining queue
        
        # For demo purposes, we'll randomly select one model
        if self.symbols and self.timeframes:
            symbol = self.symbols[0]
            timeframe = self.timeframes[0]
            
            # Add to queue with high priority
            with self.training_lock:
                self.training_queue.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'schedule_type': 'performance',
                    'priority': 0  # Highest priority
                })
            logger.info(f"Scheduled {symbol}-{timeframe} for performance-based retraining")

    def process_training_queue(self):
        """Process retraining queue in a separate thread"""
        while True:
            if self.training_queue:
                # Sort by priority (lowest number = highest priority)
                with self.training_lock:
                    self.training_queue.sort(key=lambda x: x['priority'])
                    task = self.training_queue.pop(0)
                
                # Execute training
                try:
                    self.retrain_model(
                        task['symbol'],
                        task['timeframe'],
                        task['schedule_type']
                    )
                except Exception as e:
                    logger.error(f"Retraining failed for {task['symbol']}-{task['timeframe']}: {str(e)}")
            
            time.sleep(10)  # Check queue every 10 seconds

    def retrain_model(self, symbol, timeframe, schedule_type):
        """
        Execute model retraining
        
        Args:
            symbol (str): Trading symbol
            timeframe (str): Timeframe
            schedule_type (str): Type of retraining trigger
        """
        logger.info(f"Starting retraining for {symbol}-{timeframe} ({schedule_type})")
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Fetch data
            fetcher = DataFetcher(symbol, timeframe)
            raw_data = fetcher.fetch_historical(limit=self.data_lookback[schedule_type])
            
            # Step 2: Clean data
            cleaner = DataCleaner(symbol, timeframe)
            clean_data = cleaner.clean(raw_data)
            
            # Step 3: Feature engineering
            engineer = FeatureEngineer()
            feature_data = engineer.add_features(clean_data)
            
            # Step 4: Train model
            trainer = ModelTrainer(symbol, timeframe)
            history = trainer.train(feature_data)
            
            # Step 5: Evaluate model
            accuracy = history['val_accuracy'][-1]
            precision = history['val_precision'][-1]
            recall = history['val_recall'][-1]
            
            # Step 6: Update records
            self.last_retrain[(symbol, timeframe)] = datetime.utcnow()
            
            # Step 7: Notify if performance below threshold
            if accuracy < self.performance_thresholds['min_accuracy']:
                logger.warning(f"Model accuracy below threshold: {accuracy:.2%}")
            
            # Log results
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(f"Retraining completed for {symbol}-{timeframe} in {duration:.1f}s - "
                        f"Accuracy: {accuracy:.2%}, Precision: {precision:.2%}, Recall: {recall:.2%}")
            
            return {
                'status': 'success',
                'symbol': symbol,
                'timeframe': timeframe,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'duration': duration,
                'schedule_type': schedule_type
            }
            
        except Exception as e:
            logger.error(f"Retraining failed for {symbol}-{timeframe}: {str(e)}")
            return {
                'status': 'error',
                'symbol': symbol,
                'timeframe': timeframe,
                'error': str(e),
                'schedule_type': schedule_type
            }

    def start(self):
        """Start scheduler in background thread"""
        logger.info("Starting retrain scheduler")
        
        # Start schedule runner in background
        schedule_thread = threading.Thread(target=self.run_schedule, daemon=True)
        schedule_thread.start()
        
        # Start training processor
        training_thread = threading.Thread(target=self.process_training_queue, daemon=True)
        training_thread.start()
        
        logger.info("Retrain scheduler running in background")

    def run_schedule(self):
        """Run scheduled tasks continuously"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

# Example configuration would be in config/settings.py
# settings.RETRAIN_INTERVAL = {
#     'daily': '03:00',
#     'weekly': '04:00'
# }
# 
# settings.DATA_LOOKBACK = {
#     'daily': 1000,
#     'weekly': 5000,
#     'performance': 2000
# }
# 
# settings.PERFORMANCE_THRESHOLDS = {
#     'min_accuracy': 0.65,
#     'min_precision': 0.60,
#     'min_recall': 0.55
# }
# 
# settings.SEQUENCE_LENGTH = 60
# 
# settings.PREDICTION_THRESHOLDS = {
#     'HIGH_CONFIDENCE': 0.75,
#     'MEDIUM_CONFIDENCE': 0.65
# }
