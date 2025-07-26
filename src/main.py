import os
import time
import schedule
import threading
import logging
import pandas as pd
from datetime import datetime
from config import settings
from src.data_processing.data_fetcher import MarketDataFetcher, PocketOptionAPI
from src.data_processing.data_cleaner import DataCleaner
from src.data_processing.feature_engineer import FeatureEngineer
from src.trading_signals.po_signal import POSignalGenerator
from src.trading_signals.po_ai_signal import POAISignalGenerator
from src.trading_signals.signal_validator import SignalValidator
from src.ai_engine.model_trainer import LSTMModelTrainer
from src.ai_engine.retrain_scheduler import RetrainScheduler
from src.ai_engine.predictor import MarketPredictor
from src.visualization.performance_dashboard import TradingDashboard
from src.visualization.plot_signals import SignalVisualizer

logger = logging.getLogger('MainSystem')

class PocketOptionTradingSystem:
    def __init__(self):
        self.symbols = settings.SYMBOLS
        self.timeframes = settings.TIMEFRAMES
        self.data_fetchers = self._init_data_fetchers()
        self.po_api = PocketOptionAPI()
        self.signal_history = {}
        self.performance_metrics = {}
        self.trading_active = False
        logger.info("Initialized PocketOption Trading System")

    def _init_data_fetchers(self) -> dict:
        """Initialize data fetchers for all symbols and timeframes"""
        fetchers = {}
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                fetchers[(symbol, timeframe)] = MarketDataFetcher(symbol, timeframe)
        return fetchers

    def update_market_data(self) -> dict:
        """Fetch and process market data for all symbols"""
        logger.info("Updating market data...")
        market_data = {}
        
        for (symbol, timeframe), fetcher in self.data_fetchers.items():
            try:
                # Fetch and clean data
                raw_data = fetcher.fetch_historical(limit=1000)
                cleaner = DataCleaner(symbol, timeframe)
                clean_data = cleaner.clean(raw_data)
                
                # Add features
                engineer = FeatureEngineer()
                feature_data = engineer.add_features(clean_data)
                
                market_data[(symbol, timeframe)] = feature_data
                logger.debug(f"Updated data for {symbol} {timeframe}")
            except Exception as e:
                logger.error(f"Data update failed for {symbol} {timeframe}: {str(e)}")
        
        logger.info(f"Updated data for {len(market_data)} assets")
        return market_data

    def generate_signals(self, market_data: dict) -> dict:
        """Generate trading signals for all assets"""
        logger.info("Generating trading signals...")
        all_signals = {'PO': {}, 'PO_AI': {}}
        
        for (symbol, timeframe), df in market_data.items():
            # Generate PO signals
            po_gen = POSignalGenerator(symbol, timeframe)
            po_signals = po_gen.generate_signals(df)
            
            # Generate PO AI signals
            ai_gen = POAISignalGenerator(symbol, timeframe)
            ai_signals = []
            for i in range(30, len(df)):
                ai_signal = ai_gen.generate_signal(df.iloc[:i+1])
                if ai_signal:
                    ai_signals.append(ai_signal)
            
            # Store signals
            all_signals['PO'][(symbol, timeframe)] = po_signals
            all_signals['PO_AI'][(symbol, timeframe)] = ai_signals
            
            # Update history
            self.signal_history.setdefault(symbol, []).extend(po_signals + ai_signals)
        
        logger.info(f"Generated {sum(len(v) for v in all_signals['PO'].values())} PO signals")
        logger.info(f"Generated {sum(len(v) for v in all_signals['PO_AI'].values())} PO AI signals")
        return all_signals

    def validate_signals(self, signals: dict, market_data: dict) -> dict:
        """Validate signals against next candle"""
        logger.info("Validating signals...")
        validator = SignalValidator()
        validated_signals = {'PO': {}, 'PO_AI': {}}
        
        for signal_type in ['PO', 'PO_AI']:
            for (symbol, timeframe), signal_list in signals[signal_type].items():
                df = market_data[(symbol, timeframe)]
                validated = validator.validate_signals(signal_list, df)
                filtered = validator.filter_signals(validated)
                
                # Generate pure AI signals
                if signal_type == 'PO_AI':
                    pure_signals = validator.generate_pure_signals(filtered)
                    validated_signals[signal_type][(symbol, timeframe)] = pure_signals
                else:
                    validated_signals[signal_type][(symbol, timeframe)] = filtered
        
        logger.info("Signal validation complete")
        return validated_signals

    def execute_trades(self, signals: dict):
        """Execute trades based on validated signals"""
        if not self.trading_active:
            logger.warning("Trading is not active - skipping execution")
            return
        
        logger.info("Executing trades...")
        for signal_type in ['PO', 'PO_AI']:
            for (symbol, timeframe), signal_list in signals[signal_type].items():
                for signal in signal_list:
                    if signal['signal_type'] == 'PO_AI_Signal' or signal_type == 'PO_AI':
                        logger.info(f"Executing {signal_type} trade: {symbol} {timeframe} {signal['direction']}")
                        result = self.po_api.place_trade(signal)
                        logger.info(f"Trade result: {result}")
                        signal['trade_executed'] = result

    def start_trading(self):
        """Start live trading"""
        self.trading_active = True
        logger.info("Live trading activated")

    def stop_trading(self):
        """Stop live trading"""
        self.trading_active = False
        logger.info("Live trading deactivated")

    def run_trading_cycle(self):
        """Run a complete trading cycle"""
        try:
            # Step 1: Update market data
            market_data = self.update_market_data()
            
            # Step 2: Generate signals
            raw_signals = self.generate_signals(market_data)
            
            # Step 3: Validate signals
            validated_signals = self.validate_signals(raw_signals, market_data)
            
            # Step 4: Execute trades
            self.execute_trades(validated_signals)
            
            # Step 5: Update performance metrics
            self.update_performance_metrics(validated_signals)
            
            logger.info("Trading cycle completed successfully")
        except Exception as e:
            logger.error(f"Trading cycle failed: {str(e)}")

    def update_performance_metrics(self, signals: dict):
        """Update performance tracking metrics"""
        # Placeholder implementation
        logger.info("Updating performance metrics...")
        self.performance_metrics['last_updated'] = datetime.utcnow()
        
        # In a real system, we would calculate:
        # - Win rates for each signal type
        # - Profitability metrics
        # - Model accuracy statistics
        # - Risk/reward ratios

    def start_scheduled_trading(self):
        """Start scheduled trading based on timeframes"""
        logger.info("Starting scheduled trading")
        
        # Schedule trading cycles for each timeframe
        for timeframe in self.timeframes:
            schedule.every().hour.at(":00").do(self.run_trading_cycle).tag(timeframe)
        
        # Start the scheduler in a background thread
        def schedule_runner():
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        threading.Thread(target=schedule_runner, daemon=True).start()

    def run(self):
        """Main system execution loop"""
        logger.info("Starting PocketOption Trading System")
        
        # Initialize AI models
        self.initialize_models()
        
        # Start scheduled trading
        self.start_scheduled_trading()
        
        # Start retraining scheduler
        retrain_scheduler = RetrainScheduler()
        retrain_scheduler.start()
        
        # Start dashboard
        dashboard = TradingDashboard(self)
        dashboard.run()
        
        # Keep main thread alive
        while True:
            time.sleep(3600)

    def initialize_models(self):
        """Initialize AI models for all symbols"""
        logger.info("Initializing AI models...")
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                predictor = MarketPredictor(symbol, timeframe)
                logger.info(f"Initialized predictor for {symbol} {timeframe}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("trading_system.log"),
            logging.StreamHandler()
        ]
    )
    
    # Start the system
    system = PocketOptionTradingSystem()
    system.run()
