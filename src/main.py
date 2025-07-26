import time
import schedule
from datetime import datetime
from src.data_processing.data_fetcher import DataFetcher
from src.data_processing.feature_engineer import FeatureEngineer
from src.trading_signals.po_signal import POSignalGenerator
from src.trading_signals.po_ai_signal import POAISignalGenerator
from src.ai_engine.retrain_scheduler import RetrainScheduler
from src.visualization.plot_signals import SignalVisualizer
from config import settings

class TradingSystem:
    def __init__(self):
        self.symbols = settings.SYMBOLS
        self.timeframes = settings.TIMEFRAMES
        self.signal_generators = {}
        self.initialize_generators()
        
    def initialize_generators(self):
        for symbol in self.symbols:
            for tf in self.timeframes:
                self.signal_generators[(symbol, tf)] = {
                    'PO': POSignalGenerator(symbol, tf),
                    'PO_AI': POAISignalGenerator(symbol, tf)
                }
    
    def fetch_and_process(self, symbol, timeframe):
        fetcher = DataFetcher(symbol, timeframe)
        df = fetcher.fetch_historical(limit=1000)
        df = FeatureEngineer.add_features(df)
        return df
    
    def generate_signals(self):
        all_signals = {'PO': {}, 'PO_AI': {}}
        
        for (symbol, tf), generators in self.signal_generators.items():
            df = self.fetch_and_process(symbol, tf)
            
            # Generate signals
            po_signal = generators['PO'].generate_signal(df)
            po_ai_signal = generators['PO_AI'].generate_signal(df)
            
            if po_signal:
                all_signals['PO'].setdefault(symbol, []).append((tf, po_signal))
            if po_ai_signal:
                all_signals['PO_AI'].setdefault(symbol, []).append((tf, po_ai_signal))
        
        return all_signals
    
    def run(self):
        print("Starting Trading System...")
        schedule.every(5).minutes.do(self.execute_trading_cycle)
        RetrainScheduler().schedule_retraining()
        
        while True:
            schedule.run_pending()
            time.sleep(1)
    
    def execute_trading_cycle(self):
        print(f"\n{datetime.now()}: Executing trading cycle")
        signals = self.generate_signals()
        self.process_signals(signals)
        print("Trading cycle completed")
    
    def process_signals(self, signals):
        # Placeholder for trading execution logic
        # This would connect to Pocket Option API in production
        print("\nPO Signals:")
        for symbol, signal_list in signals['PO'].items():
            for tf, signal in signal_list:
                print(f"{symbol} {tf}: {signal[0]} at {signal[1]} (Confidence: {signal[3]:.2%})")
        
        print("\nPO AI Signals:")
        for symbol, signal_list in signals['PO_AI'].items():
            for tf, signal in signal_list:
                print(f"{symbol} {tf}: {signal[0]} at {signal[1]} (Confidence: {signal[3]:.2%})")
        
        # Visualize signals
        for symbol in self.symbols:
            SignalVisualizer.visualize(symbol, signals, self.timeframes[0])

if __name__ == "__main__":
    system = TradingSystem()
    system.run()
