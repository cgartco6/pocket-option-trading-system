from binance import Client
import pandas as pd
import time
from config import security

class DataFetcher:
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        credentials = security.secure_credentials()
        self.client = Client(
            api_key=credentials['BINANCE_API_KEY'],
            api_secret=credentials['BINANCE_SECRET']
        )
    
    def fetch_historical(self, limit=1000):
        klines = self.client.get_klines(
            symbol=self.symbol,
            interval=self.timeframe,
            limit=limit
        )
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].astype(float)
    
    def stream_live_data(self, callback, interval=60):
        from binance.websockets import BinanceSocketManager
        bm = BinanceSocketManager(self.client)
        conn_key = bm.start_kline_socket(self.symbol, callback, interval=self.timeframe)
        bm.start()
        return conn_key
