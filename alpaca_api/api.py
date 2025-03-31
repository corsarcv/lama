from config import Config
from utils.constants import APCA_API_KEY_ID, APCA_API_SECRET_KEY, BASE_URL
import alpaca_trade_api as tradeapi
import datetime
import pandas as pd
import random

# ========================
# 🔹 CONFIGURATION SECTION
# ========================

# Alpaca API Keys (Replace with your actual keys)
ALPACA_API_KEY = Config[APCA_API_KEY_ID]  # Add your Alpaca API Key
ALPACA_SECRET_KEY = Config[APCA_API_SECRET_KEY]  # Add your Alpaca Secret Key

BASE_URL = Config.BASE_URL  # Use live trading by changing to live endpoint

ALPACA_ENABLED = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)

# Connect to Alpaca API
if ALPACA_ENABLED:
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

class AlpacaAPI:

    def __init__(self, cash=10000, historical_batch_size=100):
        self.cash = cash
        self.historical_data = {}
        self.historical_batch_size = historical_batch_size
        

    # ========================
    # 🔹 EXECUTING LIVE ORDER
    # ========================
    def submit_order(self, symbol, qty, side, type='market', time_in_force='gtc'):
        api.submit_order(
            symbol=symbol, qty=qty, side=side, type=type, time_in_force=time_in_force)

    # ========================
    # 🔹 ALPACA ACCOUNT CHECK
    # ========================
    def check_alpaca_account(self):
        """Fetch account balance from Alpaca API."""
        if not ALPACA_ENABLED:
            return
        account = api.get_account()
        print(f"💰 Alpaca Account Balance: ${account.cash}")
        self.cash = float(account.cash())
        return float(self.cash)


    # ==========================
    # 🔹 LIVE MARKET DATA FETCH
    # ==========================

    def fetch_live_price(self, ticker):
        """Fetch real-time stock price from Alpaca."""
        if not ALPACA_ENABLED:
            return random.uniform(100, 200)  # Simulated live price
        try:
            latest_bar = api.get_latest_bar(ticker)
            if not latest_bar:
                return None

            latest_data = {
                "timestamp": latest_bar.t,
                "open": latest_bar.o,
                "high": latest_bar.h,
                "low": latest_bar.l,
                "close": latest_bar.c,
                "volume": latest_bar.v
            }

            # Update historical prices dataset
            latest_df = pd.DataFrame([latest_data]) 
            self.historical_data[ticker] = pd.concat([self.historical_data[ticker], latest_df], ignore_index=True)
            self.perform_timestamp_calculations_on_historical_prices(self.historical_data[ticker])

            return latest_bar.c
        except Exception as ex:
            print("⚠️ Error on fetching historical data:", ex)
            return None

    def perform_timestamp_calculations_on_historical_prices(self, df):
            # Convert timestamps to datetime format and adjust to EST
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York')

            # Extract time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['minutes_since_open'] = (df['hour'] - 9) * 60 + df['minute']  # Market opens at 9:30 AM EST

    # ==========================
    # 🔹 GET HOLDING FOR SYMBOL
    # ==========================
    def get_position(self, symbol):
        try:
            positions = api.list_positions()

            for position in positions:
                if position.symbol.upper() == symbol.upper():
                    return float(position.qty)
            return None

        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            return False

    # ==========================
    # 🔹 HISTORICAL DATA FETCHING & HANDLING
    # ==========================
    def fetch_historical_data(self, ticker, period='1Min', start=None, end=None):
        """Fetch historical stock data once per ticker."""
        if ALPACA_ENABLED:
            bars = api.get_bars(
                ticker, period, limit=self.historical_batch_size, sort='desc',
                start=start, end=end)
            
            bars.df.reset_index()  # Reset index to move timestamp from index to a column
            
            df = pd.DataFrame({
                'timestamp': [bar.t for bar in bars],  # UTC Timestamps from Alpaca
                'open': [bar.o for bar in bars],
                'high': [bar.h for bar in bars],
                'low': [bar.l for bar in bars],
                'close': [bar.c for bar in bars],
                'volume': [bar.v for bar in bars],
                'moving_average': [bar.vw for bar in bars]
            })
            # Sorting by timestamp  because it is in desc order now
            df = df.sort_values(by='timestamp', ascending=True) 
            df = df.reset_index(drop=True)
            
            self.perform_timestamp_calculations_on_historical_prices(df)
            self.historical_data[ticker] = df
        else:
            data = []
            base_price = random.uniform(100, 200)
            for _ in range(100):
                price = base_price * random.uniform(0.98, 1.02)
                data.append({'open': price, 'high': price * 1.01, 'low': price * 0.99, 
                             'close': price, 'volume': random.randint(1000, 5000),
                             'timestamp': datetime.utcnow()})
                             
            df = pd.DataFrame(data)
            df['minutes_since_open'] = df.index
            self.historical_data[ticker] = df

