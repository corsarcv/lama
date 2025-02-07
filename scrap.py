import alpaca_trade_api as tradeapi
import pandas as pd
from config import Config
from utils.constants import APCA_API_KEY_ID, APCA_API_SECRET_KEY

# Alpaca API Credentials
ALPACA_API_KEY = Config()[APCA_API_KEY_ID]  # Add your Alpaca API Key
ALPACA_SECRET_KEY = Config()[APCA_API_SECRET_KEY]  # Add your Alpaca Secret Key
BASE_URL = "https://paper-api.alpaca.markets"

# Initialize Alpaca API
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

def fetch_latest_bars(ticker, n_bars=100):
    """Fetch the latest N bars (candlesticks) for a given stock."""
    try:
        bars = api.get_bars(
            ticker,
            timeframe="1Min",  # Adjust for other timeframes: "5Min", "15Min", "1Hour", etc.
            limit=n_bars,
            sort='desc'
        ).df  # Convert to Pandas DataFrame
        
        bars = bars.reset_index()  # Reset index to move timestamp from index to a column
        #bars['timestamp'] = bars['timestamp'].dt.tz_convert('America/New_York')  # Convert to EST
        # Sort DataFrame by 'timestamp' in ascending order
        bars = bars.sort_values(by='timestamp', ascending=True)

        # Reset index after sorting
        bars = bars.reset_index(drop=True)
        
        return bars
    
    except Exception as e:
        print(f"⚠️ Error fetching data for {ticker}: {e}")
        return None

# Example: Fetch the latest 100 1-minute bars for Apple (AAPL)
latest_bars = fetch_latest_bars("AAPL", 10)

# Display Data
if latest_bars is not None:
    print(latest_bars.to_string(index=False))
