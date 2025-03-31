import asyncio
import logging
from collections import defaultdict, deque
from typing import List, Dict, Any, Deque
import pandas as pd  # For easy timestamp conversion

# Use alpaca_trade_api for streaming V2 data
import alpaca_trade_api as tradeapi
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL
from alpaca_api.api import ALPACA_API_KEY as API_KEY
from alpaca_api.api import ALPACA_SECRET_KEY as SECRET_KEY
from alpaca_api.api import BASE_URL
from alpaca_api.api import AlpacaAPI
from gemini_model.gemini import StockSuggester

# --- Configuration ---
# Load API keys from environment variables for security
# Set these in your environment:
# export APCA_API_KEY_ID='YOUR_KEY_ID'
# export APCA_API_SECRET_KEY='YOUR_SECRET_KEY'
# export APCA_API_BASE_URL='https://paper-api.alpaca.markets' # Or live URL
# API_KEY = # os.getenv("APCA_API_KEY_ID")
# SECRET_KEY = os.getenv("APCA_API_SECRET_KEY")
# BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets") # Default to paper

# Configure which data feed to use ('sip', 'iex', 'otaa')
# 'sip' generally requires a paid subscription for real-time minute bars
DATA_FEED = 'iex' 

# Moving Average Configuration
MA_PERIOD = 5 # Calculate a 5-period simple moving average

# --- Global Data Structure ---
# Use defaultdict to easily manage price history per stock
# Store recent closing prices using a deque with a max length
price_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=MA_PERIOD))

alpaca_api = AlpacaAPI(historical_batch_size=60)

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Placeholder Function ---
def process_stock_data(data: List[Dict[str, Any]]):
    """
    This function receives the processed stock data.
    Implement your logic here (e.g., store in DB, trigger alerts, etc.).
    """
    print("-" * 30)
    print(f"Received data for processing:")
    events = []
    for item in data:
        events = collect_events_history(item['stock'])
        events.append(item)
        sector = 'Information Technology'
        prediction = StockSuggester(sector=sector).predict(events)
        logging.info(prediction)

    # {'time': '2025-03-31T19:57:00+00:00', 'stock': 'MSFT', 
    # 'price': 375.625, 'volume': 12110, 'moving_average': 375.52489}
    # {'timestamp': Timestamp('2025-03-31 04:00:00-0400', tz='America/New_York'), 
    # 'open': 373.94, 'high': 376.41, 'low': 373.94, 'close': 375.01, 'volume': 16432, 'moving_average': 375.359104, 'hour': 4, 'minute': 0, 'minutes_since_open': -300}
    #time,stock,price,volume,moving_average
    print("-" * 30)
    
def collect_events_history(stock):
    events = []
    for hst in alpaca_api.historical_data[stock].to_dict('records'):
        events.append({ 'time': hst['timestamp'], 'stock': stock, 'price': hst['close'], 
            'volume': hst['volume'], 'moving_average': hst['moving_average'] })
    return events

# --- Core Logic ---

async def handle_bar(bar):
    """
    Async callback function to handle incoming bar data from the stream.
    Parses the bar, calculates the moving average, formats the output,
    and calls the processing function.
    """
    try:
        symbol = bar.symbol
        close_price = bar.close
        volume = bar.volume
        moving_average = bar.vwap
        # Alpaca timestamps are typically nanoseconds POSIX epoch. Convert to ISO string.
        timestamp_dt = pd.Timestamp(bar.timestamp, unit='ns', tz='UTC')
        timestamp_iso = timestamp_dt.isoformat()

        logging.debug(f"Raw bar received: {symbol} @ {timestamp_iso} - Price: {close_price}, Vol: {volume}")

        # Update price history for the specific stock
        history = price_history[symbol]
        history.append(close_price)

        # # Calculate Moving Average
        # moving_average = None
        # if len(history) == MA_PERIOD:
        #     moving_average = sum(history) / MA_PERIOD
        #     # Optional: round the moving average
        #     moving_average = round(moving_average, 4)

        # Prepare output dictionary
        output_data = {
            "time": timestamp_iso,
            "stock": symbol,
            "price": close_price,
            "volume": volume,
            "moving_average": moving_average # Will be None if not enough data yet
        }

        # Call the downstream processing function with a list containing this single event
        # (as requested by the prompt: "sending the list of dicts")
        process_stock_data([output_data])

    except Exception as e:
        logging.error(f"Error processing bar for {getattr(bar, 'symbol', 'N/A')}: {e}", exc_info=True)


async def subscribe_and_process_bars(symbols: List[str]):
    """
    Subscribes to Alpaca's real-time bar stream for the given symbols
    and processes incoming bars using the handle_bar callback.
    """
    if not API_KEY or not SECRET_KEY:
        logging.error("API Key ID or Secret Key not found in environment variables.")
        print("Error: Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables.")
        return

    logging.info(f'Getting recent history for stocks {symbols}')
    for symbol in symbols:
        alpaca_api.fetch_historical_data(ticker=symbol, period='15Min')

    logging.info(f"Attempting to connect to Alpaca Stream API at {BASE_URL} for symbols: {symbols}")
    logging.info(f"Using data feed: {DATA_FEED}")
    logging.info(f"Calculating {MA_PERIOD}-period SMA.")

    # Instantiate Stream
    # Note: use_raw_data=False ensures we get SDK objects (like Bar) instead of raw bytes
    stream = Stream(API_KEY,
                    SECRET_KEY,
                    base_url=URL(BASE_URL),
                    data_feed=DATA_FEED,
                    raw_data=False) # Process SDK objects

    # Register the handler for bars for the specified symbols
    # The '*' unpacks the list of symbols
    stream.subscribe_bars(handle_bar, *symbols)

    logging.info("Subscription successful. Starting stream listener...")
    try:
        # Run the stream connection - this will block until disconnected or stopped
        #await stream.run()
        await stream._run_forever()
    except KeyboardInterrupt:
        logging.info("Stream stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logging.error(f"An error occurred during streaming: {e}", exc_info=True)
    finally:
        logging.info("Closing Alpaca stream connection...")
        await stream.close()
        logging.info("Stream closed.")


# --- Example Usage ---
if __name__ == "__main__":
    # List of stocks to monitor
    stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"] # Example stocks

    print("Starting Alpaca data stream subscription...")
    print(f"Monitoring: {', '.join(stock_symbols)}")
    print("Press Ctrl+C to stop.")

    # Run the main async function using asyncio's event loop
    try:
        asyncio.run(subscribe_and_process_bars(stock_symbols))
    except KeyboardInterrupt:
        print("\nExiting application.")
    except Exception as e:
        print(f"\nApplication exited with an error: {e}")