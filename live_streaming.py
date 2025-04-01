import asyncio
import logging
import pandas as pd 

from collections import defaultdict, deque
from typing import List, Dict, Any, Deque
from alpaca_trade_api.stream import Stream
from alpaca_trade_api.common import URL
from alpaca_api.api import ALPACA_API_KEY as API_KEY
from alpaca_api.api import ALPACA_SECRET_KEY as SECRET_KEY
from alpaca_api.api import BASE_URL
from alpaca_api.api import AlpacaAPI

from config import Config
from gemini_model.gemini import StockSuggester
from utils.common import THREE_DAYS_AGO_DATE, get_sector_for_sp500_ticker, load_watchlist, play_failure, play_success
from utils.enums import Action

# ========================
# 🔹 Logging Setup
# ========================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ========================
# 🔹 Configuration
# ========================
# Configure which data feed to use ('sip', 'iex', 'otaa')
# 'sip' generally requires a paid subscription for real-time minute bars
DATA_FEED = 'iex' 

# Use defaultdict to easily manage events queue for stocks
# Price is tickeing each minute, though we will be processing only configured intervals (15 min by default)
# We will process event if either queue becomes full or when the oldest event timestamp is older or equal to the interval
live_events_history: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=Config.TIME_INTERVAL_MINS))
historical_events: Dict[str, Dict] = {}
predictions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
alpaca_api = AlpacaAPI(historical_batch_size=60)

# 💾 ⏳ ✅ 📥 💰 💵 ⚖️ 🔻 🚀 📊 ⚠️ 🔹

# ========================
# 🔹 Processing of live pricing event
# ========================
def process_stock_data(live_event: Dict[str, Any]):
    """
    This function receives the processed stock data.
    Implement your logic here (e.g., store in DB, trigger alerts, etc.).
    """
    symbol = live_event['stock']
    logging.info(f"Received data for processing for {symbol}")
    historical_events[symbol].append(live_event)
    sector = get_sector_for_sp500_ticker(live_event['stock'])
    prediction = StockSuggester(sector=sector).predict(historical_events[symbol])
    logging.debug(prediction)

# ========================
# 🔹 In case if model suggest to buy/sell, 
# add the suggestion to the list to analyze later
# ========================
def post_process_suggestion(prediction, symbol):
    action = prediction['action']
    if action in (Action.STRONG_BUY, Action.BUY):
        # alpaca_api.submit_order(symbol, 1,n.BUY, Action.STRONG_BUY):
        play_success()
    elif  action in (Action.SELL, Action.STRONG_SELL):
        play_failure()
    if action != Action.HOLD:
        logging.info(f'  🔹 Current Position: {alpaca_api.get_position(symbol)}')
        from datetime import datetime, timedelta
        # 5 is a number of future interls for a prediction, need to be changed to a config,
        # as we are gonna also run 30 intervals and 1 interval (N_PREDICT_STEPS)
        prediction['target_date'] =  datetime.now() + timedelta(minutes=Config.TIME_INTERVAL_MINS * 5 )  
        predictions[symbol].append(prediction)


# ========================
# 🔹 Core Logic, async handling of live events
# ========================
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

        # Prepare output dictionary
        live_event_data = {
            "time": timestamp_iso,
            "stock": symbol,
            "price": close_price,
            "volume": volume,
            "moving_average": moving_average # Will be None if not enough data yet
        }

        should_process = should_process_event(symbol, timestamp_iso)
        live_events_history.append(live_event_data)
        
        if should_process:
            # Call the downstream processing function with a list containing this single event
            process_stock_data(live_event_data)
        else:
            logging.debug('Skipping event processing')

        check_due_predictions(symbol, live_event_data)

    except Exception as e:
        logging.error(f"⚠️ Error processing bar for {getattr(bar, 'symbol', 'N/A')}: {e}", exc_info=True)


# ========================
# 🔹 Only add event if the queue full or empty (first request)
#  or oldest event is greater then the time interval)
# ========================   
def should_process_event(symbol, timestamp_iso):
    price_history_for_symbol = live_events_history[symbol]        
    if len(price_history_for_symbol) == Config.TIME_INTERVAL_MINS:
        logging.debug(f"Queue for {symbol} is full. Cleaning the queue and processing event.")
        live_events_history[symbol] = deque(maxlen=Config.TIME_INTERVAL_MINS)
        should_process = True
    elif len(price_history_for_symbol) == 0:
        logging.debug(f"Queue for {symbol} is empty. Processing event.")
        should_process = True            
    else:
        oldest_event_timestamp = pd.Timestamp(price_history_for_symbol[0]['time'])
        current_timestamp = pd.Timestamp(timestamp_iso)
        time_diff = current_timestamp - oldest_event_timestamp
        if time_diff >= pd.Timedelta(minutes=Config.TIME_INTERVAL_MINS):
            logging.debug(f"Oldest event for {symbol} is older than {Config.TIME_INTERVAL_MINS} minutes. Processing events.")
            should_process = True
        else:
            should_process = False
    return should_process

# ========================
# 🔹 Check accuracy of old predictions becoming due
# ========================   
def check_due_predictions(symbol, live_event_data):
    # Check predictions and if we have a predition for the stock passed due (target time < now), then remove that preditction from the list
    now = pd.Timestamp.now()
    for symbol, pred_list in predictions.items():
        for pred in pred_list[:]:  # Iterate over a copy to allow removal
            if pred['target_date'] < now:
                logging.info(f"Prediction for {symbol} became due with a price {live_event_data['price']}: {pred}")
                predictions[symbol].remove(pred)

# ========================
# 🔹 Main entry point: subscribing and registering a callback for live price events
# ========================
async def subscribe_and_process_bars():
    """
    Subscribes to Alpaca's real-time bar stream for the given symbols
    and processes incoming bars using the handle_bar callback.
    """

    if not API_KEY or not SECRET_KEY:
        logging.error(" ⚠️ API Key ID or Secret Key configuration not found")
        return
    
    # List of stocks to monitor
    stock_symbols = load_watchlist()
    logging.info(f"Monitoring {len} stocks: {', '.join(stock_symbols)}")

    logging.info(f'Getting recent history for stocks {stock_symbols}')
    for symbol in stock_symbols:
        historical_events[symbol] = alpaca_api.fetch_historical_data_as_events(ticker=symbol, period='15Min', start=THREE_DAYS_AGO_DATE)

    logging.info(f"Attempting to connect to Alpaca Stream API feed {DATA_FEED} at {BASE_URL} for symbols: {stock_symbols}")

    # Instantiate Stream
    # Note: use_raw_data=False ensures we get SDK objects (like Bar) instead of raw bytes
    # TODO: Move to AlpacaAPI class
    stream = Stream(API_KEY,
                    SECRET_KEY,
                    base_url=URL(BASE_URL),
                    data_feed=DATA_FEED,
                    raw_data=False) # Process SDK objects

    # Register the handler for bars for the specified symbols
    # The '*' unpacks the list of symbols
    stream.subscribe_bars(handle_bar, *stock_symbols)

    logging.info("✅ Subscription successful. Starting stream listener...")
    try:
        # Run the stream connection - this will block until disconnected or stopped
        await stream._run_forever()
    except KeyboardInterrupt:
        logging.info("⚠️ Stream stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logging.error(f"⚠️ An error occurred during streaming: {e}", exc_info=True)
    finally:
        logging.info("⏳ Closing Alpaca stream connection...")
        await stream.stop_ws()
        logging.info("✅ Stream closed.")


# ========================
# 🔹 Starting main app loop
# ========================
if __name__ == "__main__":

    logging.info("⏳ Starting Alpaca data stream subscription. Press Ctrl+C to stop...")

    # Run the main async function using asyncio's event loop
    try:
        asyncio.run(subscribe_and_process_bars())
    except KeyboardInterrupt:
        logging.warning("\n⏳  Exiting application.")
        exit(0)
    except Exception as e:
        logging.error(f"\n⚠️  Application exited with an error: {e}")