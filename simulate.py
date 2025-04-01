import logging
import statistics

from alpaca_api.api import AlpacaAPI
from config import Config
from gemini_model.gemini import StockSuggester
from utils.common import THREE_DAYS_AGO_DATE, build_stocks_map, get_sector_for_sp500_ticker, \
    load_watchlist, play_failure, play_success
from utils.enums import Action

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s %(message)s')

"""
This is prediction similation based on the latest 60 historical prices for stocks below.
"""

stock_symbols = load_watchlist()
grouped_stocks_data = build_stocks_map()
alpaca_api = AlpacaAPI(historical_batch_size=65)

market_prediction_pct = []

def post_process_suggestion(prediction):
    action = prediction['action']
    if action in (Action.STRONG_BUY, Action.BUY):
        # alpaca_api.submit_order(symbol, 1,n.BUY, Action.STRONG_BUY):
        play_success()
    elif  action in (Action.SELL, Action.STRONG_SELL):
        play_failure()
    if action != Action.HOLD:
        logging.info(f'  🔹 Current Position: {alpaca_api.get_position(symbol)}')


for symbol in stock_symbols:
    events = alpaca_api.fetch_historical_data_as_events(ticker=symbol, period='15Min', start=THREE_DAYS_AGO_DATE)
    sector = get_sector_for_sp500_ticker(symbol)
    prediction = StockSuggester(sector=sector).predict(events)[symbol]
    post_process_suggestion(prediction)
    market_prediction_pct.append(prediction['percentage_change'])

logging.info(f'⚖️  Averge market prediction: {statistics.mean(market_prediction_pct):.4f}%')

