from datetime import datetime, timedelta
from alpaca_api.api import AlpacaAPI
from config import Config
from gemini_model.gemini import StockSuggester
from utils.common import DATA_FOLDER, build_stocks_map, play_failure, play_success
import logging
import statistics
import os
import csv

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s %(message)s')

"""
This is prediction similation based on the latest 60 historical prices for stocks below.
"""

watchlist_file_name = os.path.join(DATA_FOLDER, 'watchlist.csv')
with open(watchlist_file_name, newline='') as watchlist:
    reader = csv.reader(watchlist)
    next(reader)  # Skip the header
    stock_symbols =[row[0] for row in reader if row]
print(stock_symbols)

grouped_stocks_data = build_stocks_map()

alpaca_api = AlpacaAPI(historical_batch_size=65)
three_days_ago_date = (datetime.now() - timedelta(days=3)).date()
market_prediction_pct = []
for symbol in stock_symbols:
    alpaca_api.fetch_historical_data(ticker=symbol, period='15Min', start=three_days_ago_date)
    events = []
    for hst in alpaca_api.historical_data[symbol].to_dict('records'):
        events.append({ 'time': hst['timestamp'], 'stock': symbol, 'price': hst['close'], 
            'volume': hst['volume'], 'moving_average': hst['moving_average'] })
    for sector, symbols in grouped_stocks_data.items():
        if symbol in symbols:
            sector = sector
            break
    else:
        sector = 'Unknown'

    prediction = StockSuggester(sector=sector).predict(events)[symbol]
    
    action = prediction['action']
    if action in ('buy' , 'strong_buy'):
        play_success()
    elif  action in ('sell' , 'strong_sell'):
        play_failure()
    if action != 'hold':
        logging.info(f'  🔹 Current Position: {alpaca_api.get_position(symbol)}')
    market_prediction_pct.append(prediction['percentage_change'])
logging.info(f'⚖️ Averge market prediction: {statistics.mean(market_prediction_pct):.4f}%')

