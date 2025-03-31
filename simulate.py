from datetime import datetime, timedelta
from alpaca_api.api import AlpacaAPI
from config import Config
from gemini_model.gemini import StockSuggester
from utils.common import build_stocks_map, play_failure, play_success
import logging
import statistics

logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s %(message)s')

"""
This is prediction similation based on the latest 60 historical prices for stocks below.
"""

stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"] # Example stocks
stock_symbols = stock_symbols + ['ERIE', 'STLD', 'MKTX', 'BX', 'NUE', 'MOS', 'BLK', 'COF', 'KKR']
stock_symbols = stock_symbols + ['WRB', 'PRU', 'KEY', 'CTVA', 'MTB', 'IP','CF', 'VMC', 'EMN', 'SHW', 'FCX',
                                 'HBAN', 'RF', 'NDAQ', 'AMCR', 'IFF', 'FITB', 'PKG', 'GS', 'MLM',
                                 'USB', 'ACGL', 'BALL', 'C', 'CME', 'CFG', 'BEN', 'LYB', 'IVZ', 'AVY' ]

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
        # TODO: Check if there are holdings for the symbol
        play_failure()
    market_prediction_pct.append(prediction['percentage_change'])
logging.info(f'⚖️ Averge market prediction: {statistics.mean(market_prediction_pct):.4f}%')

