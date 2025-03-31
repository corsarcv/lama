from datetime import datetime, timedelta
from alpaca_api.api import AlpacaAPI
from gemini_model.gemini import StockSuggester
from utils.common import build_stocks_map


grouped_stocks_data = build_stocks_map()
stock_symbols = ["AAPL", "MSFT", "GOOGL", "TSLA"] # Example stocks


alpaca_api = AlpacaAPI(historical_batch_size=65)
three_days_ago_date = (datetime.now() - timedelta(days=3)).date()
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
    prediction = StockSuggester(sector=sector).predict(events)
    print('Prediction:', prediction)
    # stock = 'COST'
    # sector = 'Consumer Staples'
    # price_file = f'./data/model_learning/gemini/trade_history_6f_{sector}_5.csv'
    # with open(price_file, mode='r') as file:
    #     reader = csv.DictReader(file)
    #     data = [row for row in reader if row['stock']==stock]
    # while len(data) >=65:
    #     events = data[:60]
    #     suggester = StockSuggester(sector=sector, n_predict_steps=5)
    #     predicted = suggester.predict(events)[stock]
    #     actual = data[64]['price']
    #     print(actual, predicted)
    #     data = data[60:]
    # Get stocks grouped by sector
    # Iterate by sectors. Each sector needs a separate predictor object
    # For each stock in a specific sector group prepare data:
    # Need to make sure that we have at least LOOKBACK_PERIOD (60 by default) historical events
    # We should keep going back in history until we reach latest date for this stck the model was learned at
    # Get current price and add it to historical events. 
    # We need to make sure that differene between last historical and latest price event not less than 1h
    # React if there is 'buy_strong'/'sell strong' prediction (log and sound)
    # Me wight need to add recent events for the model to learn. It can be done at the beginning, 
    # but at some point it can become too slow, so we better do it at the end