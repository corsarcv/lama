import csv
from datetime import datetime, timedelta
from alpaca_api.api import AlpacaAPI
from gemini_model.gemini import StockSuggester

# We have 2 modes: learning and predicting
L = "Learning"
P = "Prediction"
MODE = P

SOURCE = './data/symbols/sp500.csv'

ALPACA_API = AlpacaAPI(historical_batch_size=10000)

def build_stocks_map():
    with open(SOURCE, mode='r') as file:
        reader = csv.DictReader(file)
        stock_data = [row for row in reader]
    grouped_stocks_data = {}
    for row in stock_data:
        symbol = row['Symbol']
        sector = row['Sector'] 
        if sector not in grouped_stocks_data:
            grouped_stocks_data[sector] = [symbol]
        else:
            grouped_stocks_data[sector].append(symbol) 
    return grouped_stocks_data

grouped_stocks_data = build_stocks_map()

if MODE == L:
    # For Learning
    single_stock = None  # 'COST'

  

    year_back_date = datetime.now() - timedelta(days=365)

    for sector, symbols in grouped_stocks_data.items():
        print(f'🔹 Processing {symbols} {sector}')
        if single_stock:
            sector = single_stock
        suggester = StockSuggester(sector=sector, n_predict_steps=20)
        events = []
        for symbol in symbols:
            if single_stock and symbol != single_stock:
                continue
            last_timestamp = suggester.get_last_history_timestamp_for_stock(symbol)
            if last_timestamp:
                print(f'Model already has data for {symbol} as of {last_timestamp}. Skipping...')
                continue
        
            start_date = last_timestamp if last_timestamp is not None else year_back_date
            ALPACA_API.fetch_historical_data(ticker=symbol, period='15Min', start=start_date.strftime('%Y-%m-%d'))
            df = ALPACA_API.historical_data[symbol]
            list_of_dicts = df.to_dict(orient='records')
            if len(list_of_dicts) == 0:
                print(f'No data for symbol {symbol}. Skipping...')
                continue
            else:
                print(f'Found {len(list_of_dicts)} events for {symbol}')
            
            events.extend([
                dict(time=r['timestamp'], stock=symbol, price=r['close'],
                    volume=r['volume'], moving_average=r['moving_average']) for r in list_of_dicts])
        # ['time', 'stock', 'price', 'volume', 'moving_average']
        if len(events) > 0:
            suggester.learn(events)
        else:
            print(f'No events for {sector}. Movng to the next group')
        print(f'🔹 Done processing {sector}\n')
    # Define list of stocks/sectors
    # Need to learch each sector separatelly (though multiple stocks are allowed for seme predictor)
    # Set the start date (~12 months back)
    # Check if we have existing date for the stock in the model and if so, start with that date
    # Start getting historical data in batches (1,000 ?) with 1h interval separately for each stock, 
    # approx 260 days * 6.5 h ~= 1,800 records per stock
    # Format data before sending
    # Feed each batch to predictor
    # stop getting stock data when reaching current time (or when result count less than batch size)
    # We can train model for different numbers of predict steps (1, 5, 20). 
    # We can get price date from history if it is already there for a different number of steps model

else:
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