import csv
from datetime import datetime, timedelta
from alpaca_api.api import AlpacaAPI
from gemini_model.gemini import StockSuggester

# We have 2 modes: learning and predicting
L = "Learning"
P = "Prediction"
MODE = L

SOURCE = './data/symbols/sp500.csv'

API = AlpacaAPI(historical_batch_size=1000)

if MODE == L:
    # For Learning
    with open(SOURCE, mode='r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    for row in data:
        symbol = row['Symbol']
        sector = row['Sector']
        print(f'🔹 Processing {symbol} {sector}')
        suggester = StockSuggester(sector=sector)
        last_timestamp = suggester.get_last_history_timestamp_for_stock(symbol)
        year_back_date = datetime.now() - timedelta(days=365)
        start_date = last_timestamp if last_timestamp is not None else year_back_date
        API.fetch_historical_data(ticker=symbol)
        print(f'🔹 Done processing {symbol} {sector}\n')
    # Define list of stocks/sectors
    # Need to learch each sector separatelly (though multiple stocks are allowed for seme predictor)
    # Set the start date (~12 months back)
    # Check if we have existing date for the stock in the model and if so, start with that date
    # Start getting historical data in batches (1,000 ?) with 1h interval separately for each stock, 
    # approx 260 days * 6.5 h ~= 1,800 records per stock
    # Format data before sending
    # Feed each batch to predictor
    # stop getting stock data when reaching current time (or when result count less than batch size)

else:
    # For prediction
    pass
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