import csv
from datetime import datetime, timedelta
from alpaca_api.api import AlpacaAPI
from gemini_model.gemini import StockSuggester
from utils.common import build_stocks_map


ALPACA_API = AlpacaAPI(historical_batch_size=10000)

grouped_stocks_data = build_stocks_map()

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

