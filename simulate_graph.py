import os
import csv
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from gemini_model.gemini import StockSuggester
from utils.common import DATA_FOLDER, build_stocks_map, get_sector_for_sp500_ticker, \
    load_watchlist, play_failure, play_success

for stock in ["AAPL", "TSLA", "MSFT", "LYB", "COST"]:
    sector = get_sector_for_sp500_ticker(stock)
    source_file_name = os.path.join(DATA_FOLDER, 'model_learning', 'gemini', f'trade_history_6f_{sector}.csv')

    with open(source_file_name, mode='r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader if row['stock'] == stock]
    prices = []
    for rec in data:
        prices.append(rec)
        print(rec, ',')
        if len(prices) > 300:
            break
    #print(prices)
    break
"""
    sector = get_sector_for_sp500_ticker(stock)
    source_file_name = os.path.join(DATA_FOLDER, 'model_learning', 'gemini', f'trade_history_6f_{sector}.csv')

    with open(source_file_name, mode='r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader if row['stock'] == stock]
        
    print(f"Set length: {len(data)}")
    start_index = 128
    history = data[:start_index]
    for rec in history:
        rec['time'] = datetime.strptime(rec['time'], "%Y-%m-%d %H:%M:%S%z")+ timedelta(days=730)

    timestamps_list = []
    actual_price = []
    predicted_5_price = [] # [0.0] * 5
    predicted_20_price = [] # [0.0] * 20

    predict5 = StockSuggester(sector=sector, n_predict_steps=5, lookback_period=60)
    predict20 = StockSuggester(sector=sector, n_predict_steps=20, lookback_period=128)

    for index in range(start_index, len(data)):
        record = data[index]
        timestamp =  datetime.strptime(record['time'], "%Y-%m-%d %H:%M:%S%z")+ timedelta(days=730)
        timestamps_list.append(timestamp)
        record['time'] = timestamp
        actual_price.append(float(record['price']), int(record['volume']))

        if index < len(data) -5:
            
            prediction5 = predict5.predict(history)
            predicted_5_price.append(float(prediction5[stock]['predicted_price']))

        if index < len(data) -20:
            
            prediction20 = predict20.predict(history)
            predicted_20_price.append(float(prediction20[stock]['predicted_price']))

        history.append(record)
        index += 1
        print(f"Index {index} out of {len(data)}")

        if index > 500:
            break

    
    # time_objects = [datetime.strptime(ts, "%Y-%m-%d %H:%M") for ts in timestamps_list]
    print([str(d) for d in timestamps_list])
    print(actual_price)
    print(predicted_5_price)
    print(predicted_20_price)
    # Plot all price series
    plt.plot(timestamps_list, actual_price, marker='o', label='Actual Price', color='blue')
    plt.plot(timestamps_list, predicted_5_price[:len(timestamps_list)], marker='s', label='Predicted 5', color='red')
    plt.plot(timestamps_list, predicted_20_price[:len(timestamps_list)], marker='D', label='Predicted 20', color='orange')

    from matplotlib.dates import DateFormatter
    plt.gca().xaxis.set_major_formatter(DateFormatter("%Y-%m-%d %H:%M"))

    y_min = min(actual_price)
    y_max = max(max(actual_price), max(predicted_5_price), max(predicted_20_price))
    plt.ylim(y_min, y_max)

    # Formatting
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title(f"Price prediction for {stock}")
    plt.legend()
    plt.grid(True)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45)

    # Show the plot
    plt.tight_layout()
    plt.show(block=False)
"""