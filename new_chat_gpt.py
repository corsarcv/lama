import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import os
import _osx_support
from utils.common import DATA_FOLDER, get_sector_for_sp500_ticker

# --- Sample mini dataset ---
# for stock in ["AAPL", "TSLA", "MSFT", "LYB", "COST"]:
stock = 'AAPL'
sector = get_sector_for_sp500_ticker(stock)
source_file_name = os.path.join(DATA_FOLDER, 'model_learning', 'gemini', f'trade_history_6f_{sector}.csv')

with open(source_file_name, mode='r') as file:
    reader = csv.DictReader(file)
    data = [row for row in reader if row['stock'] == stock]



sample_data = [('220.0', '1516004'), ('219.1575', '1114458'), ('219.29', '947676'), ('219.79', '699957'),
               ('219.6', '803145'), ('219.0', '813963'), ('219.14', '734861'), ('219.29', '603120'),
               ('219.285', '1159853'), ('219.29', '559305'), ('219.175', '629328'), ('219.57', '636693'),
               ('219.35', '620460'), ('219.76', '644413'), ('220.0274', '947956'), ('220.125', '1034688'),
               ('219.79', '945634'), ('219.77', '1210656'), ('219.895', '977983'), ('220.43', '5023376'),
               ('220.73', '10742334'), ('220.7', '372995'), ('220.72', '38577'), ('220.72', '60500'),
               ('220.71', '6937'), ('220.71', '5607'), ('220.73', '3735'), ('220.76', '6225'), ('220.7', '11502'),
               ('220.72', '7963'), ('220.75', '6139'), ('220.71', '3250'), ('220.81', '4799'), ('220.75', '1097'),
               ('220.86', '7421'), ('220.74', '5165'), ('220.63', '5706'), ('220.48', '5448'), ('220.22', '5127'),
               ('220.21', '2350'), ('220.1', '7814'), ('220.15', '14403'), ('220.22', '1776'), ('220.39', '2816'),
               ('220.31', '5442'), ('220.42', '3653'), ('220.59', '1741'), ('220.74', '8313'), ('220.81', '15359'),
               ('220.75', '8216'), ('220.84', '9893'), ('220.84', '2601'), ('220.98', '41824'), ('221.2', '25170'),
               ('221.49', '26185'), ('221.3', '24051'), ('221.43', '70638'), ('220.6', '89088')]
import random

pnl = []
for _ in range(20):
    samples = 250
    index = random.randint(0, len(data)-samples)
    predict = 8
    window_size = 65  # how many past points to use

    #model params
    n_estimators=100
    learning_rate=0.05
    max_depth=5

    sample_data = [(r['price'], r['volume']) for r in data[index: index + samples]]
    actual_prices = [float(r['price']) for r in data[index + samples: index + samples + predict]]

    # Parse into numpy arrays
    prices = np.array([float(p) for p, v in sample_data])
    volumes = np.array([int(v) for p, v in sample_data])

    from sklearn.ensemble import GradientBoostingRegressor

    # Recreate features for a Gradient Boosting model
    # We'll use a sliding window of past prices and volumes as features


    features = []
    targets = []

    for i in range(window_size, len(prices)):
        price_window = prices[i-window_size:i]
        volume_window = volumes[i-window_size:i]
        feature_vector = np.concatenate([price_window, volume_window])
        features.append(feature_vector)
        targets.append(prices[i])

    X = np.array(features)
    y = np.array(targets)

    # Train a Gradient Boosting model
    gb_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3)
    gb_model.fit(X, y)

    # Predict next 8 values
    last_price_window = prices[-window_size:].tolist()
    last_volume_window = volumes[-window_size:].tolist()

    gb_predictions = []

    for _ in range(predict):
        feature_vector = np.concatenate([last_price_window, last_volume_window]).reshape(1, -1)
        next_price = gb_model.predict(feature_vector)[0]
        gb_predictions.append(next_price)

        # Update sliding windows
        last_price_window = last_price_window[1:] + [next_price]
        last_volume_window = last_volume_window[1:] + [last_volume_window[-1]]  # reusing last known volume

    # Reduce the number of record on screen
    prices = prices[-30:]
    current_price = prices[-1]
    predicted_price = [gb_predictions[-1]]
    actual_price = actual_prices[-1]
    amount = 1000
    if predicted_price > current_price:
        marginal = actual_price - current_price
        pct_marginal = marginal/current_price
        print("%", pct_marginal*100)
        pnl.append(pct_marginal)
    # Plot the resultsactual_prices
    plt.figure(figsize=(10, 5))
    plt.plot(prices, label="Historical Price")
    plt.plot(range(len(prices), len(prices) + predict), gb_predictions, label="Predicted Price (GBR)", marker='o', linestyle='--')
    plt.plot(range(len(prices), len(prices) + predict), actual_prices, label="Actual Price", marker='x', linestyle='-')

    plt.xlabel("Time (15-min intervals)")
    plt.ylabel("Price")
    plt.title("Stock Price Forecast with Gradient Boosting Regressor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # plt.show(block=False)

import statistics

print('Sum', sum(pnl))
print('Average', statistics.mean(pnl))
print('Median', statistics.median(pnl))
plt.show()