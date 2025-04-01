import pandas as pd
import numpy as np
# TODO: Make it starting from a subfolder (PYTHONPATH?) and do not use relative import
from config import Config
from gemini_model.gemini import StockSuggester
import logging

# ========================
# 🔹 Sample model training and predicting
# ========================
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s %(message)s')

class ModelTests:

    @classmethod
    def proceed(cls):
        # Create instance (will load data if exists)
        suggester = StockSuggester(n_predict_steps=5)

        # === Example Learning Phase ===
        # Simulate data with volume and a dummy moving average
        learning_events = []
        base_time = pd.Timestamp.now() - pd.Timedelta(days=20) # Need more days for indicators
        stocks = {'STOCK_A': 100, 'STOCK_B': 200, 'STOCK_C': 50}
        trends = {'STOCK_A': 0.0008, 'STOCK_B': -0.0006, 'STOCK_C': 0.0001}
        vols = {'STOCK_A': 10000, 'STOCK_B': 50000, 'STOCK_C': 5000}
        noise = {'STOCK_A': 0.015, 'STOCK_B': 0.020, 'STOCK_C': 0.010}

        all_stock_data = {}
        column_names = ['time', 'stock', 'price', 'volume', 'moving_average'] # Define column names

        for stock, price in stocks.items():
            stock_prices = []
            stock_times = []
            current_stock_learning_events = []
            for i in range(300): # Generate more data points
                time = base_time + pd.Timedelta(hours=i)
                price = price * (1 + np.random.normal(trends[stock], noise[stock]))
                volume = vols[stock] * (1 + np.random.uniform(-0.3, 0.3))
                stock_prices.append(price)
                stock_times.append(time)
                # Simple Moving Average calculation for dummy data (e.g., 10 periods)
                current_ma = np.mean(stock_prices[-10:]) if len(stock_prices) >= 10 else price
                event_tuple = (time, stock, price, max(0, volume), current_ma)
                learning_events.append(event_tuple) # Ensure volume >= 0
                current_stock_learning_events.append(event_tuple) # Append to list for this stock only

            # Store for prediction example continuity
            temp_df_stock_only = pd.DataFrame(current_stock_learning_events, columns=column_names)
            # Store this stock-specific DataFrame
            # No need to filter temp_df_stock_only further as it only contains data for 'stock'
            all_stock_data[stock] = temp_df_stock_only.copy()


        # Add data for a stock that will likely have insufficient history for prediction later
        price_d = 75
        for i in range(40): # Less than MIN_HISTORY_FOR_PREDICTION
            time = base_time + pd.Timedelta(hours=i)
            price_d = price_d * (1 + np.random.normal(0, 0.01))
            volume = 1000 * (1 + np.random.uniform(-0.3, 0.3))
            ma = price_d # Simplified MA for short history
            learning_events.append((time, 'STOCK_D', price_d, max(0, volume), ma))

        # Call the learn method
        suggester.learn(learning_events)

        # === Example Prediction Phase ===
        prediction_events = []
        current_time = pd.Timestamp.now()

        for stock in ['STOCK_A', 'STOCK_B', 'STOCK_C']: # Predict for learned stocks
            try:
                # Get last known state to continue simulation realistically
                last_row = suggester.history[suggester.history['stock'] == stock].iloc[-1]
                last_price = last_row['price']
                last_vol = last_row['volume']
                # Use the actual history for MA calculation base
                hist_prices = suggester.history[suggester.history['stock'] == stock]['price'].tolist()

                for i in range(5): # Add a few recent points
                    time = current_time + pd.Timedelta(minutes=i * 10)
                    # Simulate price change similar to its trend
                    last_price *= (1 + np.random.normal(trends[stock]*0.5, noise[stock]*0.5))
                    # Simulate volume change
                    last_vol *= (1 + np.random.uniform(-0.1, 0.1))
                    # Calculate MA based on updated history + new points
                    current_hist_prices = hist_prices + [last_price]
                    current_ma = np.mean(current_hist_prices[-10:]) if len(current_hist_prices) >= 10 else last_price
                    hist_prices.append(last_price) # Add to temp history for next step MA calc

                    prediction_events.append((time, stock, last_price, max(0, last_vol), current_ma))

            except IndexError:
                logging.debug(f"\nWarning: Could not retrieve last price for {stock}. Skipping prediction simulation for it.")
                # Add dummy events if needed for testing structure
                # prediction_events.append((current_time, stock, stocks[stock]*1.01, vols[stock], stocks[stock]*1.01))


        # Add event for the stock with known insufficient history
        try:
            last_price_d = suggester.history[suggester.history['stock'] == 'STOCK_D']['price'].iloc[-1]
            prediction_events.append((current_time, 'STOCK_D', last_price_d * 1.01, 1100, last_price_d * 1.01))
        except IndexError:
            prediction_events.append((current_time, 'STOCK_D', 76, 1100, 76))

        # Add event for a completely new stock (model hasn't seen it)
        prediction_events.append((current_time, 'STOCK_NEW', 500, 5000, 500))


        # Call the predict method
        predictions = suggester.predict(prediction_events)

        logging.debug("\n💰 Final Suggestions:")
        # Use logging.debug for better dictionary formatting if many stocks
        import logging
        logging.debug(predictions)

        # Expected outcome hints:
        # STOCK_A: Likely Buy/Strong Buy (upward trend simulated)
        # STOCK_B: Likely Sell/Strong Sell (downward trend simulated)
        # STOCK_C: Likely Hold (slight trend, depends on recent noise)
        # STOCK_D: insufficient history (had only 40 records, likely not enough after indicator NaNs + LOOKBACK_PERIOD)
        # STOCK_NEW: insufficient history (no scaler exists)

