import pandas as pd
import numpy as np
import os
import joblib  # For saving/loading scalers more efficiently than pickle
import pandas_ta as ta # For technical indicators
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input
import warnings

# Suppress TensorFlow logs and warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='pandas_ta')


# --- Constants ---
MODEL_FILENAME = 'stock_lstm_model_6f.keras'
HISTORY_FILENAME = 'stock_trade_history_6f.csv'
SCALERS_FILENAME = 'stock_scalers_6f.joblib'

# Technical Indicator Parameters (adjust as needed)
EMA_PERIOD = 12
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14

# Define the feature columns we will use (after calculations)
# Order matters, especially for identifying the price index later
FEATURE_COLS = [
    'price',
    'volume',
    'moving_average',
    f'EMA_{EMA_PERIOD}',
    f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}', # pandas_ta creates this column name
    f'RSI_{RSI_PERIOD}'
]
PRICE_INDEX = FEATURE_COLS.index('price') # Index of the price column within features
N_FEATURES = len(FEATURE_COLS) # Now 6 features

# Training and Prediction Parameters
# Need slightly more history for indicators to become stable
MIN_RECORDS_FOR_INDICATORS = max(MACD_SLOW, RSI_PERIOD) + 15 # Min data needed before calculating indicators reliably
MIN_TRAIN_RECORDS_PER_STOCK = MIN_RECORDS_FOR_INDICATORS + 100 # Min records overall for training a stock
N_STEPS = 60  # Sequence length (lookback period)
MIN_HISTORY_FOR_PREDICTION = N_STEPS + MIN_RECORDS_FOR_INDICATORS # Need enough history to calculate indicators AND form a sequence

# Suggestion thresholds (remain the same, but interpretation might change with better model)
THRESHOLDS = {
    "strong_buy": 0.03,  # Predict price increase > 3%
    "buy": 0.01,         # Predict price increase > 1% and <= 3%
    "sell": -0.01,       # Predict price decrease < -1% and >= -3%
    "strong_sell": -0.03 # Predict price decrease < -3%
    # Hold is the default between sell and buy thresholds
}

class StockSuggester:
    """
    Provides stock trading suggestions based on historical price data and technical indicators
    using an LSTM model with 6 features.
    """
    def __init__(self, model_path=MODEL_FILENAME, history_path=HISTORY_FILENAME, scalers_path=SCALERS_FILENAME):
        """
        Initializes the StockSuggester, loading existing model, history, and scalers.
        """
        print("Initializing Stock Suggester (6 Features)...")
        self.model_path = model_path
        self.history_path = history_path
        self.scalers_path = scalers_path
        self.model = None
        # Initialize history with new columns
        self.history = pd.DataFrame(columns=['time', 'stock', 'price', 'volume', 'moving_average'])
        self.scalers = {} # Dictionary to store scalers for each stock {'STOCK_NAME': scaler}
        self._load()
        self.is_sufficiently_trained = self.model is not None
        print(f"Model loaded: {self.is_sufficiently_trained}")
        print(f"History records loaded: {len(self.history)}")
        print(f"Using Features: {FEATURE_COLS}")
        print(f"Price Index for Scaling: {PRICE_INDEX}")


    def _load(self):
        """Loads the model, history, and scalers from local files."""
        # Load Keras model
        if os.path.exists(self.model_path):
            try:
                # Provide custom_objects if you use custom layers/activations later
                self.model = load_model(self.model_path) #, custom_objects={...})
                print(f"Model loaded successfully from {self.model_path}")
            except Exception as e:
                print(f"Warning: Could not load model from {self.model_path}. Error: {e}")
                self.model = None
        else:
            print(f"Model file {self.model_path} not found.")
            self.model = None

        # Load history CSV
        if os.path.exists(self.history_path):
            try:
                self.history = pd.read_csv(self.history_path, parse_dates=['time'])
                # Ensure correct types after loading
                for col in ['price', 'volume', 'moving_average']:
                     if col in self.history.columns:
                           self.history[col] = pd.to_numeric(self.history[col], errors='coerce')
                self.history = self.history.sort_values(by='time').reset_index(drop=True)
                # Drop rows where essential numeric data might be missing after load
                self.history.dropna(subset=['price', 'volume', 'moving_average'], inplace=True)
                print(f"History loaded successfully from {self.history_path}")
            except Exception as e:
                print(f"Warning: Could not load history from {self.history_path}. Error: {e}")
                self.history = pd.DataFrame(columns=['time', 'stock', 'price', 'volume', 'moving_average'])
        else:
            print(f"History file {self.history_path} not found.")

        # Load scalers
        if os.path.exists(self.scalers_path):
            try:
                self.scalers = joblib.load(self.scalers_path)
                print(f"Scalers loaded successfully from {self.scalers_path}")
            except Exception as e:
                print(f"Warning: Could not load scalers from {self.scalers_path}. Error: {e}")
                self.scalers = {}
        else:
            print(f"Scalers file {self.scalers_path} not found.")

    def _save_history(self):
        """Saves the current history DataFrame to a CSV file."""
        try:
            self.history.to_csv(self.history_path, index=False)
        except Exception as e:
            print(f"Error saving history to {self.history_path}: {e}")

    def _save_model(self):
        """Saves the current Keras model to a file."""
        if self.model:
            try:
                self.model.save(self.model_path)
                print(f"Model saved to {self.model_path}")
            except Exception as e:
                print(f"Error saving model to {self.model_path}: {e}")
        else:
            print("No model to save.")

    def _save_scalers(self):
        """Saves the scalers dictionary to a file."""
        try:
            joblib.dump(self.scalers, self.scalers_path)
            print(f"Scalers saved to {self.scalers_path}")
        except Exception as e:
            print(f"Error saving scalers to {self.scalers_path}: {e}")

    def _calculate_indicators(self, stock_data):
        """
        Calculates technical indicators (EMA, MACD, RSI) for a single stock's data.
        Assumes stock_data is sorted by time and has 'price', 'volume' columns.
        Returns DataFrame with indicators added, NaNs dropped.
        """
        if stock_data.empty or len(stock_data) < MIN_RECORDS_FOR_INDICATORS:
             # print(f"   -> Not enough data ({len(stock_data)}) to calculate indicators (min {MIN_RECORDS_FOR_INDICATORS}).")
             return pd.DataFrame() # Return empty if not enough data

        df = stock_data.copy()
        df.ta.ema(length=EMA_PERIOD, close='price', append=True)
        # MACD: Use typical defaults or specify fast, slow, signal lengths
        df.ta.macd(fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL, close='price', append=True)
        df.ta.rsi(length=RSI_PERIOD, close='price', append=True)

        # Drop rows with NaN values generated by indicator calculations (mostly at the start)
        df.dropna(inplace=True)

        # Select and potentially reorder columns to match FEATURE_COLS
        # Ensure the calculated columns exist before selecting
        required_calculated_cols = [f'EMA_{EMA_PERIOD}', f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}', f'RSI_{RSI_PERIOD}']
        if not all(col in df.columns for col in required_calculated_cols):
            print(f"Warning: Not all calculated indicator columns found in DataFrame for stock.")
            # Find which are missing:
            missing = [col for col in required_calculated_cols if col not in df.columns]
            print(f"Missing columns: {missing}")
            # Attempt to select available columns from FEATURE_COLS anyway
            available_feature_cols = [col for col in FEATURE_COLS if col in df.columns]
            if not available_feature_cols: return pd.DataFrame() # Cannot proceed
            #print(f"Using available feature columns: {available_feature_cols}") # Debug
            # This case shouldn't happen often with pandas_ta but handles edge cases
            return df[available_feature_cols]

        # Ensure all expected base columns are present too
        if not all(col in df.columns for col in ['price', 'volume', 'moving_average']):
             print("Warning: Base columns missing after indicator calculation.")
             return pd.DataFrame()

        # Select the final features in the desired order
        df = df[FEATURE_COLS]

        return df

    def _prepare_data_for_stock(self, stock_data):
        """
        Calculates indicators and prepares sequences for LSTM training for a single stock.
        Returns X (sequences), y (targets), and the scaler used.
        """
        # 1. Calculate indicators
        stock_data_with_indicators = self._calculate_indicators(stock_data)

        if stock_data_with_indicators.empty or len(stock_data_with_indicators) < N_STEPS + 1:
            # print(f"   -> Not enough data after indicator calculation ({len(stock_data_with_indicators)}) to create sequences (min {N_STEPS + 1}).")
            return None, None, None # Not enough data

        # 2. Select Feature columns and Scale Data
        features = stock_data_with_indicators[FEATURE_COLS].values
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Scale all features together based on the history of this specific stock
        scaled_features = scaler.fit_transform(features)

        # 3. Create sequences
        X, y = [], []
        for i in range(N_STEPS, len(scaled_features)):
            # Input sequence contains all N_FEATURES for N_STEPS
            X.append(scaled_features[i-N_STEPS:i, :])
            # Target 'y' is the *next* scaled price
            y.append(scaled_features[i, PRICE_INDEX]) # Use PRICE_INDEX

        if not X:
             return None, None, scaler # Return scaler even if no sequences

        return np.array(X), np.array(y), scaler

    def learn(self, events):
        """
        Learns from a series of events (including price, volume, MA) and updates the model.

        Args:
            events (list): A list of tuples or lists, where each element is
                           (time, stock_ticker, price, volume, moving_average).
                           'time' should be convertible to datetime.
        """
        print(f"\n--- Starting Learning Process ({len(events)} new events) ---")
        if not events:
            print("No new events provided for learning.")
            return

        # 1. Input Validation and Data Preparation
        try:
            # Expecting 5 columns now
            new_data = pd.DataFrame(events, columns=['time', 'stock', 'price', 'volume', 'moving_average'])
            new_data['time'] = pd.to_datetime(new_data['time'])
            # Convert numeric types, coerce errors to NaN
            for col in ['price', 'volume', 'moving_average']:
                 new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
            # Drop rows where essential inputs are missing
            new_data.dropna(subset=['time', 'stock', 'price', 'volume', 'moving_average'], inplace=True)
            if new_data.empty:
                print("Warning: All input events were invalid or incomplete after validation.")
                return

        except Exception as e:
            print(f"Error processing input events: {e}")
            print("Expected format: list of (time, stock, price, volume, moving_average)")
            return

        # 2. Append to History and Save
        self.history = pd.concat([self.history, new_data], ignore_index=True)
        # Drop based on more columns now to avoid duplicates if needed
        self.history = self.history.drop_duplicates(subset=['time', 'stock']).sort_values(by='time').reset_index(drop=True)
        self._save_history()
        print(f"History updated. Total records: {len(self.history)}")

        # 3. Prepare Data for Training (per stock)
        all_X, all_y = [], []
        updated_scalers = {}
        sufficient_data_for_training = False

        unique_stocks = self.history['stock'].unique()
        print(f"Found {len(unique_stocks)} unique stocks in history.")

        for stock in unique_stocks:
            # Get data for the specific stock, ensure it's sorted
            stock_data = self.history[self.history['stock'] == stock].copy()
            stock_data = stock_data.sort_values(by='time').reset_index(drop=True)

            if len(stock_data) >= MIN_TRAIN_RECORDS_PER_STOCK:
                print(f"Processing data for stock: {stock} ({len(stock_data)} records)")
                X_stock, y_stock, scaler = self._prepare_data_for_stock(stock_data)

                if X_stock is not None and y_stock is not None and scaler is not None:
                    if len(X_stock) > 0: # Ensure sequences were actually created
                        all_X.append(X_stock)
                        all_y.append(y_stock)
                        updated_scalers[stock] = scaler # Store the latest scaler for this stock
                        sufficient_data_for_training = True
                        print(f"  Prepared {len(X_stock)} sequences for {stock}.")
                    else:
                        print(f"  No sequences generated for {stock} despite sufficient initial records (check indicator/NaN logic).")

                else:
                     # Message printed inside _prepare_data_for_stock if data too short after indicators
                     print(f"  Could not prepare sequences for {stock} (likely insufficient data after indicator calculation).")
            else:
                print(f"Skipping stock {stock}: Insufficient records ({len(stock_data)} < {MIN_TRAIN_RECORDS_PER_STOCK})")

        if not sufficient_data_for_training:
            print("Insufficient data across all stocks to perform training.")
            self.scalers.update(updated_scalers) # Still save any new scalers if created
            self._save_scalers()
            return

        # Combine data from all stocks that had sufficient data
        X_train = np.concatenate(all_X, axis=0)
        y_train = np.concatenate(all_y, axis=0)

        # Reshape X for LSTM [samples, time_steps, features] - N_FEATURES is now 6
        # No reshape needed if _prepare_data_for_stock returns correctly shaped np.arrays
        # X_train should already be (num_sequences, N_STEPS, N_FEATURES)
        # y_train should already be (num_sequences,)
        print(f"Total training sequences: {X_train.shape[0]}")
        print(f"Training data shape (X): {X_train.shape}") # Should be (num_seq, 60, 6)
        print(f"Training data shape (y): {y_train.shape}") # Should be (num_seq,)

        # 4. Define or Update Model
        if self.model is None:
            print("Building new LSTM model.")
            self.model = Sequential([
                Input(shape=(N_STEPS, N_FEATURES)), # Input shape uses N_FEATURES
                LSTM(64, return_sequences=True, activation='relu'), # Increased units slightly
                LSTM(64, activation='relu'),
                Dense(32, activation='relu'), # Increased units slightly
                Dense(1) # Output layer predicts the next scaled price (still 1 output)
            ])
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            self.model.summary() # Print model summary
            print("Model compiled.")
        else:
            print("Using existing model for further training.")

        # 5. Train the Model
        print("Starting model training...")
        # Consider using validation_split if you have more data, e.g., validation_split=0.1
        history_callback = self.model.fit(X_train, y_train, epochs=15, batch_size=64, verbose=1) # Increased epochs/batch slightly
        print("Model training finished.")
        self.is_sufficiently_trained = True # Mark as trained

        # 6. Save Updated Model and Scalers
        self._save_model()
        self.scalers.update(updated_scalers) # Update with the latest scalers
        self._save_scalers()

        print("--- Learning Process Finished ---")


    def predict(self, events):
        """
        Predicts future price movement using the 6-feature model.

        Args:
            events (list): A list of tuples or lists, representing the most recent
                           events: (time, stock_ticker, price, volume, moving_average).

        Returns:
            dict: A dictionary with stock tickers as keys and suggestion strings as values.
        """
        print(f"\n--- Starting Prediction Process ({len(events)} recent events) ---")
        suggestions = {}

        if not self.is_sufficiently_trained or self.model is None:
            print("Prediction cannot proceed: Insufficient learning.")
            try:
                unique_stocks_in_event = {ev[1] for ev in events}
                return {stock: "insufficient learning" for stock in unique_stocks_in_event}
            except Exception:
                 return {"error": "insufficient learning and could not parse input events"}

        if not events:
            print("No recent events provided for prediction.")
            return {}

        # 1. Prepare Input Data
        try:
            recent_data = pd.DataFrame(events, columns=['time', 'stock', 'price', 'volume', 'moving_average'])
            recent_data['time'] = pd.to_datetime(recent_data['time'])
            for col in ['price', 'volume', 'moving_average']:
                 recent_data[col] = pd.to_numeric(recent_data[col], errors='coerce')
            recent_data.dropna(subset=['time', 'stock', 'price', 'volume', 'moving_average'], inplace=True)
            recent_data = recent_data.sort_values(by='time')
            if recent_data.empty:
                print("Warning: All recent events were invalid or incomplete.")
                # Try to extract stock names even if data invalid for returning status
                try:
                    unique_stocks_in_event = {ev[1] for ev in events}
                    return {stock: "invalid input event data" for stock in unique_stocks_in_event}
                except: return {"error": "invalid input event data"}

        except Exception as e:
            print(f"Error processing input events for prediction: {e}")
            return {"error": "invalid event format"}

        # Combine recent events with history for context, avoid duplicates
        # Ensure history columns match recent_data before concat
        hist_cols = ['time', 'stock', 'price', 'volume', 'moving_average']
        combined_history = pd.concat([self.history[hist_cols], recent_data], ignore_index=True)
        combined_history = combined_history.drop_duplicates(subset=['time', 'stock']).sort_values(by='time').reset_index(drop=True)

        unique_stocks_to_predict = recent_data['stock'].unique()
        print(f"Predicting for stocks: {list(unique_stocks_to_predict)}")

        # 2. Generate Prediction for each stock
        for stock in unique_stocks_to_predict:
            print(f"Processing prediction for: {stock}")

            if stock not in self.scalers:
                print(f"  -> No scaler found for {stock}. Model cannot predict (stock likely unseen or insufficient data during learn).")
                suggestions[stock] = "insufficient history" # Or maybe "unknown stock"
                continue

            # Get all data for this stock from combined history
            stock_data = combined_history[combined_history['stock'] == stock].copy()
            stock_data = stock_data.sort_values(by='time').reset_index(drop=True)

            # Calculate indicators on the combined data for this stock
            stock_data_with_indicators = self._calculate_indicators(stock_data)

            if stock_data_with_indicators.empty or len(stock_data_with_indicators) < N_STEPS:
                print(f"  -> Insufficient history for {stock} after indicator calculation ({len(stock_data_with_indicators)} < {N_STEPS})")
                suggestions[stock] = "insufficient history"
                continue

            # Prepare the last N_STEPS data points (all features) for prediction
            last_sequence_features_unscaled = stock_data_with_indicators[FEATURE_COLS].values[-N_STEPS:]
            last_actual_price = last_sequence_features_unscaled[-1, PRICE_INDEX] # Get the last actual price

            # Scale the sequence using the stock's specific scaler
            scaler = self.scalers[stock]
            scaled_sequence = scaler.transform(last_sequence_features_unscaled)

            # Reshape for LSTM [1, time_steps, features]
            X_pred = np.reshape(scaled_sequence, (1, N_STEPS, N_FEATURES))

            # Make prediction (predicts the next scaled price)
            predicted_scaled_price = self.model.predict(X_pred, verbose=0)[0][0]

            # Inverse transform the prediction
            # Create a dummy array representing a single time step with N_FEATURES
            # Put the predicted SCALED price into the correct column (PRICE_INDEX)
            # Other feature values don't matter for inverse transforming the price if MinMaxScaler was used
            dummy_row_scaled = np.zeros((1, N_FEATURES))
            dummy_row_scaled[0, PRICE_INDEX] = predicted_scaled_price
            predicted_price = scaler.inverse_transform(dummy_row_scaled)[0, PRICE_INDEX]

            # Calculate percentage change from the last actual price
            # Avoid division by zero if last_actual_price is 0
            if last_actual_price == 0:
                 percentage_change = 0.0
            else:
                percentage_change = (predicted_price - last_actual_price) / last_actual_price

            print(f"  Last Price: {last_actual_price:.4f}, Predicted Price: {predicted_price:.4f}, Change: {percentage_change:.2%}")

            # Map percentage change to suggestion
            if percentage_change > THRESHOLDS["strong_buy"]:
                suggestions[stock] = "strong_buy"
            elif percentage_change > THRESHOLDS["buy"]:
                suggestions[stock] = "buy"
            elif percentage_change < THRESHOLDS["strong_sell"]:
                suggestions[stock] = "strong_sell"
            elif percentage_change < THRESHOLDS["sell"]:
                 suggestions[stock] = "sell"
            else:
                suggestions[stock] = "hold"
            print(f"  -> Suggestion: {suggestions[stock]}")

        print("--- Prediction Process Finished ---")
        return suggestions

# --- Example Usage ---
if __name__ == "__main__":
    # Create instance (will load data if exists)
    suggester = StockSuggester()

    # === Example Learning Phase ===
    # Simulate data with volume and a dummy moving average
    learning_events = []
    base_time = pd.Timestamp.now() - pd.Timedelta(days=20) # Need more days for indicators
    stocks = {'STOCK_A': 100, 'STOCK_B': 200, 'STOCK_C': 50}
    trends = {'STOCK_A': 0.0008, 'STOCK_B': -0.0006, 'STOCK_C': 0.0001}
    vols = {'STOCK_A': 10000, 'STOCK_B': 50000, 'STOCK_C': 5000}
    noise = {'STOCK_A': 0.015, 'STOCK_B': 0.020, 'STOCK_C': 0.010}

    all_stock_data = {}

    for stock, price in stocks.items():
        stock_prices = []
        stock_times = []
        for i in range(300): # Generate more data points
            time = base_time + pd.Timedelta(hours=i)
            price = price * (1 + np.random.normal(trends[stock], noise[stock]))
            volume = vols[stock] * (1 + np.random.uniform(-0.3, 0.3))
            stock_prices.append(price)
            stock_times.append(time)
            # Simple Moving Average calculation for dummy data (e.g., 10 periods)
            current_ma = np.mean(stock_prices[-10:]) if len(stock_prices) >= 10 else price
            learning_events.append((time, stock, price, max(0, volume), current_ma)) # Ensure volume >= 0

        # Store for prediction example continuity
        temp_df = pd.DataFrame(learning_events)
        all_stock_data[stock] = temp_df[temp_df['stock']==stock].copy()


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
             print(f"\nWarning: Could not retrieve last price for {stock}. Skipping prediction simulation for it.")
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

    print("\n--- Final Suggestions ---")
    # Use pprint for better dictionary formatting if many stocks
    import pprint
    pprint.pprint(predictions)

    # Expected outcome hints:
    # STOCK_A: Likely Buy/Strong Buy (upward trend simulated)
    # STOCK_B: Likely Sell/Strong Sell (downward trend simulated)
    # STOCK_C: Likely Hold (slight trend, depends on recent noise)
    # STOCK_D: insufficient history (had only 40 records, likely not enough after indicator NaNs + N_STEPS)
    # STOCK_NEW: insufficient history (no scaler exists)
"""
Key Changes and Explanations:

Import pandas_ta: Added the library for calculating indicators.
Constants Updated:
N_FEATURES is now 6.
FEATURE_COLS: Defined the exact list and order of feature columns used by the model.
PRICE_INDEX: Automatically finds the index of 'price' in FEATURE_COLS, crucial for scaling/inverse scaling 
and extracting the target variable y.
Indicator parameters (EMA_PERIOD, MACD_..., RSI_PERIOD) added.
Minimum record counts (MIN_RECORDS_FOR_INDICATORS, MIN_TRAIN_RECORDS_PER_STOCK, MIN_HISTORY_FOR_PREDICTION) 
adjusted to account for NaNs produced by initial indicator calculations.
History DataFrame: __init__ and _load now handle the volume and moving_average columns.
_calculate_indicators(stock_data):
New private method dedicated to calculating EMA, MACD, and RSI using pandas_ta.
Takes a DataFrame for a single stock.
Uses df.ta.indicator_name(...) to append indicator columns directly.
Handles required column names automatically generated by pandas_ta.
Crucially, it dropna() to remove rows where indicators couldn't be calculated (usually the first few rows).
Selects and returns only the columns specified in FEATURE_COLS, ensuring the correct order.
Includes checks for sufficient data before calculation.
_prepare_data_for_stock(stock_data):
Calls _calculate_indicators first.
Checks for sufficient data after indicators are calculated and NaNs are dropped.
Selects the FEATURE_COLS for scaling.
MinMaxScaler now scales all 6 features together for each stock.
X sequences now contain N_STEPS time points, each with N_FEATURES values.
y target remains the single scaled price value for the next step, extracted using PRICE_INDEX.
learn(events):
Expects input events with 5 items: (time, stock, price, volume, moving_average).
Validates and processes these new inputs.
Updates history with the new columns.
Calls the updated _prepare_data_for_stock.
Model Input layer Input(shape=(N_STEPS, N_FEATURES)) correctly reflects the 6 features.
Model architecture slightly adjusted (more units in LSTM/Dense) to potentially handle more complex patterns from 6 features (optional, can be tuned).
Training data shapes (X_train, y_train) are printed for verification.
predict(events):
Expects recent events with the 5 items.
Combines recent data and history.
Calls _calculate_indicators on the combined data for the specific stock being predicted.
Checks for sufficient history length after indicator calculation.
Selects the last N_STEPS rows of all FEATURE_COLS.
Gets the last_actual_price from the unscaled data.
Scales the 6-feature sequence using the loaded scaler for that stock.
Inverse Scaling: This is important. The model predicts a single scaled value (the price). To inverse transform it correctly using the multi-feature scaler, we create a dummy array of shape (1, N_FEATURES), place the predicted scaled price at the PRICE_INDEX, and then call scaler.inverse_transform. We then extract the value at PRICE_INDEX from the result to get the predicted price.
Calculates percentage change and determines the suggestion.
Example Usage:
Simulated data generation now includes dummy volume and moving_average values.
More historical data points (300) are generated to allow indicators to stabilize.
Prediction example tries to realistically continue the trends and calculates a rolling MA for the simulated recent events.
Added prediction for STOCK_D (insufficient history) and STOCK_NEW (unknown to model/no scaler).
This updated script now leverages multiple features, potentially leading to more nuanced and accurate predictions, although model tuning and feature engineering become even more critical. Remember that the quality of the input moving_average also affects performance.
"""
"""
    Explanation:

Constants: Define filenames, minimum data requirements (MIN_TRAIN_RECORDS_PER_STOCK, MIN_HISTORY_FOR_PREDICTION), LSTM sequence length (N_STEPS), and suggestion thresholds (THRESHOLDS).
StockSuggester Class:
__init__: Initializes paths, loads data using _load, sets is_sufficiently_trained flag.
_load: Handles loading the Keras model (load_model), Pandas history (CSV), and Scikit-learn scalers (joblib.load). Includes error handling.
_save_history, _save_model, _save_scalers: Handles saving the respective data to files.
_prepare_data_for_stock: Takes a DataFrame for a single stock, scales its price data using MinMaxScaler, creates input sequences (X) and target values (y) suitable for the LSTM, and returns X, y, and the scaler used. This per-stock scaling is crucial.
learn(events):
Takes new events, validates them, and adds them to the history DataFrame.
Saves the updated history.
Iterates through each unique stock in the history.
If a stock has enough data (MIN_TRAIN_RECORDS_PER_STOCK), it calls _prepare_data_for_stock.
Collects X, y, and scalers from all eligible stocks.
If any stock provided sufficient data, it concatenates X and y from all stocks.
Reshapes X for the LSTM input shape (samples, timesteps, features).
Builds a new LSTM model if one doesn't exist, or uses the existing one. The model is simple: two LSTM layers and two Dense layers.
Compiles the model (Adam optimizer, MSE loss).
Trains (model.fit) the model on the combined, scaled data.
Sets is_sufficiently_trained to True.
Saves the updated model and the collection of scalers (one for each stock trained).
predict(events):
Checks if the model is trained (is_sufficiently_trained). If not, returns "insufficient learning" for all input stocks.
Takes recent events, combines them with history.
Iterates through the unique stocks in the input events.
Checks if a scaler exists for the stock (it must have been seen and processed during learn).
Checks if the stock has enough historical data points (MIN_HISTORY_FOR_PREDICTION) in the combined history. If not, returns "insufficient history".
Selects the last N_STEPS prices for the stock.
Uses the specific scaler for that stock (loaded during __init__ or updated during learn) to transform this sequence.
Reshapes the sequence for prediction.
Calls model.predict.
Inverse transforms the scaled prediction back to a price value using the same scaler.
Calculates the percentage change between the predicted price and the last actual price in the sequence.
Compares the change to THRESHOLDS to determine the suggestion ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell').
Returns a dictionary of suggestions.
Example Usage (if __name__ == "__main__":)
Creates an instance of StockSuggester.
Generates synthetic data for learning (Stock A, B, C with varying trends and amounts of data).
Calls learn with the initial data.
Calls learn again with more data for Stock C to show incremental learning.
Generates synthetic recent events for prediction.
Calls predict and prints the results.
Includes examples for testing "insufficient history" and "insufficient learning" scenarios.

    """