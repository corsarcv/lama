import pandas as pd
import numpy as np
import os
import joblib  # For saving/loading scalers more efficiently than pickle
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input  # type: ignore

# --- Constants ---
MODEL_FILENAME = 'stock_lstm_model.keras'
HISTORY_FILENAME = 'stock_trade_history.csv'
SCALERS_FILENAME = 'stock_scalers.joblib'
MIN_TRAIN_RECORDS_PER_STOCK = 100 # Minimum data points needed to train for a stock
MIN_HISTORY_FOR_PREDICTION = 60 # Sequence length needed for prediction (must match n_steps)
N_STEPS = 60  # Number of time steps to look back for prediction
N_FEATURES = 1 # We are using only price for now

# Suggestion thresholds (example values, tune these based on strategy/results)
THRESHOLDS = {
    "strong_buy": 0.03,  # Predict price increase > 3%
    "buy": 0.01,         # Predict price increase > 1% and <= 3%
    "sell": -0.01,       # Predict price decrease < -1% and >= -3%
    "strong_sell": -0.03 # Predict price decrease < -3%
    # Hold is the default between sell and buy thresholds
}

class StockSuggester:
    """
    Provides stock trading suggestions based on historical price data
    using an LSTM model.
    """
    def __init__(self, model_path=MODEL_FILENAME, history_path=HISTORY_FILENAME, scalers_path=SCALERS_FILENAME):
        """
        Initializes the StockSuggester, loading existing model, history, and scalers.
        """
        print("Initializing Stock Suggester...")
        self.model_path = model_path
        self.history_path = history_path
        self.scalers_path = scalers_path
        self.model = None
        self.history = pd.DataFrame(columns=['time', 'stock', 'price'])
        self.scalers = {} # Dictionary to store scalers for each stock {'STOCK_NAME': scaler}
        self._load()
        self.is_sufficiently_trained = self.model is not None
        print(f"Model loaded: {self.is_sufficiently_trained}")
        print(f"History records loaded: {len(self.history)}")

    def _load(self):
        """Loads the model, history, and scalers from local files."""
        # Load Keras model
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
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
                self.history['price'] = pd.to_numeric(self.history['price'])
                self.history = self.history.sort_values(by='time').reset_index(drop=True)
                print(f"History loaded successfully from {self.history_path}")
            except Exception as e:
                print(f"Warning: Could not load history from {self.history_path}. Error: {e}")
                self.history = pd.DataFrame(columns=['time', 'stock', 'price'])
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
            # print(f"History saved to {self.history_path}")
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

    def _prepare_data_for_stock(self, stock_data):
        """
        Prepares sequences for LSTM training/prediction for a single stock's data.
        Returns X (sequences), y (targets), and the scaler used.
        """
        if len(stock_data) < N_STEPS + 1:
            return None, None, None # Not enough data to create even one sequence

        # Use only the 'price' column
        prices = stock_data['price'].values.reshape(-1, 1)

        # Scale data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_prices = scaler.fit_transform(prices)

        # Create sequences
        X, y = [], []
        for i in range(N_STEPS, len(scaled_prices)):
            X.append(scaled_prices[i-N_STEPS:i, 0])
            y.append(scaled_prices[i, 0])

        if not X: # Check if any sequences were created
             return None, None, scaler # Return scaler even if no sequences

        return np.array(X), np.array(y), scaler

    def learn(self, events):
        """
        Learns from a series of events and updates the model.

        Args:
            events (list): A list of tuples or lists, where each element
                           is (time, stock_ticker, price).
                           'time' should be convertible to datetime.
        """
        print(f"\n--- Starting Learning Process ({len(events)} new events) ---")
        if not events:
            print("No new events provided for learning.")
            return

        # 1. Input Validation and Data Preparation
        try:
            new_data = pd.DataFrame(events, columns=['time', 'stock', 'price'])
            new_data['time'] = pd.to_datetime(new_data['time'])
            new_data['price'] = pd.to_numeric(new_data['price'])
        except Exception as e:
            print(f"Error processing input events: {e}")
            print("Expected format: list of (time, stock, price)")
            return

        # 2. Append to History and Save
        # Avoid duplicates if the exact same event (time, stock, price) is passed again
        # More robust check might be needed based on exact requirements (e.g., time+stock is unique?)
        self.history = pd.concat([self.history, new_data], ignore_index=True)
        self.history = self.history.drop_duplicates().sort_values(by='time').reset_index(drop=True)
        self._save_history()
        print(f"History updated. Total records: {len(self.history)}")

        # 3. Prepare Data for Training (per stock)
        all_X, all_y = [], []
        updated_scalers = {}
        sufficient_data_for_training = False

        unique_stocks = self.history['stock'].unique()
        print(f"Found {len(unique_stocks)} unique stocks in history.")

        for stock in unique_stocks:
            stock_data = self.history[self.history['stock'] == stock].copy()
            stock_data = stock_data.sort_values(by='time') # Ensure order

            if len(stock_data) >= MIN_TRAIN_RECORDS_PER_STOCK:
                print(f"Processing data for stock: {stock} ({len(stock_data)} records)")
                X_stock, y_stock, scaler = self._prepare_data_for_stock(stock_data)

                if X_stock is not None and y_stock is not None:
                    all_X.append(X_stock)
                    all_y.append(y_stock)
                    updated_scalers[stock] = scaler # Store the latest scaler
                    sufficient_data_for_training = True
                    print(f"  Prepared {len(X_stock)} sequences for {stock}.")
                else:
                     print(f"  Not enough data ({len(stock_data)}) to create sequences (min {N_STEPS + 1}) for {stock}.")
            else:
                print(f"Skipping stock {stock}: Insufficient records ({len(stock_data)} < {MIN_TRAIN_RECORDS_PER_STOCK})")

        if not sufficient_data_for_training:
            print("Insufficient data across all stocks to perform training.")
            # Still save any new scalers that might have been created
            self.scalers.update(updated_scalers)
            self._save_scalers()
            return

        # Combine data from all stocks
        X_train = np.concatenate(all_X, axis=0)
        y_train = np.concatenate(all_y, axis=0)

        # Reshape X for LSTM [samples, time_steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], N_STEPS, N_FEATURES))
        print(f"Total training sequences: {X_train.shape[0]}")

        # 4. Define or Update Model
        if self.model is None:
            print("Building new LSTM model.")
            self.model = Sequential([
                Input(shape=(N_STEPS, N_FEATURES)), # Use Input layer
                LSTM(50, return_sequences=True, activation='relu'),
                LSTM(50, activation='relu'),
                Dense(25, activation='relu'),
                Dense(1) # Output layer predicts the next scaled price
            ])
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            print("Model compiled.")
        else:
            print("Using existing model for further training.")

        # 5. Train the Model
        print("Starting model training...")
        # Consider using more epochs for real-world scenarios
        # Add validation split for better monitoring if you have enough data
        history = self.model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        print("Model training finished.")
        self.is_sufficiently_trained = True # Mark as trained

        # 6. Save Updated Model and Scalers
        self._save_model()
        self.scalers.update(updated_scalers) # Update with the latest scalers
        self._save_scalers()

        print("--- Learning Process Finished ---")


    def predict(self, events):
        """
        Predicts future price movement for stocks based on recent events and history.

        Args:
            events (list): A list of tuples or lists, representing the most recent
                           events, in the format (time, stock_ticker, price).

        Returns:
            dict: A dictionary where keys are stock tickers from the input events
                  and values are suggestion strings ('strong_buy', 'buy', 'hold',
                  'sell', 'strong_sell', 'insufficient_history', or
                  'insufficient_learning').
        """
        print(f"\n--- Starting Prediction Process ({len(events)} recent events) ---")
        suggestions = {}

        if not self.is_sufficiently_trained or self.model is None:
            print("Prediction cannot proceed: Insufficient learning.")
            # Return 'insufficient learning' for all unique stocks in input
            try:
                unique_stocks_in_event = {ev[1] for ev in events}
                return {stock: "insufficient learning" for stock in unique_stocks_in_event}
            except Exception: # Handle potential malformed input
                 return {"error": "insufficient learning and could not parse input events"}


        if not events:
            print("No recent events provided for prediction.")
            return {}

        # 1. Prepare Input Data
        try:
            recent_data = pd.DataFrame(events, columns=['time', 'stock', 'price'])
            recent_data['time'] = pd.to_datetime(recent_data['time'])
            recent_data['price'] = pd.to_numeric(recent_data['price'])
            recent_data = recent_data.sort_values(by='time')
        except Exception as e:
            print(f"Error processing input events for prediction: {e}")
            return {"error": "invalid event format"}

        # Combine recent events with history for context, avoid duplicates
        combined_history = pd.concat([self.history, recent_data], ignore_index=True)
        combined_history = combined_history.drop_duplicates().sort_values(by='time').reset_index(drop=True)

        unique_stocks_to_predict = recent_data['stock'].unique()
        print(f"Predicting for stocks: {list(unique_stocks_to_predict)}")

        # 2. Generate Prediction for each stock
        for stock in unique_stocks_to_predict:
            print(f"Processing prediction for: {stock}")

            # Check if we have a scaler for this stock (meaning it was seen in training)
            if stock not in self.scalers:
                print(f"  -> No scaler found for {stock}. It might be a new stock not seen during 'learn'.")
                suggestions[stock] = "insufficient history" # Treat as insufficient history
                continue

            stock_data = combined_history[combined_history['stock'] == stock].copy()
            stock_data = stock_data.sort_values(by='time')

            if len(stock_data) < MIN_HISTORY_FOR_PREDICTION:
                print(f"  -> Insufficient history for {stock} ({len(stock_data)} < {MIN_HISTORY_FOR_PREDICTION})")
                suggestions[stock] = "insufficient history"
                continue

            # Prepare the last N_STEPS data points for prediction
            last_sequence_prices = stock_data['price'].values[-N_STEPS:].reshape(-1, 1)
            last_actual_price = last_sequence_prices[-1][0] # Get the very last price

            # Scale the sequence using the stock's specific scaler
            scaler = self.scalers[stock]
            scaled_sequence = scaler.transform(last_sequence_prices)

            # Reshape for LSTM [1, time_steps, features]
            X_pred = np.reshape(scaled_sequence, (1, N_STEPS, N_FEATURES))

            # Make prediction (predicts the next scaled price)
            predicted_scaled_price = self.model.predict(X_pred, verbose=0)[0][0]

            # Inverse transform the prediction to get the actual price prediction
            predicted_price = scaler.inverse_transform([[predicted_scaled_price]])[0][0]

            # Calculate percentage change from the last actual price
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
    # Create an instance of the suggester
    # This will automatically try to load existing model/history/scalers
    suggester = StockSuggester()

    # === Example Learning Phase ===
    # Simulate some historical data events
    # (In a real scenario, you'd load this from a file or API)
    learning_events = []
    base_time = pd.Timestamp.now() - pd.Timedelta(days=10)
    # Stock A: Upward trend with noise
    price_a = 100
    for i in range(150): # Generate enough data to meet MIN_TRAIN_RECORDS_PER_STOCK
        time = base_time + pd.Timedelta(hours=i)
        price_a = price_a * (1 + np.random.normal(0.001, 0.01)) + 0.05 # Slight upward bias
        learning_events.append((time, 'STOCK_A', price_a))

    # Stock B: Downward trend with noise
    price_b = 200
    for i in range(150):
        time = base_time + pd.Timedelta(hours=i)
        price_b = price_b * (1 + np.random.normal(-0.0005, 0.015)) - 0.03 # Slight downward bias
        learning_events.append((time, 'STOCK_B', price_b))

    # Stock C: Not enough data initially
    price_c = 50
    for i in range(30): # Less than MIN_TRAIN_RECORDS_PER_STOCK
         time = base_time + pd.Timedelta(hours=i)
         price_c = price_c * (1 + np.random.normal(0, 0.01))
         learning_events.append((time, 'STOCK_C', price_c))

    # Call the learn method
    suggester.learn(learning_events)

    # Add more data for Stock C later (optional - demonstrates incremental learning)
    more_learning_events = []
    base_time_c2 = base_time + pd.Timedelta(hours=30)
    for i in range(100): # Now enough total data for Stock C
         time = base_time_c2 + pd.Timedelta(hours=i)
         price_c = price_c * (1 + np.random.normal(0.0005, 0.01)) # Slight upward trend now
         more_learning_events.append((time, 'STOCK_C', price_c))

    print("\n=== Second Learning Phase (More Data for STOCK_C) ===")
    suggester.learn(more_learning_events)


    # === Example Prediction Phase ===
    # Simulate some recent events for prediction
    prediction_events = []
    current_time = pd.Timestamp.now()
    # Get the latest known prices from history to continue the simulation
    try:
        last_price_a = suggester.history[suggester.history['stock'] == 'STOCK_A']['price'].iloc[-1]
        last_price_b = suggester.history[suggester.history['stock'] == 'STOCK_B']['price'].iloc[-1]
        last_price_c = suggester.history[suggester.history['stock'] == 'STOCK_C']['price'].iloc[-1]
        # Add a few more recent points (less than N_STEPS is fine here)
        for i in range(5):
            time = current_time + pd.Timedelta(minutes=i*10)
            last_price_a *= (1 + np.random.normal(0.0005, 0.005))
            last_price_b *= (1 + np.random.normal(-0.0002, 0.006))
            last_price_c *= (1 + np.random.normal(0.0008, 0.004)) # Continue slight upward
            prediction_events.append((time, 'STOCK_A', last_price_a))
            prediction_events.append((time, 'STOCK_B', last_price_b))
            prediction_events.append((time, 'STOCK_C', last_price_c))
            prediction_events.append((time, 'STOCK_D', 100 + i)) # New stock

    except IndexError:
        print("\nError retrieving last prices. Skipping prediction example simulation.")
        # Add dummy events if history is empty or lacks A/B/C
        if not prediction_events:
             prediction_events = [
                 (current_time + pd.Timedelta(minutes=i), 'STOCK_A', 110+i*0.1) for i in range(5)
             ] + [
                 (current_time + pd.Timedelta(minutes=i), 'STOCK_B', 190-i*0.1) for i in range(5)
             ] + [
                 (current_time + pd.Timedelta(minutes=i), 'STOCK_D', 100 + i) for i in range(5) # New stock
             ]


    # Call the predict method
    predictions = suggester.predict(prediction_events)

    print("\n--- Final Suggestions ---")
    print(predictions)

    # Example: Predicting with insufficient history for a known stock
    print("\n--- Prediction with Insufficient History Example ---")
    # Assume STOCK_A only has 10 data points total in history + events
    # (We can't easily simulate this without manipulating the history file,
    # but if you ran predict with only the first 10 events for STOCK_A,
    # you would get 'insufficient history')
    # Let's simulate passing only a few events for a stock the model *knows*
    # but doesn't have enough *recent* context for THIS prediction call combined with history
    insufficient_pred_events = [
        (pd.Timestamp.now(), 'STOCK_A', suggester.history[suggester.history['stock'] == 'STOCK_A']['price'].iloc[-1] * 1.01)
    ] # Only 1 event, model needs N_STEPS (60)
    # NOTE: The check actually looks at the *combined* history + recent events.
    # So this specific example might still work if history has >60 points.
    # A better test is a stock whose *total* record count < N_STEPS
    # Let's predict for STOCK_D again, which definitely has < N_STEPS history.
    predictions_insufficient = suggester.predict([
         (pd.Timestamp.now(), 'STOCK_D', 105)
    ])
    print("\n--- Final Suggestions (Insufficient History Test) ---")
    print(predictions_insufficient) # Expected: {'STOCK_D': 'insufficient history'}


    # Example: Predicting before learning
    print("\n--- Prediction Before Learning Example ---")
    suggester_unlearned = StockSuggester(model_path="unlearned_model.keras", history_path="unlearned_history.csv", scalers_path="unlearned_scalers.joblib")
    # Ensure the files don't exist or delete them before running this part for a clean test
    try:
        os.remove("unlearned_model.keras")
        os.remove("unlearned_history.csv")
        os.remove("unlearned_scalers.joblib")
        print("Deleted dummy unlearned files.")
    except OSError:
        pass # Files didn't exist, which is fine

    suggester_unlearned = StockSuggester(model_path="unlearned_model.keras", history_path="unlearned_history.csv", scalers_path="unlearned_scalers.joblib")
    pred_before_learn = suggester_unlearned.predict([ (pd.Timestamp.now(), 'ANY_STOCK', 100) ])
    print("\n--- Final Suggestions (Unlearned Test) ---")
    print(pred_before_learn) # Expected: {'ANY_STOCK': 'insufficient learning'}