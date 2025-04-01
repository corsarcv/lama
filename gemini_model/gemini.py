import logging
import pandas as pd
import numpy as np
import os
import joblib  # For saving/loading scalers more efficiently than pickle
import pandas_ta as ta # For technical indicators
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Input # type: ignore
import warnings

from config import Config
from utils.common import DATA_FOLDER
from utils.enums import Action

# ========================
# 🔹 CONFIGURATION SECTION
# ========================
# --- Logging Setup ---
logging.basicConfig(level=Config.LOG_LEVEL, format='%(asctime)s %(message)s')

# Suppress TensorFlow logs and warnings (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='pandas_ta')


# ========================
# 🔹 CONSTANTS
# ========================

MODEL_DATA_FOLDER = os.path.join(DATA_FOLDER, "model_learning", "gemini")
MODEL_FILENAME = os.path.join( MODEL_DATA_FOLDER, 'lstm_model_6f_{sector}_{n_predict_steps}.keras')
SCALERS_FILENAME = os.path.join(MODEL_DATA_FOLDER, 'sscalers_6f_{sector}_{n_predict_steps}.joblib')
HISTORY_FILENAME = os.path.join(MODEL_DATA_FOLDER, 'trade_history_6f_{sector}.csv')

# ========================
# 🔹 Technical Indicator Parameters (adjust as needed)
# ========================

EMA_PERIOD = 12
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
RSI_PERIOD = 14


# ========================
# 🔹 Define the feature columns we will use (after calculations)
# 🔹 Order matters, especially for identifying the price index later
# ========================

FEATURE_COLS = [
    'price',
    'volume',
    'moving_average',
    f'EMA_{EMA_PERIOD}',
    f'MACD_{MACD_FAST}_{MACD_SLOW}_{MACD_SIGNAL}', # pandas_ta creates this column name
    f'RSI_{RSI_PERIOD}'
]
PRICE_INDEX = FEATURE_COLS.index('price') # Index of the price column within features
N_FEATURES = len(FEATURE_COLS) # 6 features as of now
N_PREDICT_STEPS = 5 # Prediction for what interval (1 - next)

# ========================
# 🔹 Training and Prediction Parameters
# 🔹 Need slightly more history for indicators to become stable
# ========================
MIN_RECORDS_FOR_INDICATORS = max(MACD_SLOW, RSI_PERIOD) + 15 # Min data needed before calculating indicators reliably
MIN_TRAIN_RECORDS_PER_STOCK = MIN_RECORDS_FOR_INDICATORS + 100 # Min records overall for training a stock
LOOKBACK_PERIOD = 60  # Sequence length (lookback period)
MIN_HISTORY_FOR_PREDICTION = LOOKBACK_PERIOD + MIN_RECORDS_FOR_INDICATORS # Need enough history to calculate indicators AND form a sequence

# ========================
# 🔹 Suggestion thresholds for 20 cycles X 15 min (4h)
# 🔹 Remain the same, but interpretation might change with better model
# 🔹 Need to scale for N_PREDICT_SRTEPS less than 20
# ========================


THRESHOLDS = {
    Action.STRONG_BUY: 0.03,  # Predict price increase > 3%
    Action.BUY: 0.01,         # Predict price increase > 1% and <= 3%
    Action.SELL: -0.01,       # Predict price decrease < -1% and >= -3%
    Action.STRONG_SELL: -0.03 # Predict price decrease < -3%
    # Hold is the default between sell and buy thresholds
}

# 💾 ⏳ ✅ 📥 💰 💵 ⚖️ 🔻 🚀 📊 ⚠️ 🔹
class StockSuggester:
    """
    Provides stock trading suggestions based on historical price data and technical indicators
    using an LSTM model with 6 features.
    """
    def __init__(self, sector='default', 
                 n_predict_steps=N_PREDICT_STEPS,
                 lookback_period = LOOKBACK_PERIOD,
                 model_path=MODEL_FILENAME, 
                 history_path=HISTORY_FILENAME, 
                 scalers_path=SCALERS_FILENAME):
        """
        Initializes the StockSuggester, loading existing model, history, and scalers.
        """
        logging.debug("Initializing Stock Suggester (6 Features)...")
        self.sector = sector
        self.n_predict_steps = n_predict_steps
        self.lookback_period = lookback_period
        self.model_path = model_path.format(sector=sector, n_predict_steps=n_predict_steps)        
        self.scalers_path = scalers_path.format(sector=sector, n_predict_steps=n_predict_steps)
        self.history_path = history_path.format(sector=sector)
        self.model = None
        # Initialize history with new columns
        self.history = pd.DataFrame(columns=['time', 'stock', 'price', 'volume', 'moving_average'])
        self.scalers = {} # Dictionary to store scalers for each stock {'STOCK_NAME': scaler}
        self._load()
        self.is_sufficiently_trained = self.model is not None
        logging.debug(f"✅ Model loaded: {self.is_sufficiently_trained}")
        logging.debug(f"✅ History records loaded: {len(self.history)}")
        logging.debug(f"✅ Using Features: {FEATURE_COLS}")
        logging.debug(f"✅ Price Index for Scaling: {PRICE_INDEX}")

    def _load(self):
        """Loads the model, history, and scalers from local files."""
        # Load Keras model
        if os.path.exists(self.model_path):
            try:
                # Provide custom_objects if you use custom layers/activations later
                self.model = load_model(self.model_path) #, custom_objects={...})
                logging.debug(f"📥 Model loaded successfully from {self.model_path}")
            except Exception as e:
                logging.warn(f"⚠️ Warning: Could not load model from {self.model_path}. Error: {e}")
                self.model = None
        else:
            logging.error(f"⚠️ Model file {self.model_path} not found.")
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
                logging.debug(f"📥 History loaded successfully from {self.history_path}")
            except Exception as e:
                logging.warning(f"⚠️ Warning: Could not load history from {self.history_path}. Error: {e}")
                self.history = pd.DataFrame(columns=['time', 'stock', 'price', 'volume', 'moving_average'])
        else:
            logging.error(f"⚠️ History file {self.history_path} not found.")

        # Load scalers
        if os.path.exists(self.scalers_path):
            try:
                self.scalers = joblib.load(self.scalers_path)
                logging.debug(f"📥 Scalers loaded successfully from {self.scalers_path}")
            except Exception as e:
                logging.warn(f"⚠️ Warning: Could not load scalers from {self.scalers_path}. Error: {e}")
                self.scalers = {}
        else:
            logging.error(f"⚠️ Scalers file {self.scalers_path} not found.")

    def _save_history(self):
        """Saves the current history DataFrame to a CSV file."""
        try:
            self.history.to_csv(self.history_path, index=False)
            logging.debug(f"💾 History saved to {self.history_path}")
        except Exception as e:
            logging.error(f"⚠️ Error saving history to {self.history_path}: {e}")

    def _save_model(self):
        """Saves the current Keras model to a file."""
        if self.model:
            try:
                self.model.save(self.model_path)
                logging.debug(f"💾 Model saved to {self.model_path}")
            except Exception as e:
                logging.error(f"⚠️ Error saving model to {self.model_path}: {e}")
        else:
            logging.error("⚠️ No model to save.")

    def _save_scalers(self):
        """Saves the scalers dictionary to a file."""
        try:
            joblib.dump(self.scalers, self.scalers_path)
            logging.debug(f"💾 Scalers saved to {self.scalers_path}")
        except Exception as e:
            logging.error(f"⚠️ Error saving scalers to {self.scalers_path}: {e}")

    def _calculate_indicators(self, stock_data):
        """
        Calculates technical indicators (EMA, MACD, RSI) for a single stock's data.
        Assumes stock_data is sorted by time and has 'price', 'volume' columns.
        Returns DataFrame with indicators added, NaNs dropped.
        """
        if stock_data.empty or len(stock_data) < MIN_RECORDS_FOR_INDICATORS:
             # logging.debug(f"   -> Not enough data ({len(stock_data)}) to calculate indicators (min {MIN_RECORDS_FOR_INDICATORS}).")
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
            logging.warning(f"⚠️ Warning: Not all calculated indicator columns found in DataFrame for stock.")
            # Find which are missing:
            missing = [col for col in required_calculated_cols if col not in df.columns]
            logging.warning(f"⚠️ Missing columns: {missing}")
            # Attempt to select available columns from FEATURE_COLS anyway
            available_feature_cols = [col for col in FEATURE_COLS if col in df.columns]
            if not available_feature_cols: return pd.DataFrame() # Cannot proceed
            #logging.debug(f"Using available feature columns: {available_feature_cols}") # Debug
            # This case shouldn't happen often with pandas_ta but handles edge cases
            return df[available_feature_cols]

        # Ensure all expected base columns are present too
        if not all(col in df.columns for col in ['price', 'volume', 'moving_average']):
             logging.warning("⚠️ Warning: Base columns missing after indicator calculation.")
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

        if stock_data_with_indicators.empty or len(stock_data_with_indicators) < self.lookback_period + self.n_predict_steps:
            # logging.debug(f"   -> Not enough data after indicator calculation ({len(stock_data_with_indicators)}) to create sequences (min {LOOKBACK_PERIOD + 1}).")
            return None, None, None # Not enough data

        # 2. Select Feature columns and Scale Data
        features = stock_data_with_indicators[FEATURE_COLS].values  
        scaler = MinMaxScaler(feature_range=(0, 1))
        # Scale all features together based on the history of this specific stock
        scaled_features = scaler.fit_transform(features)

        # 3. Create sequences
        X, y = [], []
        # Adjust loop range: stop earlier to allow N_PREDICT_STEPS targets
        for i in range(self.lookback_period, len(scaled_features) - self.n_predict_steps + 1): # MODIFIED LOOP END
            X.append(scaled_features[i-self.lookback_period:i, :])
            # Target 'y' is now a vector of the next N_PREDICT_STEPS scaled prices
            y.append(scaled_features[i : i + self.n_predict_steps, PRICE_INDEX]) # MODIFIED TARGET
            # Make sure y is a numpy array for the model

        if not X:
         return None, None, scaler

        return np.array(X), np.array(y), scaler

    # ========================
    # 🔹 LEARN Public Method
    # ========================
    def learn(self, events, update_history=True):
        """
        Learns from a series of events (including price, volume, MA) and updates the model.

        Args:
            events (list): A list of tuples or lists, where each element is
                           (time, stock_ticker, price, volume, moving_average).
                           'time' should be convertible to datetime.
        """
        logging.info(f"\n⏳ Starting Learning Process ({len(events)} new events)")
        if not events:
            logging.warning ("⚠️ No new events provided for learning.")
            return

        # 1. Input Validation and Data Preparation
        try:
            # Expecting 5 columns now, should extend to 'time since opening'
            new_data = pd.DataFrame(events, columns=['time', 'stock', 'price', 'volume', 'moving_average'])
            new_data['time'] = pd.to_datetime(new_data['time'])
            # Convert numeric types, coerce errors to NaN
            for col in ['price', 'volume', 'moving_average']:
                 new_data[col] = pd.to_numeric(new_data[col], errors='coerce')
            # Drop rows where essential inputs are missing
            new_data.dropna(subset=['time', 'stock', 'price', 'volume', 'moving_average'], inplace=True)
            if new_data.empty:
                logging.warning("⚠️ Warning: All input events were invalid or incomplete after validation.")
                return

        except Exception as e:
            logging.error(f"⚠️ Error processing input events: {e}")
            logging.error("⚠️ Expected format: list of (time, stock, price, volume, moving_average)")
            return

        # 2. Append to History and Save
        if update_history:
            self.history = pd.concat([self.history, new_data], ignore_index=True)
            # Drop based on more columns now to avoid duplicates if needed
            self.history = self.history.drop_duplicates(subset=['time', 'stock']).sort_values(by='time').reset_index(drop=True)
            self._save_history()
            logging.debug(f"✅ History updated. Total records: {len(self.history)}")
        else:
            logging.warning('⚠️ Skipping history update.')

        # 3. Prepare Data for Training (per stock)
        all_X, all_y = [], []
        updated_scalers = {}
        sufficient_data_for_training = False

        unique_stocks = self.history['stock'].unique()
        logging.debug(f"📊 Found {len(unique_stocks)} unique stocks in history.")

        for stock in unique_stocks:
            # Get data for the specific stock, ensure it's sorted
            stock_data = self.history[self.history['stock'] == stock].copy()
            stock_data = stock_data.sort_values(by='time').reset_index(drop=True)

            if len(stock_data) >= MIN_TRAIN_RECORDS_PER_STOCK:
                logging.debug(f"⏳ Processing data for stock: {stock} ({len(stock_data)} records)")
                X_stock, y_stock, scaler = self._prepare_data_for_stock(stock_data)

                if X_stock is not None and y_stock is not None and scaler is not None:
                    if len(X_stock) > 0: # Ensure sequences were actually created
                        all_X.append(X_stock)
                        all_y.append(y_stock)
                        updated_scalers[stock] = scaler # Store the latest scaler for this stock
                        sufficient_data_for_training = True
                        logging.debug(f"  ✅ Prepared {len(X_stock)} sequences for {stock}.")
                    else:
                        logging.warning(f"  ⚠️ No sequences generated for {stock} despite sufficient initial records (check indicator/NaN logic).")

                else:
                     # Message logging.debuged inside _prepare_data_for_stock if data too short after indicators
                     logging.warning(f"  ⚠️ Could not prepare sequences for {stock} (likely insufficient data after indicator calculation).")
            else:
                logging.warning(f"⚠️ Skipping stock {stock}: Insufficient records ({len(stock_data)} < {MIN_TRAIN_RECORDS_PER_STOCK})")

        if not sufficient_data_for_training:
            logging.warning("⚠️ Insufficient data across all stocks to perform training.")
            self.scalers.update(updated_scalers) # Still save any new scalers if created
            self._save_scalers()
            return

        # Combine data from all stocks that had sufficient data
        X_train = np.concatenate(all_X, axis=0)
        y_train = np.concatenate(all_y, axis=0)

        # Reshape X for LSTM [samples, time_steps, features] - N_FEATURES is now 6
        # No reshape needed if _prepare_data_for_stock returns correctly shaped np.arrays
        # X_train should already be (num_sequences, LOOKBACK_PERIOD, N_FEATURES)
        # y_train should already be (num_sequences,)
        logging.debug(f"⚖️ Total training sequences: {X_train.shape[0]}")
        logging.debug(f"⚖️ Training data shape (X): {X_train.shape}") # Should be (num_seq, 60, 6)
        logging.debug(f"⚖️ Training data shape (y): {y_train.shape}") # Should be (num_seq,)

        # 4. Define or Update Model
        if self.model is None:
            logging.debug("⏳ Building new LSTM model (Multi-Step Output).")
            self.model = Sequential([
                Input(shape=(self.lookback_period, N_FEATURES)), # Input shape uses N_FEATURES
                LSTM(64, return_sequences=True, activation='relu'), # Increased units slightly
                LSTM(64, activation='relu'),
                Dense(32, activation='relu'), # Increased units slightly
                # The softplus activation insures that the predicted price will never be negative
                Dense(self.n_predict_steps, activation='softplus') # Predict N steps, linear activation default
            ])
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            self.model.summary() # Print model summary
            logging.debug("✅ Model compiled.")
        else:
            logging.debug("✅ Using existing model for further training.")

        # 5. Train the Model
        logging.debug("⏳ Starting model training...")
        # Consider using validation_split if you have more data, e.g., validation_split=0.1
        history_callback = self.model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=1) # Increased epochs/batch slightly

        logging.debug("✅ Model training finished.")
        self.is_sufficiently_trained = True # Mark as trained

        # 6. Save Updated Model and Scalers
        self._save_model()
        self.scalers.update(updated_scalers) # Update with the latest scalers
        self._save_scalers()

        logging.info("🚀 Learning Process Finished")

    # ========================
    # 🔹 LEARN Public Method
    # ========================
    def predict(self, events):
        """
        Predicts future price movement using the 6-feature model.

        Args:
            events (list): A list of tuples or lists, representing the most recent
                           events: (time, stock_ticker, price, volume, moving_average).

        Returns:
            dict: A dictionary with stock tickers as keys and suggestion strings as values.
        """
        logging.debug(f"\n⏳ Starting Prediction Process ({len(events)} recent events)")
        suggestions = {}

        if not self.is_sufficiently_trained or self.model is None:
            logging.warning("Prediction cannot proceed: Insufficient learning.")
            try:
                unique_stocks_in_event = {ev[1] for ev in events}
                return {stock: "insufficient learning" for stock in unique_stocks_in_event}
            except Exception:
                 return {"error": "insufficient learning and could not parse input events"}

        if not events:
            logging.warning("⚠️ No recent events provided for prediction.")
            return {}

        # 1. Prepare Input Data
        try:
            recent_data = pd.DataFrame(events, columns=['time', 'stock', 'price', 'volume', 'moving_average'])
            recent_data['time'] = pd.to_datetime(recent_data['time'], utc=True)
            for col in ['price', 'volume', 'moving_average']:
                 recent_data[col] = pd.to_numeric(recent_data[col], errors='coerce')
            recent_data.dropna(subset=['time', 'stock', 'price', 'volume', 'moving_average'], inplace=True)
            recent_data = recent_data.sort_values(by='time')
            if recent_data.empty:
                logging.warning("⚠️ Warning: All recent events were invalid or incomplete.")
                # Try to extract stock names even if data invalid for returning status
                try:
                    unique_stocks_in_event = {ev[1] for ev in events}
                    return {stock: "invalid input event data" for stock in unique_stocks_in_event}
                except: return {"error": "invalid input event data"}

        except Exception as e:
            logging.error(f"⚠️ Error processing input events for prediction: {e}")
            return {"error": "invalid event format"}

        # Combine recent events with history for context, avoid duplicates
        # Ensure history columns match recent_data before concat
        hist_cols = ['time', 'stock', 'price', 'volume', 'moving_average']
        combined_history = pd.concat([self.history[hist_cols], recent_data], ignore_index=True)
        combined_history = combined_history.drop_duplicates(subset=['time', 'stock']).sort_values(by='time').reset_index(drop=True)

        unique_stocks_to_predict = recent_data['stock'].unique()
        logging.debug(f"⏳ Predicting for stocks: {list(unique_stocks_to_predict)}")

        # 2. Generate Prediction for each stock
        for stock in unique_stocks_to_predict:
            logging.debug(f" ⏳ Processing prediction for: {stock}")

            if stock not in self.scalers:
                logging.warning(f" ⚠️ No scaler found for {stock}. Model cannot predict (stock likely unseen or insufficient data during learn).")
                suggestions[stock] = "Insufficient scaler history" # Or maybe "unknown stock"
                continue

            # Get all data for this stock from combined history
            stock_data = combined_history[combined_history['stock'] == stock].copy()
            stock_data = stock_data.sort_values(by='time').reset_index(drop=True)

            # Calculate indicators on the combined data for this stock
            stock_data_with_indicators = self._calculate_indicators(stock_data)

            if stock_data_with_indicators.empty or len(stock_data_with_indicators) < self.lookback_period:
                logging.warning(f" ⚠️ Insufficient history for {stock} after indicator calculation ({len(stock_data_with_indicators)} < {self.lookback_period})")
                suggestions[stock] = "Insufficient history"
                continue

            # Prepare the last LOOKBACK_PERIOD data points (all features) for prediction
            last_sequence_features_unscaled = stock_data_with_indicators[FEATURE_COLS].values[-self.lookback_period:]
            # Still need the single last actual price for reference/comparison if needed            
            last_actual_price = last_sequence_features_unscaled[-1, PRICE_INDEX] # Get the last actual price

            # Scale the sequence using the stock's specific scaler
            scaler = self.scalers[stock]
            scaled_sequence = scaler.transform(last_sequence_features_unscaled)

            # Reshape for LSTM [1, time_steps, features]
            X_pred = np.reshape(scaled_sequence, (1, self.lookback_period, N_FEATURES))

            # Make prediction (vector, predicts the next scaled prices)
            predicted_scaled_prices_vector = self.model.predict(X_pred, verbose=0)[0] # Shape: (N_PREDICT_STEPS,)

            # Inverse transform *each* predicted scaled price
            # Create a dummy array representing a single time step with N_FEATURES
            # Put the predicted SCALED price into the correct column (PRICE_INDEX)
            # Other feature values don't matter for inverse transforming the price if MinMaxScaler was used
            predicted_prices = []
            for scaled_pred in predicted_scaled_prices_vector:
                dummy_row_scaled = np.zeros((1, N_FEATURES))
                dummy_row_scaled[0, PRICE_INDEX] = scaled_pred
                # Ensure clipping here too if using post-processing for positivity
                inv_pred = scaler.inverse_transform(dummy_row_scaled)[0, PRICE_INDEX]
                epsilon = 1e-4
                predicted_prices.append(max(inv_pred, epsilon)) # Store the inverse-transformed price
            # logging.debug(f" 📊 Last Price: {last_actual_price:.4f}")
            # logging.debug(f" 📊 Predicted Prices for next {self.n_predict_steps} steps: {[f'{p:.4f}' for p in predicted_prices]}")

            # How to generate a suggestion from N future prices (if N > 1)?
            # Option 1: Base it on the final predicted price (step N_PREDICT_STEPS)
            # Option 2: Base it on average trend, or other criteria... (requires definition)
            # For now will keep Option #1
            final_predicted_price = predicted_prices[-1]

            if len(predicted_prices) > 1 and len(set(predicted_prices)) == 1 and final_predicted_price != last_actual_price:
                logging.warning(f"The model predictions seems to be off for {stock}: {[f'{p:.4f}' for p in predicted_prices]}")

            # Calculate percentage change from the last actual price
            # Avoid division by zero if last_actual_price is 0
            if last_actual_price == 0:
                 percentage_change = 0.0
            else:
                percentage_change = (final_predicted_price - last_actual_price) / last_actual_price

            # Scale threshold as they are defined for N_STEPS=20
            scaled_percentage_threshold = percentage_change * 20 / N_PREDICT_STEPS
            # Map percentage change to suggestion
            if scaled_percentage_threshold > THRESHOLDS[Action.STRONG_BUY]:
                action = Action.STRONG_BUY
            elif scaled_percentage_threshold > THRESHOLDS[Action.BUY]:
                action = Action.BUY
            elif scaled_percentage_threshold < THRESHOLDS[Action.STRONG_SELL]:
                action = Action.STRONG_SELL
            elif scaled_percentage_threshold < THRESHOLDS[Action.SELL   ]:
                 action = Action.SELL
            else:
                action = Action.HOLD
            
            suggestions[stock] = {
                'action': action, 
                'last_real_price': last_actual_price,
                'percentage_change': round(percentage_change*100, 2),
                'predicted_price': round(final_predicted_price, 4),
                'prices_vector': [round(p, 4) for p in predicted_prices]
                }
            icon = '⏳' if action == Action.HOLD else '💰' if action in (Action.BUY, Action.STRONG_BUY) else '🔻'
            logging.info(f" {icon} {stock}: Last Price: {last_actual_price:.4f}, " + 
                  f"Predicted Price: {final_predicted_price:.4f}, " + 
                  f"Change: {percentage_change:.2%}, " + 
                  f"Suggestion: {action}")
            if action != Action.HOLD:
                logging.info(f"  🔹 Suggested pricing vector: {[f'{p:.4f}' for p in predicted_prices]}")

        logging.debug("✅ Prediction Process Finished")
        logging.debug('*' * 60)
        return suggestions

    # ========================
    # 🔹 Latest History Timestap for a Stock
    # ========================
    def get_last_history_timestamp_for_stock(self, stock):
        subset = self.history.query(f"stock == '{stock}'")
        return subset['time'].iloc[-1] if not subset.empty else None
