import os
import datetime
import pickle
import copy
import numpy as np
import pandas as pd
import talib
import time
import random
import alpaca_trade_api as tradeapi
import tensorflow as tf
from collections import deque
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

from config import Config
from utils.constants import APCA_API_KEY_ID, APCA_API_SECRET_KEY

# ========================
# 🔹 CONFIGURATION SECTION
# ========================

# Alpaca API Keys (Replace with your actual keys)
ALPACA_API_KEY = Config()[APCA_API_KEY_ID]  # Add your Alpaca API Key
ALPACA_SECRET_KEY = Config()[APCA_API_SECRET_KEY]  # Add your Alpaca Secret Key

BASE_URL = "https://paper-api.alpaca.markets" # Use live trading by changing to live endpoint

ALPACA_ENABLED = bool(ALPACA_API_KEY and ALPACA_SECRET_KEY)

# Connect to Alpaca API
if ALPACA_ENABLED:
    api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')

# File paths for saving model and memory
MODEL_PATH = "DQN_trading_model.keras"
MEMORY_PATH = "replay_memory.pkl"
CYCLES_COUNT_PATN = "training_stats.pkl"

class Strategy:

    def __init__(self, max_position_size, stop_loss_pct, trailing_stop_pct, gamma, 
                 epsilon, epsilon_min, epsilon_decay, learning_rate, batch_size):
        '''
        Args:
            max_position_size: Max amount invested per stock	🔼 Increase for more risk, 🔽 Decrease for lower risk
            gamma: Discount factor (future reward weighting)	🔼 Higher for long-term profits, 🔽 Lower for short-term safety
            epsilon:	Exploration rate (random trades vs. learned trades)	🔼 Higher means more random trades, 🔽 Lower means more optimal choices
            epsilon_decay:	How fast exploration reduces over time	🔼 Slower decay increases risk, 🔽 Faster decay stabilizes strategy
            stop_loss_pct:	Stop-loss percentage (max loss before selling)	🔼 Higher means more risk, 🔽 Lower means safer exits
            trailing_stop_pct:	Profit lock-in level (auto-sell at profit)	🔼 Lower means safer, 🔽 Higher means more aggressive
            batch_size:	Training batch size for learning	🔼 Higher means more stable learning, 🔽 Lower adapts faster but is riskier

        '''

        self.max_position_size = max_position_size  # ✅ Divesify positions
        self.stop_loss_pct = stop_loss_pct  # ✅ 5% Stop Loss
        self.trailing_stop_pct = trailing_stop_pct  # ✅ 10% Trailing Stop
        self.gamma = gamma  # ✅ Future reward discount
        self.epsilon = epsilon  # ✅ Start with high exploration
        self.epsilon_min = epsilon_min  # ✅ Minimal randomness
        self.epsilon_decay = epsilon_decay  # ✅ Slowly reduce randomness
        self.learning_rate = learning_rate  # ✅ Higher values = faster but riskier learning
        self.batch_size = batch_size  # ✅ Number of trades considered for training
    
    @classmethod
    def conservative(cls):
        '''🔵 Conservative (Low Risk)
           ✅ Small position sizes, fast stop-loss, fast exploration decay.'''
        return cls(
            max_position_size = 500,  
            stop_loss_pct = 0.02,
            trailing_stop_pct = 0.055,
            gamma = 0.80,
            epsilon = 0.75,  
            epsilon_min = 0.01,
            epsilon_decay = 0.998,
            learning_rate = 0.0003,
            batch_size = 200)
    
    @classmethod
    def balanced(cls):
        '''🟡 Balanced (Medium Risk)
           ✅ Default setup with controlled risk.''' 
        return cls(
            max_position_size = 4000,
            stop_loss_pct = 0.05,
            trailing_stop_pct = 0.10,  
            gamma = 0.95,
            epsilon = 1.0, 
            epsilon_min = 0.01, 
            epsilon_decay = 0.995, 
            learning_rate = 0.001, 
            batch_size = 50)
    
    @classmethod
    def aggresive(cls):
        '''🔴 Aggressive (High Risk)
           ✅ Large positions, slow stop-loss, high exploration.'''
        return cls(        
            max_position_size = 5000,
            stop_loss_pct = 0.10,
            trailing_stop_pct = 0.20,
            gamma = 0.99,
            epsilon = 1.2,
            epsilon_min = 0.01, 
            epsilon_decay = 0.99,
            learning_rate = 0.005,
            batch_size = 24)

class DQNTradingAgent:

    def __init__(self, tickers, cash=20000, strategy=None):
        self.tickers = tickers
        self.cash = cash
        self.holdings = {ticker: 0 for ticker in tickers}
        self.position_prices = {ticker: None for ticker in tickers}
        self.trade_log = []
        self.memory = deque(maxlen=2000)
        self.historical_batch_size = 100
        self.training_cycles = 0  # Track the number of learning updates
        self.reward_history = []  # Store average rewards per training cycle
        self.loss_history = []  # Store loss values
        self.epsilon_history = []  # Store epsilon decay over time

        
        self.strategy = strategy if strategy is not None else Strategy.balanced()

        self.realized_profit_loss = 0  # RPL
        self.unrealized_profit_loss = 0  # UPL
        self.model = self.load_or_build_model()
        self.load_replay_memory()

        # Prioritized Experience Replay (PER) Sertings
        self.alpha = 0.6  # How much prioritization to use (0 = uniform, 1 = full prioritization)
        self.beta = 0.4  # Importance sampling correction factor (starts low, increases)
        self.beta_increment = 0.001  # Increases beta over time

        # Cache historical data to prevent redundant API calls
        self.historical_data = {ticker: None for ticker in tickers}

        if ALPACA_ENABLED:
            self.check_alpaca_account()

    # ========================
    # 🔹 Strategy
    # ========================
    @property
    def stop_loss_pct(self):
        return self.strategy.stop_loss_pct
    @property
    def trailing_stop_pct(self):
        return self.strategy.trailing_stop_pct
    @property
    def gamma(self):
        return self.strategy.gamma
    
    @property
    def epsilon(self):
        return self.strategy.epsilon
    @epsilon.setter
    def epsilon(self, value):
        self.strategy.epsilon = value

    @property
    def epsilon_min(self):
        return self.strategy.epsilon_min
    @property
    def epsilon_decay(self):
        return self.strategy.epsilon_decay
    @property
    def learning_rate(self):
        return self.strategy.learning_rate
    @property
    def batch_size(self):
        return self.strategy.batch_size
    @property
    def max_position_size(self):
        return self.strategy.max_position_size
    
    # ========================
    # 🔹 ALPACA ACCOUNT CHECK
    # ========================

    def check_alpaca_account(self):
        """Fetch account balance from Alpaca API."""
        if not ALPACA_ENABLED:
            return
        account = api.get_account()
        self.cash = float(account.cash)
        print(f"💰 Alpaca Account Balance: ${self.cash}")

    # ==========================
    # 🔹 LIVE MARKET DATA FETCH
    # ==========================

    def fetch_live_price(self, ticker):
        """Fetch real-time stock price from Alpaca."""
        if not ALPACA_ENABLED:
            return random.uniform(100, 200)  # Simulated live price
        try:
            latest_bar = api.get_latest_bar(ticker)
            if not latest_bar:
                return None

            latest_data = {
                "timestamp": latest_bar.t,
                "open": latest_bar.o,
                "high": latest_bar.h,
                "low": latest_bar.l,
                "close": latest_bar.c,
                "volume": latest_bar.v
            }

            # Update historical prices dataset
            latest_df = pd.DataFrame([latest_data]) 
            self.historical_data[ticker] = pd.concat([self.historical_data[ticker], latest_df], ignore_index=True)
            self.perform_timestamp_calculations_on_historical_prices(self.historical_data[ticker])

            return latest_bar.c
        except Exception as ex:
            print("⚠️ Error on fetching historical data:", ex)
            return None

    # ==========================
    # 🔹 HISTORICAL DATA FETCHING & HANDLING
    # ==========================
    def fetch_historical_data(self, ticker):
        """Fetch historical stock data once per ticker."""
        if ALPACA_ENABLED:
            bars = api.get_bars(ticker, '1Min', limit=self.historical_batch_size, sort='desc')
            
            bars.df.reset_index()  # Reset index to move timestamp from index to a column
            
            df = pd.DataFrame({
                'timestamp': [bar.t for bar in bars],  # UTC Timestamps from Alpaca
                'open': [bar.o for bar in bars],
                'high': [bar.h for bar in bars],
                'low': [bar.l for bar in bars],
                'close': [bar.c for bar in bars],
                'volume': [bar.v for bar in bars],
            })
            # Sorting by timestamp  because it is in desc order now
            df = df.sort_values(by='timestamp', ascending=True) 
            df = df.reset_index(drop=True)
            
            self.perform_timestamp_calculations_on_historical_prices(df)
            self.historical_data[ticker] = df
        else:
            data = []
            base_price = random.uniform(100, 200)
            for _ in range(100):
                price = base_price * random.uniform(0.98, 1.02)
                data.append({'open': price, 'high': price * 1.01, 'low': price * 0.99, 
                             'close': price, 'volume': random.randint(1000, 5000),
                             'timestamp': datetime.utcnow()})
                             
            df = pd.DataFrame(data)
            df['minutes_since_open'] = df.index
            self.historical_data[ticker] = df
    
    def perform_timestamp_calculations_on_historical_prices(self, df):
            # Convert timestamps to datetime format and adjust to EST
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York')

            # Extract time-based features
            df['hour'] = df['timestamp'].dt.hour
            df['minute'] = df['timestamp'].dt.minute
            df['minutes_since_open'] = (df['hour'] - 9) * 60 + df['minute']  # Market opens at 9:30 AM EST

    # ==========================
    # 🔹 TRADE EXECUTION (LIVE)
    # ==========================

    def execute_trade(self, action, price, ticker):
        """Perform real buy/sell orders via Alpaca API."""
        max_shares = self.max_position_size // price
        action_source = '[Model Prediction]'

        # Apply Stop-Loss & Trailing Stop
        if self.holdings[ticker] > 0 and self.position_prices[ticker]:
            entry_price = self.position_prices[ticker]
            loss_limit = entry_price * (1 - self.stop_loss_pct)  
            profit_limit = entry_price * (1 + self.trailing_stop_pct)  

            if price < loss_limit:
                action = 1
                action_source = '[Loss Limit]'
            elif price > profit_limit:
                action = 1  
                action_source = '[Profit Limit]'

        # Execute Buy
        if action == 0 and self.cash > price and self.holdings[ticker] < max_shares:
            shares = min(self.cash // price, max_shares)
            self.cash -= shares * price
            # TODO: ⚠️ There could be a potential problem here if we add shares to existing position
            # In this case we will override the original position price which might cause invalid calculations of UPL/RPL
            self.holdings[ticker] += shares
            self.position_prices[ticker] = price
            if ALPACA_ENABLED:
                api.submit_order(
                    symbol=ticker, qty=shares, side='buy',
                    type='market', time_in_force='gtc'
                )
            self.trade_log.append({'Action': 'BUY', 'Ticker': ticker, 'Price': price, 'Shares': shares, 'Source': action_source})
            print(f"✅ Bought {shares} shares of {ticker} at ${price} {action_source}")

        # Execute Sell
        elif action == 1 and self.holdings[ticker] > 0:
            self.realized_profit_loss += (price - self.position_prices[ticker]) * self.holdings[ticker]
            self.cash += self.holdings[ticker] * price
            if ALPACA_ENABLED:
                api.submit_order(
                    symbol=ticker, qty=self.holdings[ticker], side='sell',
                    type='market', time_in_force='gtc'
                )
            self.holdings[ticker] = 0
            self.position_prices[ticker] = None
            self.trade_log.append({'Action': 'SELL', 'Ticker': ticker, 'Price': price, 'Source': action_source})
            print(f"✅ Sold {ticker} at ${price} {action_source}")


    def load_or_build_model(self):
        """Load existing model and training cycle stats, or create a new one."""
        if os.path.exists(MODEL_PATH):
            print("📥 Loading saved model...")
            self.training_cycles = 0  # Default
            
            if os.path.exists(CYCLES_COUNT_PATN):
                with open(CYCLES_COUNT_PATN, "rb") as f:
                    stats = pickle.load(f)
                    self.training_cycles = stats.get("training_cycles", 0)
            
            return load_model(MODEL_PATH)
        else:
            print("⚙️ No saved model found, building a new one...")
            self.training_cycles = 0  # Start fresh
            return self.build_dqn_model()

    def build_dqn_model(self):
        """Builds the Deep Q-Network (DQN) model."""
        model = Sequential([
            Dense(64, activation='relu', input_dim=11),  # Updated from 10 → 11 features
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='linear')  # Actions: Buy, Sell, Hold
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save_model(self):
        """Saves the trained model and training cycle count to disk."""
        self.model.save(MODEL_PATH)
        
        # Save training stats
        with open("training_stats.pkl", "wb") as f:
            pickle.dump({CYCLES_COUNT_PATN: self.training_cycles}, f)
        
        print(f"💾 Model saved. Training cycles: {self.training_cycles}")

    def load_replay_memory(self):
        """Loads replay memory from disk."""
        if os.path.exists(MEMORY_PATH):
            print("⏳ Loading replay memory...")
            with open(MEMORY_PATH, "rb") as f:
                self.memory = pickle.load(f)
        else:
            print("⚠️ No replay memory found, starting fresh.")

    def save_replay_memory(self):
        """💾 Saves replay memory to disk."""
        with open(MEMORY_PATH, "wb") as f:
            pickle.dump(self.memory, f)
        print("✅ Replay memory saved successfully.")

    def compute_indicators(self, df):
        """Calculate RSI, MACD, and EMA for RL training."""
        df['EMA50'] = talib.EMA(df['close'], timeperiod=50)
        df['EMA200'] = talib.EMA(df['close'], timeperiod=200)
        macd, macdsignal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['MACD'] = macd
        df['MACD_signal'] = macdsignal
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df.fillna(0, inplace=True)
        return df

    def get_state(self, df, ticker):
        """Convert stock data into RL state vector including time-based features."""
        latest = df.iloc[-1]  # Get the most recent bar

        return np.array([
            latest['close'], latest['volume'], latest['EMA50'], latest['EMA200'],
            latest['MACD'], latest['MACD_signal'], latest['RSI'],
            self.cash, self.holdings[ticker],
            latest['close'] - latest['EMA50'],
            latest['minutes_since_open']
        ])

    def select_action(self, state):
        """Choose an action: Buy (0), Sell (1), Hold (2)."""
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2])
        return np.argmax(self.model.predict(state.reshape(1, -1), verbose=0)[0])

    def calculate_unrealized_profit_loss(self, df):
        """Calculate the unrealized profit/loss for open positions."""
        self.unrealized_profit_loss = sum(
            self.holdings[t] * (df.iloc[-1]['close'] - self.position_prices[t]) if self.position_prices[t] else 0
            for t in self.tickers
        )
    
    def store_experience(self, state, action, reward, next_state):
        """Store experience with initial high priority."""
        if self.memory:
            max_priority = max(float(x[0]) for x in self.memory)
        else:
            max_priority = 1.0  # Default priority if memory is empty

        # Ensure priority is a scalar and not an array
        self.memory.append((float(max_priority), (state, action, reward, next_state)))


    def get_prioritized_sample(self):
        """Retrieve a batch based on priority sampling."""
        if len(self.memory) < self.batch_size:
            return []

        # Ensure that priorities are extracted as scalars
        priorities = np.array([float(x[0]) for x in self.memory], dtype=np.float32)

        probs = priorities ** self.alpha  # Apply prioritization
        probs /= probs.sum()  # Normalize probabilities

        indices = np.random.choice(len(self.memory), self.batch_size, p=probs)  # Sample indices
        batch = [self.memory[i][1] for i in indices]  # Get experiences

        # Compute importance sampling weights
        importance_weights = (1 / len(self.memory) * 1 / probs[indices]) ** self.beta
        importance_weights /= importance_weights.max()  # Normalize

        # Increase beta over time to reduce bias
        self.beta = min(1.0, self.beta + self.beta_increment)

        return batch, indices, importance_weights

    def update_priorities(self, indices, td_errors):
        """Update experience priorities based on TD error."""
        # TODO: Error (and priority) jumps from default of 1 to 60K+ (available cash?) 
        for i, error in zip(indices, td_errors):
            self.memory[i] = (abs(error) + 1e-5, self.memory[i][1])  # Avoid zero priority

    def plot_training_progress(self):
        """Visualizes training progress: Reward, Loss, and Epsilon decay."""
        fig, ax1 = plt.subplots(figsize=(10, 5))

        # Plot reward
        ax1.set_xlabel('Training Cycles')
        ax1.set_ylabel('Avg Reward', color='blue')
        ax1.plot(self.reward_history, label="Average Reward", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Add second y-axis for loss
        ax2 = ax1.twinx()
        ax2.set_ylabel('Loss', color='red')
        ax2.plot(self.loss_history, label="Loss", color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title("Training Progress (Reward & Loss)")
        fig.tight_layout()
        plt.legend()
        plt.show()

        # Plot Epsilon Decay
        plt.figure(figsize=(10, 4))
        plt.plot(self.epsilon_history, label="Epsilon Decay", color='green')
        plt.xlabel("Training Cycles")
        plt.ylabel("Epsilon Value")
        plt.title("Epsilon Decay Over Training")
        plt.legend()
        plt.show()


    def train_rl_agent(self):
        """Train the RL model using Prioritized Experience Replay (PER)."""
        if len(self.memory) < self.batch_size:
            print(f"⚠️ Not enough experiences to train yet ({len(self.memory)} out of {self.batch_size})")
            return
        print(f"⏳ Training model on memory size {len(self.memory)}...")
        batch, indices, importance_weights = self.get_prioritized_sample()
        td_errors = []
        # states, targets, sample_weights = [], [], []
        total_loss = 0  # Track loss

        for i, (state, action, reward, next_state) in enumerate(batch):
            target = reward
            if next_state is not None:
                target += self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1), verbose=0)[0])

            target_f = self.model.predict(state.reshape(1, -1), verbose=0)
            td_error = abs(target - target_f[0][action])  # Compute TD error
            td_errors.append(td_error)

            target_f[0][action] = target  # Update target Q-value
           
            # Collect samples for batch training
            # ✅ Deep Copy to Avoid In-Place Modification
            # states.append(copy.deepcopy(state))
            # targets.append(copy.deepcopy(target_f[0]))
            # sample_weights.append(copy.deepcopy(importance_weights[i]))

            # ✅ Train the model
            history = self.model.fit(
                state.reshape(1, -1), 
                target_f, epochs=1, 
                verbose=0, 
                sample_weight=np.array([importance_weights[i]]))
            total_loss += history.history['loss'][0]  # Track loss

        # ✅ Train the model once per batch (instead of per sample)
        # history = self.model.fit(
        #     np.array(states), np.array(targets),
        #     epochs=1, verbose=0, sample_weight=np.array(sample_weights)
        # )
        avg_reward = np.mean([x[1][2] for x in self.memory])  # Average reward from memory
        self.reward_history.append(avg_reward)  # Store reward for visualization
        self.loss_history.append(total_loss / len(batch))  # Store loss
        self.epsilon_history.append(self.epsilon)  # Track epsilon decay

        # Update priorities in memory
        self.update_priorities(indices, td_errors)

        # Increment training cycle count
        self.training_cycles += 1

        # Reduce exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  

        # Print training cycle stats every 1000 iterations
        if self.training_cycles % 10 == 0:
            print(f"📊 Training Cycle: {self.training_cycles}, Current Epsilon: {self.epsilon:.5f}")
        
    # ====================
    # 🔹 MAIN EXECUTION POINT
    # ====================
    def run_trading_loop(self, episodes=3, sleep_time=2):
        """Run the RL-based trading loop efficiently with optimized data fetching."""
        for ticker in self.tickers:
            self.fetch_historical_data(ticker)  # Fetch once per session
        print('✅ Loaded historical prices')
        print("-" * 50) 
        for episode in range(episodes):
            for ticker in self.tickers:
                current_price = self.fetch_live_price(ticker)
                if current_price is None:
                    print(f"⚠️ Skipping {ticker}, no live price available.")
                    continue  

                df = self.historical_data[ticker]
                df = self.compute_indicators(df)
                
                # This captures the current state of the market before any trade decision is made.
                #The RL agent needs to evaluate the current conditions before taking an action.
                state = self.get_state(df, ticker)

                action = self.select_action(state)
                self.execute_trade(action, current_price, ticker)
                
                self.calculate_unrealized_profit_loss(df)
                reward = self.cash + self.unrealized_profit_loss + self.realized_profit_loss
 
                # After the action is taken (e.g., executing a trade), the state of the environment changes.
                #This captures the new state after the action has influenced the market position (e.g., balance, holdings).
                #It is crucial in RL to form (state, action, reward, next_state) tuples to train the agent.
                next_state = self.get_state(df, ticker)
                
                # ✅ Store experience for PER-based replay
                self.store_experience(state, action, reward, next_state) 

            self.train_rl_agent()  

            
            print(f"=== Episode {episode+1} Summary:")
            print(f"💰 Cash: ${self.cash}")
            print(f"💵 UPL: ${self.unrealized_profit_loss:.2f} (Unrealized Profit/Loss)")
            print(f"💵 RPL: ${self.realized_profit_loss:.2f} (Realized Profit/Loss)")
            print(f"⚖️ Position Limits: ${self.max_position_size} per stock")
            print(f"🔻 Stop-Loss: {self.stop_loss_pct * 100:.1f}%")
            print(f"🚀 Trailing Stop: {self.trailing_stop_pct * 100:.1f}%")
            print(f"📊 Holdings: {self.holdings}")
            print("=" * 50)            

            if (episode + 1) % 5 == 0:
                self.save_model()
                self.save_replay_memory()
            
            if sleep_time:
                time.sleep(sleep_time)
            
        self.check_alpaca_account()
        print("Final Portfolio:")
        for tx in self.trade_log:
            print(tx)
        # After training, plot the progress
        self.plot_training_progress()

# Run the Algorithm

# tickers = ['CEG', 'WBD', 'EL', 'VST', 'DXCM', 'GL', 'TSLA', 'PLTR',  'SMCI']
tickers = [
    'ADSK', 'AEE', 'AIZ', 'AJG', 'ALL', 'AME', 'AMP', 'APD', 'ATO', 
    'AXON', 'BAC', 'BKNG', 'BSX', 'C', 'CINF', 'COST', 'CPAY', 'CPRT', 'DE'
]
rl_trader = DQNTradingAgent(tickers, strategy=Strategy.conservative())
rl_trader.run_trading_loop(episodes=5, sleep_time=60)