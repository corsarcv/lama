import os
import pickle
import numpy as np
import pandas as pd
import talib
import time
import random
import alpaca_trade_api as tradeapi
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

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

class DQNTradingAgent:

    def __init__(self, tickers, cash=20000, max_position_size=4000):
        self.tickers = tickers
        self.cash = cash
        self.holdings = {ticker: 0 for ticker in tickers}
        self.position_prices = {ticker: None for ticker in tickers}
        self.trade_log = []
        self.max_position_size = max_position_size
        self.memory = deque(maxlen=2000)

        # Strategy
        self.stop_loss_pct = 0.05  # ✅ 5% Stop Loss
        self.trailing_stop_pct = 0.10  # ✅ 10% Trailing Stop
        self.gamma = 0.95  # ✅ Future reward discount
        self.epsilon = 1.0  # ✅ Start with high exploration
        self.epsilon_min = 0.01  # ✅ Minimal randomness
        self.epsilon_decay = 0.995  # ✅ Slowly reduce randomness
        self.learning_rate = 0.001  # ✅ Higher values = faster but riskier learning
        self.batch_size = 32  # ✅ Number of trades considered for training

        self.realized_profit_loss = 0  # RPL
        self.unrealized_profit_loss = 0  # UPL
        self.model = self.load_or_build_model()
        self.load_replay_memory()

        is_live_enabled = False
        if ALPACA_ENABLED and is_live_enabled:  # Does not work on paper api
            self.check_alpaca_account()

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
            return None  
        barset = api.get_latest_trade(ticker)
        return barset.price if barset else None

    # ==========================
    # 🔹 TRADE EXECUTION (LIVE)
    # ==========================

    def execute_trade(self, action, price, ticker):
        """Perform real buy/sell orders via Alpaca API."""
        max_shares = self.max_position_size // price

        # Apply Stop-Loss & Trailing Stop
        if self.holdings[ticker] > 0 and self.position_prices[ticker]:
            entry_price = self.position_prices[ticker]
            loss_limit = entry_price * (1 - self.stop_loss_pct)  
            profit_limit = entry_price * (1 + self.trailing_stop_pct)  

            if price < loss_limit or price > profit_limit:
                action = 1  # Force SELL

        # Execute Buy
        if action == 0 and self.cash > price and self.holdings[ticker] < max_shares:
            shares = min(self.cash // price, max_shares)
            self.cash -= shares * price
            self.holdings[ticker] += shares
            self.position_prices[ticker] = price
            if ALPACA_ENABLED:
                api.submit_order(
                    symbol=ticker, qty=shares, side='buy',
                    type='market', time_in_force='gtc'
                )
            self.trade_log.append({'Action': 'BUY', 'Ticker': ticker, 'Price': price, 'Shares': shares})
            print(f"✅ Bought {shares} shares of {ticker} at ${price}")

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
            self.trade_log.append({'Action': 'SELL', 'Ticker': ticker, 'Price': price})
            print(f"✅ Sold {ticker} at ${price}")


    '''
    Strategies:

    🔵 Conservative (Low Risk)

    ✅ Small position sizes, fast stop-loss, fast exploration decay.

        max_position_size = 2000  
        stop_loss_pct = 0.03  
        trailing_stop_pct = 0.07  
        gamma = 0.90  
        epsilon = 0.8  
        epsilon_decay = 0.998  
        batch_size = 64  

        🟡 Balanced (Medium Risk)

        ✅ Default setup with controlled risk.

            max_position_size = 4000  
            stop_loss_pct = 0.05  
            trailing_stop_pct = 0.10  
            gamma = 0.95  
            epsilon = 1.0  
            epsilon_decay = 0.995  
            batch_size = 32  
       
        🔴 Aggressive (High Risk)

        ✅ Large positions, slow stop-loss, high exploration.

            max_position_size = 8000  
            stop_loss_pct = 0.10  
            trailing_stop_pct = 0.20  
            gamma = 0.99  
            epsilon = 1.2  
            epsilon_decay = 0.99  
            batch_size = 16  

    Parameter	        Description	                    How to Adjust Risk
    max_position_size	Max amount invested per stock	🔼 Increase for more risk, 🔽 Decrease for lower risk
    gamma	Discount factor (future reward weighting)	🔼 Higher for long-term profits, 🔽 Lower for short-term safety
    epsilon	Exploration rate (random trades vs. learned trades)	🔼 Higher means more random trades, 🔽 Lower means more optimal choices
    epsilon_decay	How fast exploration reduces over time	🔼 Slower decay increases risk, 🔽 Faster decay stabilizes strategy
    stop_loss_pct	Stop-loss percentage (max loss before selling)	🔼 Higher means more risk, 🔽 Lower means safer exits
    trailing_stop_pct	Profit lock-in level (auto-sell at profit)	🔼 Lower means safer, 🔽 Higher means more aggressive
    batch_size	Training batch size for learning	🔼 Higher means more stable learning, 🔽 Lower adapts faster but is riskier

    '''


    def load_or_build_model(self):
        """Load existing model or build a new one."""
        if os.path.exists(MODEL_PATH):
            print("Loading saved model...")
            return load_model(MODEL_PATH)
        else:
            print("No saved model found, building a new one...")
            return self.build_dqn_model()

    def build_dqn_model(self):
        """Builds the Deep Q-Network (DQN) model."""
        model = Sequential([
            Dense(64, activation='relu', input_dim=10),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(3, activation='linear')  # Actions: Buy, Sell, Hold
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def save_model(self):
        """Saves the trained model to disk."""
        self.model.save(MODEL_PATH)
        print("Model saved successfully.")

    def load_replay_memory(self):
        """Loads replay memory from disk."""
        if os.path.exists(MEMORY_PATH):
            print("Loading replay memory...")
            with open(MEMORY_PATH, "rb") as f:
                self.memory = pickle.load(f)
        else:
            print("No replay memory found, starting fresh.")

    def save_replay_memory(self):
        """Saves replay memory to disk."""
        with open(MEMORY_PATH, "wb") as f:
            pickle.dump(self.memory, f)
        print("Replay memory saved successfully.")

    def fetch_data(self, ticker):
        """Fetch real-time stock data from Alpaca or generate simulated data."""
        if ALPACA_ENABLED:
            bars = api.get_bars(ticker, '1Min', limit=30)
            #bars = api.get_bars(ticker, '1Min', "2021-06-08", "2021-06-08", limit=10)
            #[ticker]
            df = pd.DataFrame({
                'open': [bar.o for bar in bars],
                'high': [bar.h for bar in bars],
                'low': [bar.l for bar in bars],
                'close': [bar.c for bar in bars],
                'volume': [bar.v for bar in bars]
            })
            print('Ticker', ticker)
            for bar in bars:
                print(bar)
        else:
            data = []
            base_price = random.uniform(100, 200)
            for _ in range(100):
                price = base_price * random.uniform(0.98, 1.02)
                data.append({'open': price, 'high': price * 1.01, 'low': price * 0.99, 'close': price, 'volume': random.randint(1000, 5000)})
            df = pd.DataFrame(data)

        return df

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
        """Convert stock data into RL state vector."""
        latest = df.iloc[-1]
        return np.array([
            latest['close'], latest['volume'], latest['EMA50'], latest['EMA200'],
            latest['MACD'], latest['MACD_signal'], latest['RSI'],
            self.cash, self.holdings[ticker], latest['close'] - latest['EMA50']
        ])

    def select_action(self, state):
        """Choose an action: Buy (0), Sell (1), Hold (2)."""
        if np.random.rand() <= self.epsilon:
            return random.choice([0, 1, 2])
        return np.argmax(self.model.predict(state.reshape(1, -1))[0])

    def execute_trade(self, action, price, ticker):
        """Perform Buy/Sell/Hold based on RL decision and apply risk management."""
        max_shares = self.max_position_size // price

        # Stop-Loss / Trailing Stop Check
        if self.holdings[ticker] > 0 and self.position_prices[ticker]:
            entry_price = self.position_prices[ticker]
            loss_limit = entry_price * (1 - self.stop_loss_pct)  # Stop-Loss
            profit_limit = entry_price * (1 + self.trailing_stop_pct)  # Trailing Stop

            if price < loss_limit:  # Stop-Loss Triggered
                action = 1  # Force SELL
            elif price > profit_limit:  # Trailing Stop Triggered
                action = 1  # Force SELL

        # Execute Buy
        if action == 0 and self.cash > price and self.holdings[ticker] < max_shares:
            shares = min(self.cash // price, max_shares)
            self.cash -= shares * price
            self.holdings[ticker] += shares
            self.position_prices[ticker] = price
            self.trade_log.append({'Action': 'BUY', 'Ticker': ticker, 'Price': price, 'Shares': shares})

        # Execute Sell
        elif action == 1 and self.holdings[ticker] > 0:
            self.realized_profit_loss += (price - self.position_prices[ticker]) * self.holdings[ticker]
            self.cash += self.holdings[ticker] * price
            self.holdings[ticker] = 0
            self.position_prices[ticker] = None
            self.trade_log.append({'Action': 'SELL', 'Ticker': ticker, 'Price': price})

    def calculate_unrealized_profit_loss(self, df):
        """Calculate the unrealized profit/loss for open positions."""
        self.unrealized_profit_loss = sum(
            self.holdings[t] * (df.iloc[-1]['close'] - self.position_prices[t]) if self.position_prices[t] else 0
            for t in self.tickers
        )

    def train_rl_agent(self):
        """Train the DQN model using experience replay."""
        if len(self.memory) < self.batch_size:
            return  # Not enough experiences to train yet

        batch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state in batch:
            target = reward
            if next_state is not None:
                target += self.gamma * np.amax(self.model.predict(next_state.reshape(1, -1))[0])

            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][action] = target  # Update target value for action taken

            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)  # Train model

        # Decay epsilon (less exploration over time)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # ====================
    # 🔹 MAIN EXECUTION
    # ====================
    def run_trading_loop(self, episodes=10):
        """Run the RL-based trading loop with live execution."""
        for episode in range(episodes):
            for ticker in self.tickers:
                if ALPACA_ENABLED:
                    current_price = self.fetch_live_price(ticker)  
                else:
                    current_price = random.uniform(100, 200)  

                if current_price is None:
                    print(f"⚠️ Skipping {ticker}, no live price available.")
                    continue  

                df = self.fetch_data(ticker)
                df = self.compute_indicators(df)
                state = self.get_state(df, ticker)

                action = self.select_action(state)
                self.execute_trade(action, current_price, ticker)
                
                self.calculate_unrealized_profit_loss(df)
                reward = self.cash + self.unrealized_profit_loss + self.realized_profit_loss
                next_state = self.get_state(df, ticker)
                self.memory.append((state, action, reward, next_state))

            self.train_rl_agent()  


            print(f"Episode {episode+1} Summary:")
            print(f"💰 Cash: ${self.cash}")
            print(f"📊 UPL: ${self.unrealized_profit_loss:.2f} (Unrealized Profit/Loss)")
            print(f"💵 RPL: ${self.realized_profit_loss:.2f} (Realized Profit/Loss)")
            print(f"⚖️ Position Limits: ${self.max_position_size} per stock")
            print(f"🔻 Stop-Loss: {self.stop_loss_pct * 100:.1f}%")
            print(f"🚀 Trailing Stop: {self.trailing_stop_pct * 100:.1f}%")
            print(f"Holdings: {self.holdings}")
            print("=" * 50)            

            if (episode + 1) % 5 == 0:
                self.save_model()
                self.save_replay_memory()
            time.sleep(1)

        print("Final Portfolio:")
        for tx in self.trade_log:
            print(tx)


# Run the Algorithm
tickers = ['AAPL', 'MSFT', 'GOOG', 'TSLA']
rl_trader = DQNTradingAgent(tickers)
rl_trader.run_trading_loop()
