import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


class TradingAlgorithm:
    def __init__(self, risk_level):
        self.risk_level = risk_level
        self.data = pd.DataFrame(columns=['timestamp', 'close', 'high', 'low', 'open', 'volume', 'vwap'])
        self.position = 0  # Current position in shares
        self.cash = 0  # Tracks realized PNL
        self.entry_price = 0  # Price at which the current position was opened
        self.history = []  # To keep track of suggestions and actions

    def process_price_event(self, price_event):
        """
        Process a single price event and update the trading decision.

        Parameters:
            price_event (dict): A dictionary containing 'c', 'h', 'l', 'o', 't', 'v', and 'vw'.

        Returns:
            str: "BUY", "SELL", or "HOLD" based on trend analysis.
        """
        # Transform the input to the internal format
        transformed_event = {
            'timestamp': price_event['t'],
            'close': price_event['c'],
            'high': price_event['h'],
            'low': price_event['l'],
            'open': price_event['o'],
            'volume': price_event['v'],
            'vwap': price_event['vw']
        }

        # Add the new price event to the data
        self.data = pd.concat([
            self.data, 
            pd.DataFrame([transformed_event])
        ], ignore_index=True)

        # Ensure we have enough data to analyze trends
        if len(self.data) < 20:
            # self.history.append("HOLD")
            return "HOLD"

        # Calculate moving averages for trend detection
        self.data['ma_short'] = self.data['close'].rolling(window=5).mean()
        self.data['ma_long'] = self.data['close'].rolling(window=20).mean()

        # Compute percentage change in closing price for trend analysis
        self.data['price_change'] = self.data['close'].pct_change()

        # Linear regression to predict the price trend
        scaler = StandardScaler()
        X = scaler.fit_transform(np.arange(len(self.data)).reshape(-1, 1))
        y = self.data['close'].values

        model = LinearRegression()
        model.fit(X, y)
        trend_slope = model.coef_[0]

        # Risk adjustment: Higher risk allows more aggressive trades
        risk_adjusted_slope = trend_slope * self.risk_level

        # Decision-making logic
        suggestion = "HOLD"
        if risk_adjusted_slope > 0 and self.data['ma_short'].iloc[-1] > self.data['ma_long'].iloc[-1]:
            if self.position == 0:  # Only buy if no position is open
                suggestion = "BUY"
                self.position += 100  # Assume buying 100 shares
                self.entry_price = self.data['close'].iloc[-1]  # Record entry price
        elif risk_adjusted_slope < 0 and self.data['ma_short'].iloc[-1] < self.data['ma_long'].iloc[-1]:
            if self.position > 0:  # Only sell if we have a position
                suggestion = "SELL"
                self.cash += (self.data['close'].iloc[-1] - self.entry_price) * self.position  # Realized PNL
                self.position = 0  # Reset position

        # Record the suggestion in history
        if suggestion != "HOLD":
            self.history.append((suggestion, transformed_event['timestamp'], self.data['close'].iloc[-1]))
        return suggestion

    def calculate_unrealized_pnl(self):
        """
        Calculate the unrealized profit and loss for the current position.

        Returns:
            float: Unrealized PNL
        """
        if self.position > 0:
            return (self.data['close'].iloc[-1] - self.entry_price) * self.position
        return 0