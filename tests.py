# from gemini_model.test_gemini import ModelTests
# # Running tests from here, as relative import does not work
# ModelTests.proceed()

import pandas as pd
import os
from pathlib import Path


# Display result
pd.set_option('display.max_rows', 100) 

# Load CSV file
folder_name = 'data/model_learning/gemini/'
csv_files = list(Path(folder_name).glob('*.csv'))
csv_file_names = [str(file) for file in csv_files]

total_df = None
for file_name in csv_file_names:
    print(file_name)
    #file_name  = 'data/model_learning/gemini/trade_history_6f_Information Technology_5.csv'
    df = pd.read_csv(os.path.abspath(file_name))  # Replace with your filename
    if total_df is None:
        total_df = df
    else:
        total_df.update(df)

# Ensure proper types
total_df['time'] = pd.to_datetime(total_df['time'], utc=True)
total_df['price'] = pd.to_numeric(total_df['price'], errors='coerce')

# Sort by stock and time
total_df = total_df.sort_values(by=['stock', 'time'])

# Calculate percentage change per stock
total_df['pct_change'] = total_df.groupby('stock')['price'].pct_change()

# Drop NaNs from pct_change (first value per group will be NaN)
total_df = total_df.dropna(subset=['pct_change'])

# Calculate standard deviation of pct_change per stock
volatility = total_df.groupby('stock')['pct_change'].std().reset_index()
volatility.columns = ['stock', 'volatility_index']

# Sort stocks by highest volatility
volatility = volatility.sort_values(by='volatility_index', ascending=False)


print(volatility)

print('Done')