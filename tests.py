from collections import defaultdict
from datetime import timedelta
import numpy as np
import pandas as pd
import os



# Define time windows to analyze after drops
future_offsets = {
    #'1h': timedelta(hours=1),
    #'4h': timedelta(hours=4),
    #'1d': timedelta(days=1),
    #'3d': timedelta(days=3),
    '5d': timedelta(days=5),
}

def analyze_post_drop_behavior(df, drop_events):
    behavior = {k: [] for k in future_offsets}
    for event in drop_events:
        drop_time = event[1]
        drop_price = event[3]
        for label, offset in future_offsets.items():
            future_time = drop_time + offset
            # Find next available time
            future_time = pd.to_datetime(future_time)
            next_available = df[df.index >= future_time]
            if not next_available.empty:
                future_price = next_available.iloc[0]['price']
                pct_change = (future_price - drop_price) / drop_price
                behavior[label].append(pct_change)
    return behavior

def summarize_behavior(behavior_dict):
    summary = {}
    for label, changes in behavior_dict.items():
        if not changes:
            summary[label] = {'avg': None, 'std': None, 'count': 0, 'pos': 0, 'neg': 0}
            continue
        avg = np.mean(changes)
        std = np.std(changes)
        count = len(changes)
        pos = sum(1 for x in changes if x > 0)
        neg = sum(1 for x in changes if x < 0)
        if std < 0.005:
            confidence = 'high'
        elif std < 0.02:
            confidence = 'medium'
        else:
            confidence = 'low'
        summary[label] = {
            'avg': round(avg, 4), 
            'std': round(std, 4), 
            'confidence':confidence, 
            'count': count, 
            'pos': pos, 
            'neg': neg}
    return summary


# Identify intraday drops (>=1% in 4-hour span)
def detect_anomalies(df, drop_threshold=0.01, span_steps=16, is_drop=True):
    anomalies = []
    for i in range(len(df) - span_steps):
        start_idx = df.index[i]
        end_idx = df.index[i + span_steps]
        start_price = df.iloc[i]['price']
        end_price = df.iloc[i + span_steps]['price']
        if is_drop:
            change = (end_price - start_price) / start_price
        else:
            change = (start_price - end_price) / start_price
        
        
        if start_idx.date() == end_idx.date():
            if change > 0 and abs(change) >= drop_threshold:
                anomalies.append((start_idx, end_idx, start_price, end_price, change))
        else:
            for x in range(span_steps-1, 0, -1):
                end_price = df.iloc[i + x]['price']
            if is_drop:
                change = (end_price - start_price) / start_price
            else:
                change = (start_price - end_price) / start_price
                if start_idx.date() == end_idx.date() and change >= drop_threshold:
                    anomalies.append((start_idx, end_idx, start_price, end_price, change))
                    break

    return anomalies

def detect_overnight_anomalies(df, drop_threshold=0.01, is_drop=True):
    df_daily_open = df.groupby(df.index.date).first()
    anomalies = []
    for i in range(len(df_daily_open) - 1):
        date_today = df_daily_open.index[i]
        date_next = df_daily_open.index[i + 1]
        close_today = df[df.index.date == date_today].iloc[-1]['price']
        open_next = df[df.index.date == date_next].iloc[0]['price']
        if is_drop:
            change = (close_today - open_next) / close_today
        else:
            change = (open_next - close_today) / close_today

        if change > 0 and abs(change) >= drop_threshold:
            anomalies.append((date_today, date_next, close_today, open_next, change))
    return anomalies

def get_summary(anomalies):
    if not anomalies:
        return "No anomalies"
    anomalies_avg_list = [rec[key].get('avg', 0) for rec in anomalies if key in rec and rec[key]['avg'] is not None]
    if not anomalies_avg_list:
        return "No anomalies"
    mean = round(100 * np.mean(anomalies_avg_list), 4)
    count = sum([rec[key]['count'] for rec in anomalies if rec[key]['count'] is not None])
    return f"Diff: {mean}%, Records#: {count}"


suffixes = ['Financials', "Real Estate", 'Energy', 'Health Care', 'Industrials', 'Information Technology', 
            'Materials', 'Utilities', 'Communication Services', 'Consumer Discretionary', 'Consumer Staples']
suffixes = ['Financials', "Real Estate", 'Industrials', 
            'Utilities', 'Communication Services']


# suffixes = ['COST', 'AAPL', 'JPM']
# suffixes = ['SPYG']



for suffix in suffixes:
    for pct in [0.005, 0.01, 0.02, 0.05]:
        for is_drop in [True, False]:
            file_name = f'/Users/ihorderevianskyi/dev/alpaca/data/model_learning/gemini/trade_history_6f_{suffix}.csv'
            print("*"*75)
            
            print("Processing file:", suffix, ', is_drop:', is_drop, ', pct:', pct)

            # Load the full dataset
            df = pd.read_csv(os.path.abspath(file_name))
            # Convert to DataFrame
            #df = pd.DataFrame(bars_json['bars'])
            df['time'] = pd.to_datetime(df['time'], utc=True)
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)

            # Resample to 15 minute (ensure no gaps, for simplicity)
            #extended_df = df.resample('15min').ffill()

            # Extend dataset artificially for meaningful analysis
            # extended_df = pd.concat([df]*50)
            # Group by stock symbol and iterate
            intraday_behaviors = []
            overnight_behaviors = []


            for stock, extended_df in df.groupby('stock'):
                # print(f"Processing stock: {stock}")
                #extended_df = df.copy()
                extended_df.index = pd.date_range(start="2024-04-01 00:00", periods=len(extended_df), freq='15min')

                # Use extended data to detect drops
                intraday_drops = detect_anomalies(extended_df, drop_threshold=pct, span_steps=16, is_drop=is_drop)

                overnight_drops = detect_overnight_anomalies(extended_df, drop_threshold=pct, is_drop=is_drop)

                # Analyze and summarize
                intraday_behavior = analyze_post_drop_behavior(extended_df, intraday_drops)
                overnight_behavior = analyze_post_drop_behavior(extended_df, overnight_drops)

                intraday_summary = summarize_behavior(intraday_behavior)
                overnight_summary = summarize_behavior(overnight_behavior)
                intraday_behaviors.append(intraday_summary)
                overnight_behaviors.append(overnight_summary)
                

                # for key, rec in intraday_summary.items():
                #     if rec['pos'] + rec['neg'] == 0:
                #         continue
                #     print(key, rec)
                # print("*" * 100)
                # for key, rec in overnight_summary.items():
                #     if rec['pos'] + rec['neg'] == 0:
                #         continue
                #     print(key, rec)

            for key in future_offsets:
                
                print("Intraday:", get_summary(intraday_behaviors))
                print("Overnight:", get_summary(overnight_behaviors))


    


