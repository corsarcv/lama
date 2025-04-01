from datetime import datetime, date, timedelta
from playsound import playsound
import random
import csv
import os

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
THREE_DAYS_AGO_DATE = (datetime.now() - timedelta(days=3)).date()

def generate_timestamp():
    return str(datetime.now()).replace(' ', '').replace(':', '_')

def get_random_business_day_and_next():
    current_year = date.today().year
    
    # Generate all business days in the current year
    business_days = [
        date(current_year, month, day)
        for month in range(1, 13)
        for day in range(1, 32)
        if (day <= (date(current_year, month, 1).replace(day=28) + timedelta(days=4)).day
            and date(current_year, month, day).weekday() < 5)  # Monday to Friday
        if (1 <= month <= 12)  # Month validation
    ]
    
    # Pick a random business day
    random_date = random.choice(business_days)
    
    # Find the next business day
    next_business_date = random_date + timedelta(days=1)
    while next_business_date.weekday() >= 5:  # Skip to Monday if Saturday/Sunday
        next_business_date += timedelta(days=1)
    
    return random_date, next_business_date

def calculate_median(values):
    if not values:
        return None
    
    # Sort the list
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    # Check if the number of elements is odd or even
    if n % 2 == 1:
        # If odd, return the middle value
        return sorted_values[n // 2]
    else:
        # If even, return the average of the two middle values
        mid1 = sorted_values[n // 2 - 1]
        mid2 = sorted_values[n // 2]
        return (mid1 + mid2) / 2

def build_stocks_map():
    SP500_SOURCE = './data/symbols/sp500.csv'
    with open(SP500_SOURCE, mode='r') as file:
        reader = csv.DictReader(file)
        stock_data = [row for row in reader]
    grouped_stocks_data = {}
    for row in stock_data:
        symbol = row['Symbol']
        sector = row['Sector'] 
        if sector not in grouped_stocks_data:
            grouped_stocks_data[sector] = [symbol]
        else:
            grouped_stocks_data[sector].append(symbol) 
    return grouped_stocks_data


def play_success():
    playsound(os.path.join(DATA_FOLDER, 'success.mp3')) 

def play_failure():
    playsound(os.path.join(DATA_FOLDER, 'failure.mp3')) 

def load_watchlist():
    watchlist_file_name = os.path.join(DATA_FOLDER, 'watchlist.csv')
    with open(watchlist_file_name, newline='') as watchlist:
        reader = csv.reader(watchlist)
        next(reader)  # Skip the header
        stock_symbols =[row[0] for row in reader if row]
    return stock_symbols

def get_sector_for_sp500_ticker(ticker):
    grouped_stocks_data = build_stocks_map()
    for sector, symbols in grouped_stocks_data.items():
        if ticker in symbols:
            return sector
    return 'Unknown'
