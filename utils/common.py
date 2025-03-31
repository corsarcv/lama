from datetime import datetime, date, timedelta
import random

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