
from account.account_balance import AccountBalance
from api_request.api import get_historical_bars_for_a_symbol
from datetime import datetime
import json
import os
import csv

from strategies.bars_analysis import BarsAnalysisStrategy
from strategies.strategy_params import StrategyParams
from strategies.strategy_params_generator import StrategyParamGenerator
from utils.common import calculate_median, generate_timestamp, get_random_business_day_and_next

def collect_market_data(input_file_name):
    with open(input_file_name, mode='r') as file:
        reader = csv.DictReader(file)
        data = [row for row in reader]
    caught_up = False
    for row in data:
        start, end = get_random_business_day_and_next()
        symbol = row['Symbol']  # SRAD is the last symbol
        if ' ' in symbol:
            symbol = symbol.split(' ')[0]
        print(load_bars(symbol, start=start, end=end, limit=1000))

def load_bars(symbol, start, end, limit):
    bars = get_historical_bars_for_a_symbol(
        symbol=symbol, start=start, end=end, limit=limit)

    file_name = f'./data/dumps/bars/{symbol}_{generate_timestamp()}'
    with open(file_name, 'w') as f:
    # with open('./data/dumps/qwe.dat', 'w') as f:
        f.write(bars)
    print(f'File name: {file_name}')
    return file_name

def analyse_bars(bar_dump_name, strategy_params) -> BarsAnalysisStrategy:

    with open(bar_dump_name, 'r') as f:
        bars_json = json.loads(f.read())
        # print(f'Data set length: {len(bars_json['bars'])}')

    strategy = BarsAnalysisStrategy(symbol=bars_json['symbol'], strategy_params=strategy_params)
    if bars_json.get('bars', None) is None:
         print("Warning: error for ", bar_dump_name)
         return (None, 0, "NONE", 0, 0)
    prev_date = None
    prev_row = None
    for row in bars_json['bars']:
        bar_date = datetime.strptime(row['t'], "%Y-%m-%dT%H:%M:%SZ").date()
        if prev_date and bar_date != prev_date and len(strategy.open_suggested_trades) > 0:
            # print(f'End of the day positions sale for {row['symbol']}')
            strategy.close_all_open(prev_row) # Make it configurable?
        prev_date = bar_date
        prev_row = row
        _ = strategy.analyse_next_bar(row)
    # print('*'* 50)
    # print('Done processing')
    # print(f'Remaining open trades: {len(strategy.open_suggested_trades)}')
    # for t in strategy.open_suggested_trades:
    #     print (t)
    
    strategy.close_all_open(row)
    # total_profit = sum([t.pnl_pct for t in strategy.trades if t.profit_or_loss == '[PROFIT]'])
    # total_loss = sum([t.pnl_pct for t in strategy.trades if t.profit_or_loss == '[LOSS]'])  
    #print("PROFIT: ", profit))
    #print("LOSS: ", loss)
    # status = 'PROFIT' if abs(total_profit) > abs(total_loss) else "LOSS"
    #print(bars_json['symbol'], status, total_profit, total_loss, round(total_profit + total_loss, 4))
    
    # if strategy.trades:
    #     print('Trades:')
    #     for t in strategy.trades:
    #         print(t)
    return strategy

def analyze_for_all_data(subfolder):
    folder_path = f'./data/dumps/bars/{subfolder}/'
    files = [file for file in os.listdir(folder_path) 
             if os.path.isfile(os.path.join(folder_path, file))]
    max_profit = 0
    max_strategy = None
    strategies = StrategyParamGenerator.newest_sp500_pnl()
    # strategies = [StrategyParamGenerator.get_sp_strategy()]
    # strategies = StrategyParamGenerator.generate_strategies()
    # strategies = [
    #     StrategyParamGenerator.get_conservative_strategy(),
    #     StrategyParamGenerator.get_gainers_conservative_strategy()
    #     ]
    for i, strategy_params in enumerate(strategies):
        print('-'* 50)
        print(datetime.now())
        print(f'[{i}] Strategy: {strategy_params}')
        account =  AccountBalance(verbose=False)
        account.reset_balance()  #  Imitation only!
        strategy_params.verbose = False
        # total_list = []
        for file_name in files:
            #print(folder_path + file_name)
            strategy = analyse_bars(folder_path + file_name, strategy_params=strategy_params)
        

        if account.pnl > max_profit:
            #print(f'New profit strategy: {strategy_params}')
            max_profit = account.pnl
            max_strategy = strategy
        print(f'Strategy pnl: {account.pnl} In money: {account.in_money}' )
        print(f'Account stats: {account.stats}')
            
    print(f'Max profit: {max_profit}, max strategy: {max_strategy}')
    


# file_name = load_bars('HOOD', 
#     start = "2024-11-01T00:00:00Z", 
#     end = "2024-11-21T00:00:00Z", limit = 10000)

# analyse_bars(file_name, strategy=StrategyParamGenerator.get_single_strategy())

# analyse_bars('./data/dumps/bars/HOOD_2024-11-2311_13_51.252174', strategy=StrategyParamGenerator.get_single_strategy())

# analyse_bars('./data/dumps/bars/CVNA_2024-11-2310_55_27.983450', strategy=StrategyParamGenerator.get_single_strategy())

# collect_market_data('./data/symbols/sp500.csv')
# collect_market_data('./data/symbols/gainers-52wk.csv')

#analyze_for_all_data('GAINERS')
# analyze_for_all_data('SP500')


from strategies.chat_gpt_algo import TradingAlgorithm
# Example usage
trading_algo = TradingAlgorithm(risk_level=.1)  # Medium risk level

price_events = [
    {"c": 112.24, "h": 112.25, "l": 112.145, "n": 180, "o": 112.15, "t": "2024-04-01T15:23:00Z", "v": 9480, "vw": 112.18193},
    {"c": 112.30, "h": 112.35, "l": 112.20, "n": 190, "o": 112.25, "t": "2024-04-01T15:24:00Z", "v": 9500, "vw": 112.25000},
    # Add more price events over time
]

bar_dump_name = './data/dumps/bars/SP500/AMZN_2024-11-2312_57_56.385003'
with open(bar_dump_name, 'r') as f:
    bars_json = json.loads(f.read())

for event in bars_json["bars"]:
    suggestion = trading_algo.process_price_event(event)
    print(suggestion)

# Access the history and position

# Calculate final PNL
unrealized_pnl = trading_algo.calculate_unrealized_pnl()
print("History:", trading_algo.history)
print("Realized PNL:", trading_algo.cash)
print("Unrealized PNL:", unrealized_pnl)
print("Current Position:", trading_algo.position)