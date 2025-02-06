from strategies.strategy_params import StrategyParams
from itertools import product

class StrategyParamGenerator:

    def newest_sp500_pnl():
        st = {'buffer_size': 25, 'min_price_rate_change': 0.0005, 
              'stop_loss_pct': -0.025, 'price_trend_fit': 0.75, 'require_asc_volume_trend': True, 
              'volume_trend_fit': 0.25, 'max_profit_pct': 0.02, 
              'keep_open_if_rising': True, 'rounding_decimals': 4, 'one_open_trade': False, 'verbose': False}
        st0 = {
               'buffer_size': 25, 'min_price_rate_change': 0.0005, 'stop_loss_pct': -0.025, 
               'price_trend_fit': 0.75, 'require_asc_volume_trend': True, 'volume_trend_fit': 0.25, 
               'max_profit_pct': 0.015, 'keep_open_if_rising': True, 
               'rounding_decimals': 4, 'one_open_trade': False, 'verbose': False
        }
        st1 = {'buffer_size': 30, 'min_price_rate_change': 0.0025, 'stop_loss_pct': -0.01, 
               'price_trend_fit': 0.85, 'require_asc_volume_trend': True, 'volume_trend_fit': 0.15, 
               'max_profit_pct': 0.015, 'keep_open_if_rising': True, 
               'rounding_decimals': 4, 'one_open_trade': False, 'verbose': False}
        st2 = {'buffer_size': 30, 'min_price_rate_change': 0.0005, 'stop_loss_pct': -0.01, 
               'price_trend_fit': 0.6, 'require_asc_volume_trend': True, 'volume_trend_fit': 0.25, 
               'max_profit_pct': 0.015, 
               'keep_open_if_rising': True, 'rounding_decimals': 4, 'one_open_trade': False, 'verbose': False}
        st3 = {'buffer_size': 30, 'min_price_rate_change': 0.0025, 'stop_loss_pct': -0.01, 
               'price_trend_fit': 0.85, 'require_asc_volume_trend': True, 'volume_trend_fit': 0.15, 
               'max_profit_pct': 0.015, 'keep_open_if_rising': True, 'rounding_decimals': 4, 
               'one_open_trade': False, 'verbose': False}
        st4 = {'buffer_size': 12, 'min_price_rate_change': 0.0025, 'stop_loss_pct': -0.01, 
               'price_trend_fit': 0.6, 'require_asc_volume_trend': True, 'volume_trend_fit': 0.15, 
               'max_profit_pct': 0.015, 
               'keep_open_if_rising': True, 'rounding_decimals': 4, 'one_open_trade': False, 'verbose': False}
        st5 = {'buffer_size': 30, 'min_price_rate_change': 0.0025, 'stop_loss_pct': -0.01, 
               'price_trend_fit': 0.6, 'require_asc_volume_trend': True, 'volume_trend_fit': 0.25, 
               'max_profit_pct': 0.015, 
               'keep_open_if_rising': True, 'rounding_decimals': 4, 'one_open_trade': False, 'verbose': False}
        st6 = {'buffer_size': 12, 'min_price_rate_change': 0.0025, 'stop_loss_pct': -0.01, 
               'price_trend_fit': 0.85, 'require_asc_volume_trend': True, 'volume_trend_fit': 0.15,
                'max_profit_pct': 0.015, 'keep_open_if_rising': True, 'rounding_decimals': 4,
                'one_open_trade': False, 'verbose': False}
        return [ StrategyParams(**params) for params in [st]]# st0, st1, st2, st3, st4, st5, st6]]

    def get_conservative_strategy():
        return StrategyParams(
            buffer_size=30, 
            min_price_rate_change=0.03, 
            stop_loss_pct=0.0, 
            price_trend_fit=0.75,
            require_asc_volume_trend=True,
            volume_trend_fit=0.20,
            max_profit_pct=0.01,
            keep_open_if_rising=True,
            one_open_trade=False,
            verbose=False
        )
    def get_gainers_conservative_strategy():  # 4.9922
        params = {'buffer_size': 30, 'min_price_rate_change': 0.03, 
                  'stop_loss_pct': -0.005, 'price_trend_fit': 0.75, 
                  'require_asc_volume_trend': True, 'volume_trend_fit': 0.15, 
                  'max_profit_pct': 0.01, 'keep_open_if_rising': True, 
                  'rounding_decimals': 4, 'one_open_trade': False, 'verbose': False}
        return StrategyParams(**params)
        
    def get_sp_strategy(): # 1.0343%
        params = {
            'buffer_size': 12, 'min_price_rate_change': 0.008, 
            'stop_loss_pct': -0.01, 'price_trend_fit': 0.9, 
            'require_asc_volume_trend': True, 'volume_trend_fit': 0.25, 
            'max_profit_pct': 0.01, 'keep_open_if_rising': True, 
            'rounding_decimals': 4, 'verbose': False}
        return StrategyParams(**params)
    
    def get_gainers_strategy_1():  # 1.3828%
        params = {'buffer_size': 10, 'min_price_rate_change': 0.1, 
                  'stop_loss_pct': -0.01, 'price_trend_fit': 0.7,
                  'require_asc_volume_trend': True, 'volume_trend_fit': 0.1, 'max_profit_pct': 0.01, 
                  'keep_open_if_rising': True, 'rounding_decimals': 4, 'verbose': False}
        return StrategyParams(**params)

    def get_gainer_strategy_2(): # 2.7214
        params =  {'buffer_size': 15, 'min_price_rate_change': 0.1, 
                   'stop_loss_pct': -0.01, 'price_trend_fit': 0.7, 
                   'require_asc_volume_trend': True, 'volume_trend_fit': 0.1, 'max_profit_pct': 0.025, 
                   'keep_open_if_rising': True, 'rounding_decimals': 4, 'verbose': False}
        return StrategyParams(**params)
    
    
    def generate_strategies():
        params = dict(
            stop_loss_pct = [-0.025, 0.0015],
            price_trend_fit = [0.75, 0.85, 0.95],
            volume_trend_fit = [0.25],
            max_profit_pct = [0.015, 0.02],
            min_price_rate_change = [0.0005, 0.001],
            buffer_size = [25, 20, 30],
        )
        # Generate all combinations of parameter values
        keys = params.keys()
        combinations = list(product(*params.values()))
        # Convert combinations into a list of dictionaries

        print(f"Total combinations: {len(combinations)}")
        
        for combination in combinations:
            params = dict(zip(keys, combination))
            yield StrategyParams(
                verbose=False, require_asc_volume_trend=True,
                keep_open_if_rising=True, **params)






