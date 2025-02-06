from account.account_balance import AccountBalance
from strategies.strategy_params import StrategyParams
from strategies.suggested_trade import SuggestedTrade 
from scipy.stats import linregress
import statistics

from strategies.trend import Trend
from utils.enums import ACTIONS, TREND

class Bar:
    def __init__(self, bar_dict: dict):
        self.bar_dict = bar_dict
    
    def __str__(self):
        return f'Open: {self.open}, close: {self.close}, weighted: {self.weighted}, volume: {self.volume}'
    
    @property
    def open(self):
        return self.bar_dict['o']
    
    @property
    def close(self):
        return self.bar_dict['c']
    
    @property
    def weighted(self):
        return self.bar_dict['vw']
    
    @property
    def volume(self):
        return self.bar_dict['v']
    
    @property
    def timestamp(self):
        return self.bar_dict['t']    
    

class BarsAnalysisStrategy:
    ''' This class will process single symbol only.
        We might need a diapatcher to process myltiple symbols at the same time
    '''

    MIN_PRICE_LIMIT = 10.0  # Move to config


    def __init__(self, symbol: str, strategy_params: StrategyParams) -> None:
        self.symbol = symbol
        self.params = strategy_params
        self.queue: list[Bar] = []
        self.trades: list[SuggestedTrade] = []
    
    @property
    def open_suggested_trades(self) -> list[SuggestedTrade]:
        return [t for t in self.trades if t.is_open]
    
    def close_all_open(self, bar_dict: dict) -> None:
        if self.open_suggested_trades:
            self.print(f'Forcing closing position for {self.symbol}')
        for t in self.open_suggested_trades:
            bar = Bar(bar_dict)
            t.close_trade(bar.close)
            self.print(f'[{bar.timestamp}] {t}, PnL: {t.pnl_pct}%')


    def print(self, msg):
        if self.params.verbose:
            print(msg)

    def analyse_next_bar(self, next_bar: dict) -> list[SuggestedTrade]:
        bar = Bar(next_bar)

        liquidate_position = False
        if self.is_unreliable_volume(bar):
            self.print(f'Unreliable volume: {bar.volume}. Skipping')
            return None
        
        elif self.is_big_volume_drop_identified(bar):
            liquidate_position = True

        for trade in self.trades:
            if trade.is_open is False:
                continue
            if len(self.queue) >= self.params.buffer_size: # Move this comparison into trend calculation
                if self.params.keep_open_if_rising:
                    adjusted_trend_fit = (3.0 + self.params.price_trend_fit)/4.0 # Make trade fit closer to 1 by 75%
                    price_trend = self.analyze_trend(
                        [b.close for b in self.queue], min_r_sqr=adjusted_trend_fit)
                    if price_trend.trend == TREND.ASCENDING:  # TODO: Count for SellShorts in the future
                        # Price is still rising
                        continue
            if trade.close_if_reaches_limit(bar.close) is True:
                self.print(f'[{bar.timestamp}] {trade}, PnL: {trade.pnl_pct}%')
                liquidate_position = True  # Move to config?

    
        if liquidate_position and self.open_suggested_trades:
            self.print('Liquidating the whole position to minimise losses')
            self.close_all_open(next_bar)
            return None
        
        suggested_trade = self._suggest(bar)
        if suggested_trade is not None:
            self.trades.append(suggested_trade)
            self.print(f'[{bar.timestamp}] [{len(self.trades)}] Opening trade {suggested_trade}')
        return suggested_trade
    
    def is_big_volume_drop_identified(self, bar: Bar) -> bool:
        if len(self.queue) < self.params.buffer_size:
            return False
        avg_volume = statistics.mean([b.volume for b in self.queue])
        if avg_volume < 100:
            return False
        is_dropped = bar.volume * 50 < avg_volume # Volumn dropping 50+ times - something is going on
        if is_dropped and len(self.open_suggested_trades) > 0:
            self.print(f'Identified huge volume drop for {self.symbol}: from {avg_volume} to {bar.volume}')
        return is_dropped
    
    def is_unreliable_volume(self, bar):
        # Need to find a balance between current method and the one above - when to liquidate vs. ignore
        # Probably also take a look at pnl
        return bar.volume < 5  # Config?
        
        

    def _verify_if_open_tx_need_to_be_closed(self, bar: Bar) -> list[SuggestedTrade]:
        return 
    
    def _suggest(self, bar: Bar):

        if self.params.one_open_trade and len(self.open_suggested_trades) > 0:
            # print("Skipping adding more trades")
            return None
        if bar.close < self.MIN_PRICE_LIMIT:
            return None
        
        if len(self.queue) == self.params.buffer_size:
            self.queue = self.queue[1:]       
        self.queue.append(bar)
        if len(self.queue) < self.params.buffer_size:
            # Not enough samples in thew queue to determine a trend
            return None
        

        price_trend = self.analyze_trend([b.close for b in self.queue], min_r_sqr=self.params.price_trend_fit)
        
        # prior_price =  self.queue[0].close
        #if abs(bar.close - prior_price)/prior_price < self.params.min_price_rate_change:
        if price_trend.rate_of_change < self.params.min_price_rate_change:
            # Price is not changed enough
            return None
        
        if self.params.require_asc_volume_trend:
            volume_trend = self.analyze_trend([b.volume for b in self.queue], min_r_sqr=self.params.volume_trend_fit)
        else:
            volume_trend = None
        
        if self.trades and self.trades[-1].is_open and self.trades[-1].price >= bar.close:
            # Do not buy more if price is dropping
            # print("Stop order", self.trades[-1].price, bar.close)
            return None

        # TODO: Handle Descending price trend later for SellShort transactions
        if price_trend.trend == TREND.ASCENDING and \
            (volume_trend is None or volume_trend.trend == TREND.ASCENDING):
            # Got a ASC hit. Suggest a trade
            quantity = AccountBalance().trade_amount/bar.close
            trade = SuggestedTrade(
                action=ACTIONS.BUY, symbol=self.symbol, price=bar.close, quantity=quantity, params=self.params)
            if AccountBalance().open_position(trade.premium): # TODO Change pricing/quantity part
                return trade
            else:
                # Not enough money
                return None
        return None


    def analyze_trend(self, numbers: list, min_r_sqr=None) -> Trend:
        
        # Calculate the linear regression slope to determine the overall trend
        x = list(range(len(numbers)))  # x-axis values
        slope, intercept, r_value, p_value, std_err = linregress(x, numbers)   
        # Interpret the slope
        rsqr = float(r_value**2)

        if min_r_sqr is not None and rsqr < min_r_sqr:
            trend = TREND.NO_CLEAR_TREND
        elif slope > 0 and rsqr >= min_r_sqr:
            trend = TREND.ASCENDING
        elif slope < 0 and rsqr >= min_r_sqr:
            trend = TREND.DESCENDING
        else:
            trend = TREND.NO_CLEAR_TREND

        slope_pct = slope/statistics.mean(numbers) # average change of price in pct per unit of time

        return Trend(trend = trend, 
                     rate_of_change= round(slope_pct, self.params.rounding_decimals).item(),
                     rsqr = round(rsqr, self.params.rounding_decimals))
    