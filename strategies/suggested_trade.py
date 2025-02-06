from datetime import datetime

from account.account_balance import AccountBalance
from strategies.strategy_params import StrategyParams
from utils.enums import ACTIONS


class SuggestedTrade:

    def __init__(self, action: ACTIONS, symbol: str, price: float, params: StrategyParams, quantity: int=1) -> None:
        assert action in ACTIONS
        self.action = action
        self.symbol = symbol
        self.price = price
        self.params = params
        self.is_open = True
        self.closing_price = None
        self.creation_time = datetime.now()
        self.closing_time = None
        self.quantity = quantity
    
    def __str__(self):
        if self.is_open:
            return(f'[OPEN]  {self.quantity} {self.symbol} @ {self.price}')
        else:
            profit_loss = "[PROFIT]" if self.pnl > 0 else "[LOSS]"
            return(f'[CLOSED] {profit_loss} {self.quantity} {self.symbol} @ {self.price} => {self.closing_price}, PnL {self.pnl} ({self.pnl_pct}%)')
    
    @property
    def premium(self):
        return self.quantity * self.price

    @property
    def profit_or_loss(self):
        if self.is_open:
            return "[NONE]"
        elif self.pnl > 0:
            return "[PROFIT]"
        else:
            return "[LOSS]"

    def close_trade(self, closing_price: float) -> None:
        self.closing_price = closing_price
        self.closing_time = datetime.now()
        self.is_open = False
        AccountBalance().close_position(self.quantity * self.closing_price)
    
    def close_if_reaches_limit(self, current_price: float):
        if self.action == ACTIONS.BUY:
            price_diff_pct = (current_price - self.price)/self.price
            if price_diff_pct <= self.lower_limit_pct or (
                self.upper_limit_pct is not None and price_diff_pct >= self.upper_limit_pct):
                self.close_trade(current_price)
                return True
            return False
        else:
            raise NotImplementedError(f'Need to add support for {ACTIONS.SELL_SHORT}')
    
    @property
    def lower_limit_pct(self):
        return self.params.stop_loss_pct

    @property
    def upper_limit_pct(self):
        return self.params.max_profit_pct
    
    @property
    def pnl(self) -> float:
        if self.is_open:
            return None
        else:
            if self.action == ACTIONS.BUY:
                pnl = (self.closing_price - self.price) * self.quantity
            elif self.action == ACTIONS.SELL_SHORT:
                pnl = (self.price - self.closing_price) * self.quantity
            else:
                raise Exception(f'Unsupported action: {self.action.value}')
            return round(pnl, self.params.rounding_decimals)
    
    @property
    def pnl_pct(self) -> float:
        if self.is_open:
            return None
        else:
            if self.action == ACTIONS.BUY:
                pnl = (self.closing_price - self.price) * 100 /self.price
            elif self.action == ACTIONS.SELL_SHORT:
                pnl = (self.price - self.closing_price) * 100 /self.price
            else:
                raise Exception(f'Unsupported action: {self.action.value}')
            return round(pnl, self.params.rounding_decimals)




