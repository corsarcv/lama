
class StrategyParams:
    def __init__(self, buffer_size: int, min_price_rate_change: float, 
                 stop_loss_pct: float, price_trend_fit: float, 
                 require_asc_volume_trend: bool,
                 volume_trend_fit: float = None,
                 max_profit_pct: float = None,
                 keep_open_if_rising=True,
                 rounding_decimals: int = 4,
                 one_open_trade=False,
                 verbose=True) -> None:
        # Size of the sample set to run analyzis on
        self.buffer_size = buffer_size

        # Min rate of price change in pct to consider making a trade
        self.min_price_rate_change = min_price_rate_change

        # Trigger stop loss short transaction if price got to this point (in pct)
        # positive value mean to sell BEFORE we reached the original price
        # negative value allows to get below the original price by x percent
        self.stop_loss_pct = stop_loss_pct

        # This measure should be between 0 and 1 and shows goodness of fit of the trend
        # Value 1 means that all elementes match the trend. Getting close to 0 means that trend is not clear.
        assert price_trend_fit is not None and 0 <= price_trend_fit <= 1, "Price trend fit should be in range [0-1]"
        self.price_trend_fit = price_trend_fit

        # If set to True, suggest a trade only if volume trend is ascending
        self.require_asc_volume_trend = require_asc_volume_trend

        # Goes to effect only if require_asc_volume_trend is True
        # Minimal r square (goodness of fit) for trade volume
        # Value 1 means that all elementes match the trend. Getting close to 0 means that trend is not clear
        if require_asc_volume_trend:
            assert volume_trend_fit is not None and 0 <= volume_trend_fit <= 1, \
            "Volume trend fit should be in range [0-1]"
        self.volume_trend_fit = volume_trend_fit

        # Automatically close a transaction if profit reaches the value (in pct). None value means no upper limit
        self.max_profit_pct = max_profit_pct

        # Do not close transaction on max_profit_pct point if trend is still good 
        # (with rsqr corrected closer to 1 by 2/3)
        self.keep_open_if_rising = keep_open_if_rising

        # Round decimals to n decimal points
        self.rounding_decimals = rounding_decimals

        self.one_open_trade = one_open_trade

        self.verbose = verbose
    
    def __str__(self) -> str:
        return str(self.__dict__)