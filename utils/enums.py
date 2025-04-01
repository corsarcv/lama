from enum import Enum

class MARKET_SOURCES(Enum):
    SIP = 'sip'
    IEX = 'iex'

class ACTIONS(Enum):
    BUY = 'Buy'
    SELL_SHORT = 'SellShort'

class TREND(Enum):
    ASCENDING = "Ascending"
    DESCENDING = "Descending"
    NO_CLEAR_TREND = "No clear trend"

class Action(Enum):
    BUY = 'buy'
    STRONG_BUY = 'strong_buy'
    SELL = 'sell'
    STRONG_SELL = 'strong_sell'
    HOLD = 'hold'