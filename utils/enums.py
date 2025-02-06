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