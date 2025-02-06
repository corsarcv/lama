from utils.constants import APCA_API_KEY_ID, APCA_API_SECRET_KEY
from config import Config

from utils.enums import MARKET_SOURCES
from utils.route_builder import RouteBuilder

def get_historical_bars_for_a_symbol(
        symbol, start, end, interval='1Min', limit=1000, feed=MARKET_SOURCES.SIP.value):
    base_url = RouteBuilder.get_base_url()
    url = f"{base_url}/stocks/{symbol}/bars?timeframe={interval}&start={start}&end={end}&limit={limit}&feed={feed}"
    return RouteBuilder.get(url)