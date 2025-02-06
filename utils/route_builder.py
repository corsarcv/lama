import requests
from config import Config
from utils.constants import APCA_API_KEY_ID, APCA_API_SECRET_KEY


class RouteBuilder:

        @classmethod
        def get_base_url(cls):
            return Config()['HISTORICAL_MARKET_DATA_ENDPOINT']
        
        @classmethod
        def get_headers(cls):
            return {
                "accept": "application/json",
                "APCA-API-KEY-ID": Config()[APCA_API_KEY_ID],
                "APCA-API-SECRET-KEY": Config()[APCA_API_SECRET_KEY ]
        }

        @classmethod
        def get(cls, url):
            response = requests.get(url, headers=cls.get_headers())
            response.raise_for_status()  # Raises an HTTPError if the status code is not 2xx
            return response.text