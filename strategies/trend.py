from utils.enums import TREND


class Trend:

    def __init__(self, trend: TREND, rate_of_change: float, rsqr: float) -> None:
        self.trend = trend
        self.rate_of_change = rate_of_change
        self.rsqr = rsqr


    def __str__(self):
        return str({
            "Trend": self.trend.value,
            # Measure the rate of change using the slope
            "Rate of Change": self.rate_of_change,
            # Goodness of fit (how well the trend fits the data)
            "R-squared": self.rsqr  
        })
