import yfinance as yf

ticker = yf.Ticker("TSLA")
info = ticker.info

print("Sector:", info.get("sector"))
print("Industry:", info.get("industry"))