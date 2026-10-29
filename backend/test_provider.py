from app.providers.yahoo import fetch_stock_data

df = fetch_stock_data("AAPL", period="1y", interval="1d")
print(df.tail())
print("Rows:", len(df))