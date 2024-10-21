import yfinance as yf
import pandas as pd

# Download Apple stock data for the last year
ticker = 'AAPL'
data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

# Keep only the 'Close' column (price at market close)
apple_data = data[['Close']]

# Save the data to a CSV file
file_path = 'C:/Users/Navid/Desktop/441/apple_stock_ytd_close.csv'
apple_data.to_csv(file_path)

print(f"Data saved to {file_path}")
