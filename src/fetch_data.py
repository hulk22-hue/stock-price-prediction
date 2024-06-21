import yfinance as yf
import os

def fetch_and_save_data(ticker, start_date, end_date, file_path):
    data = yf.download(ticker, start=start_date, end=end_date)
    
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
    
    data.to_csv(file_path)
    print(f"Data for {ticker} downloaded and saved to {file_path}")
