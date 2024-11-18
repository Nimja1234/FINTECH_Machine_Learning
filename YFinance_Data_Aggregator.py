import yfinance as yf
import pandas as pd
import os
from datetime import datetime

def get_historical_data(stock, start_date='2024-10-01', end_date='2024-10-31', interval='1h'):
    # Fetch data from yfinance
    data = yf.download(stock, start=start_date, end=end_date, interval=interval)
    
    # Check if data is empty
    if data.empty:
        print(f"Error: No data found for {stock} in the given date range.")
        return None

    # Reset index to get timestamp as a column
    data.reset_index(inplace=True)
    data = data[['Datetime', 'Close']]
    data.columns = ['timestamp', 'price']
    data['stock'] = stock
    
    return data

def calculate_hourly_returns(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Calculate hourly returns only if there are enough prices
    if len(df) < 2:
        print("Not enough price data to calculate hourly returns.")
        return None
    
    df['hourly_return'] = df['price'].pct_change()
    df.dropna(inplace=True)
    return df

def save_to_csv(df, filename):
    if df is None or df.empty:
        print("No data to save.")
        return

    folder_path = "equities_data"
    os.makedirs(folder_path, exist_ok=True)
    
    filepath = os.path.join(folder_path, filename)
    df.to_csv(filepath)
    print(f"Data saved to {filepath}")
    
    return filepath

def collect_data_for_stocks(stocks, start_date='2024-10-01', end_date='2024-10-31', interval='1h'):
    all_data = pd.DataFrame()
    
    for stock in stocks:
        data = get_historical_data(stock, start_date, end_date, interval)
        if data is not None:
            data = calculate_hourly_returns(data)
            if data is not None:
                data = data[['hourly_return']].rename(columns={'hourly_return': stock})
                if all_data.empty:
                    all_data = data
                else:
                    all_data = all_data.join(data, how='outer')
    
    all_data.reset_index(inplace=True)
    return all_data
