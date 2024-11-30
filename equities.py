import pandas as pd
import os
import csv

def search_file_in_volatility(word="AAPL"):
    directory = 'equity_data/var'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if word in file:
                return os.path.join(root, file)
    return None

def search_file_in_prices(word="AAPL"):
    directory = 'equity_data/prices'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if word in file:
                return os.path.join(root, file)
    return None

def get_volatility(file):
    df = pd.read_csv(file, header=None)   
    # Extract the dates from the first row
    dates = df.iloc[0, 1:].values
    # Extract the times and values from the remaining rows
    times = df.iloc[1:, 0].values
    values = df.iloc[1:, 1:].values
    
    vol_file = 'equity_data/volatility.csv'
    
    # Open the CSV file for writing
    with open(vol_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['DateTime', 'Volatility'])

        for i, date in enumerate(dates):
            for j, time in enumerate(times):
                if "09:30:00" <= time <= "16:00:00" and (time.endswith(":00:00") or time.endswith(":30:00")):
                    csvwriter.writerow([f"{date} {time}", values[j, i]])

    return vol_file

def get_prices(file):
    df = pd.read_csv(file, header=None)   
    # Extract the dates from the first row
    dates = df.iloc[0, 1:].values
    # Extract the times and values from the remaining rows
    times = df.iloc[1:, 0].values
    values = df.iloc[1:, 1:].values
    
    prices_file = 'equity_data/prices.csv'
    
    # Open the CSV file for writing
    with open(prices_file, mode='w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['DateTime', 'Price'])

        for i, date in enumerate(dates):
            for j, time in enumerate(times):
                if "09:30:00" <= time <= "16:00:00" and (time.endswith(":00:00") or time.endswith(":30:00")):
                    csvwriter.writerow([f"{date} {time}", values[j, i]])

    return prices_file

def merge_data(word):
    prices_file = search_file_in_prices(word)
    volatility_file = search_file_in_volatility(word)
    
    prices_file = get_prices(prices_file)
    vol_file = get_volatility(volatility_file)
    
    prices_df = pd.read_csv(prices_file)
    volatility_df = pd.read_csv(vol_file)
    
    merged_df = pd.merge(prices_df, volatility_df, on='DateTime', how='inner')
    
    file = 'equity_data/merged_data.csv'
    merged_df.to_csv(file, index=False)

    return file
