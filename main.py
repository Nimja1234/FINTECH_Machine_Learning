from YFinance_Data_Aggregator import collect_data_for_stocks, save_to_csv
from ML_Model import ML_Model

def main():
    # Get user input
    stocks = input("Enter the stock symbols separated by commas (e.g., AAPL,MSFT,GOOGL): ").replace("'", "").split(',')
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")

    # Get historical data and calculate hourly returns
    if start_date == '' or end_date == '':
        start_date = '2024-10-01'
        end_date = '2024-10-31'

    combined_data = collect_data_for_stocks(stocks, start_date, end_date)
    
    # Error handling
    if combined_data is None or combined_data.empty:
        print("Failed to retrieve historical data.")
        return  

    filename = save_to_csv(combined_data, f"combined_data_{start_date}_{end_date}.csv")
    model = ML_Model()
    model.read_csv(filename)
    model.create_lag(5)
    model.create_target()
    model.train_test_split()
    model.scaler()
    model.build_model()
    model.evaluate()
    model.show_performance()
    

if __name__ == "__main__":
    main()


