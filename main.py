from YFinance_Data_Aggregator import collect_data_for_stocks, save_to_csv
from ML_Model import ML_Model
from equities import merge_data

def main():
    word = "AAPL"
    filename = merge_data(word)
    if filename:
        print(f"File found: {filename}")
    else:
        print("No file found containing the specified stock name.")

    model = ML_Model()
    model.read_csv(filename)
    model.create_features(1)
    model.create_target()
    model.train_test_split()
    model.scaler()
    model.build_model()
    model.evaluate()
    model.show_performance()
    

if __name__ == "__main__":
    main()


