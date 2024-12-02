import pandas as pd
import matplotlib.pyplot as plt

def main():
    mse_agg = [["FNN",1.22*10**-5],
               ["LSTM",1.81*10**-5],
               ["Random Forest",1.46*10**-5],
               ["Linear Regression",1.54*10**-5]]
    
    r_squared_agg = [["FNN",0.7222],
                     ["LSTM",0.7215],
                     ["Random Forest",0.75],
                     ["Linear Regression",0.7377]]
    
    mse_df = pd.DataFrame(mse_agg, columns=["Model", "MSE"])
    r_squared_df = pd.DataFrame(r_squared_agg, columns=["Model", "R-Squared"])

    mse_df.plot(kind='bar', x='Model', y='MSE', legend=False, 
                title='Mean Squared Error by Model', 
                color=['blue', 'green', 'red', 'purple'])
    plt.yscale('linear')
    plt.ylabel('MSE')
    plt.xticks(rotation=45)

    
    r_squared_df.plot(kind='bar', x='Model', y='R-Squared', legend=False, 
                      title='R-Squared by Model', 
                      color=['blue', 'green', 'red', 'purple'])
    plt.yscale('linear')
    plt.ylabel('R-Squared')
    plt.ylim(0.72, 0.76)
    plt.xticks(rotation=45)
    
    plt.show()

if __name__ == "__main__":
    main()