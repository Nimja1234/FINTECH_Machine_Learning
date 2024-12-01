import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

def save_to_csv(df, filename):
    if df is None or df.empty:
        print("No data to save.")
        return

    folder_path = "equity_data"
    os.makedirs(folder_path, exist_ok=True)
    
    filepath = os.path.join(folder_path, filename)
    df.to_csv(filepath, index=False)
    print(f"Data saved to {filepath}")
    
    return filepath

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

class ML_Model:
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.model = None
        self.history = None
        self.mse = None
        self.score = None
        
    @classmethod
    def read_csv(self, filename, chunksize=1000):
        set_seed(42)
        
        processed_data = pd.DataFrame()
        for chunk in pd.read_csv(filename, chunksize=chunksize):
            # Process each chunk
            chunk['returns'] = chunk['Price'].pct_change()
            chunk.drop(columns=['Price'], inplace=True)
            chunk.dropna(inplace=True)
            
            # Append the processed chunk to the processed_data DataFrame
            processed_data = pd.concat([processed_data, chunk])
        
        self.data = processed_data
        return None
    
    @classmethod
    def create_features(self, lag):
        
        # Creating lagged returns for the model
        for col in self.data.columns:
            if col not in ['DateTime']:
                self.data[f'{col}_lag_{lag}'] = self.data[col].shift(-1*lag)
        self.data = self.data.dropna()
        
        # noise = np.random.normal(1, 0.04, self.data.shape[0])
        # self.data[f'returns_lag_{lag}_noisy'] = self.data[f'returns_lag_{lag}'] + noise
        # self.data[f'return_noisy'] = self.data['returns'] + noise
        # self.data[f'volaility_lag_{lag}_noisy'] = self.data[f'Volatility_lag_{lag}'] + noise
        # self.data[f'volatility_noisy'] = self.data['Volatility'] + noise
        # self.data = self.data.dropna()
        
        self.data.set_index('DateTime', inplace=True)
        return None
    
    @classmethod
    def create_target(self):
        self.data['Volatility_T+1'] = self.data['Volatility'].shift(1)
        self.data = self.data.dropna()
        save_to_csv(self.data, "ML_data.csv")
        return None
    
    @classmethod
    def train_test_split(self):
        X = self.data.drop(columns=['Volatility_T+1'])
        y = self.data['Volatility_T+1']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        return None
    
    @classmethod
    def scaler(self):
        self.y_train_index = self.y_train.index
        self.y_test_index = self.y_test.index

        self.scaler_X = StandardScaler()
        self.X_train = self.scaler_X.fit_transform(self.X_train)
        self.X_test = self.scaler_X.transform(self.X_test)
        
        self.scaler_y = StandardScaler()
        self.y_train = self.scaler_y.fit_transform(self.y_train.values.reshape(-1, 1))
        self.y_test = self.scaler_y.transform(self.y_test.values.reshape(-1, 1))
        return None
    
    @classmethod
    def build_model(self):
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train.shape[1],)),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1, activation='linear')  # Output layer for regression
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        self.history = self.model.fit(self.X_train, 
                            self.y_train, 
                            epochs=100, 
                            batch_size=128, 
                            validation_split=0.2, 
                            callbacks=[early_stopping],
                            verbose=0)
        return None
    
    @classmethod   
    def evaluate(self):
        self.score = self.model.evaluate(self.X_test, self.y_test, batch_size=128)
        return None
    
    @classmethod
    def show_performance(self):
        # Predict the values for X_test
        self.y_pred = self.model.predict(self.X_test)

        # Inverse transform the predictions and actual values
        self.y_test_inv = self.scaler_y.inverse_transform(self.y_test)
        self.y_pred_inv = self.scaler_y.inverse_transform(self.y_pred)

        # Reassign the DateTime index
        self.y_test_inv = pd.DataFrame(self.y_test_inv, index=self.y_test_index, columns=['y_test_inv'])
        self.y_pred_inv = pd.DataFrame(self.y_pred_inv, index=self.y_test_index, columns=['y_pred_inv'])
        
        # Calculate the Mean Squared Error
        self.mse = mean_squared_error(self.y_test_inv, self.y_pred_inv)
        print(f'Mean Squared Error: {self.mse}')
        
        # Calculate the Mean Absolute Error
        self.mae = mean_absolute_error(self.y_test_inv, self.y_pred_inv)
        print(f'Mean Absolute Error: {self.mae}')
        
        # Calculate the R^2 value
        self.r2 = r2_score(self.y_test_inv, self.y_pred_inv)
        print(f'R^2 Value: {self.r2}')
        
        # Line graph of y_test vs y_pred values
        plt.figure(figsize=(10, 6))
        plt.plot(self.y_test_inv.index, self.y_test_inv['y_test_inv'], label='Actual Values')
        plt.plot(self.y_pred_inv.index, self.y_pred_inv['y_pred_inv'], label='Predicted Values')
        plt.xlabel('Date-Time')
        plt.ylabel('Volatility')
        plt.title('Actual Vol. vs Predicted Vol.')
        plt.legend()
        plt.show()

        # Residual plot
        residuals = self.y_test_inv['y_test_inv'] - self.y_pred_inv['y_pred_inv']
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_pred_inv['y_pred_inv'], residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()

        # Histogram of residuals
        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residuals')
        plt.title('Histogram of Residuals')
        plt.show()

        # Scatter plot of actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test_inv['y_test_inv'], self.y_pred_inv['y_pred_inv'])
        plt.plot([self.y_test_inv['y_test_inv'].min(), self.y_test_inv['y_test_inv'].max()], 
                 [self.y_test_inv['y_test_inv'].min(), self.y_test_inv['y_test_inv'].max()], 
                 color='r', linestyle='--')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()

        # Learning curves
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.show()

        return None
