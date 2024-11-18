import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from sklearn.metrics import mean_squared_error

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
    def read_csv(self, filename):
        self.data = pd.read_csv(filename)
        self.data.drop(columns=['Unnamed: 0'], inplace=True)
        return None
    
    @classmethod
    def create_lag(self, lag):
        # Creating lagged returns for the model
        for col in self.data.columns:
            if col not in ['timestamp']:
                self.data[f'{col}_lag_{lag}'] = self.data[col].shift(-1*lag)
        self.data = self.data.dropna()
        
        self.data.set_index('timestamp', inplace=True)
        save_to_csv(self.data, "lagged_data.csv")
        return None
    
    @classmethod
    def create_target(self):
        self.data['JETS_T+1'] = self.data['JETS'].shift(1)
        self.data = self.data.dropna()
        return None
    
    @classmethod
    def train_test_split(self):
        X = self.data.drop(columns=['JETS_T+1'])
        y = self.data['JETS_T+1']
        
        # Split data into training and testing sets, validation set 80%
        split = int(len(X) * 0.8)
        self.X_train, self.X_test = X[:split], X[split:]
        self.y_train, self.y_test = y[:split], y[split:]
        return None
    
    @classmethod
    def scaler(self):
        scaler = StandardScaler()
        scaler.fit(self.X_train)
        self.X_train = scaler.transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)
        return None
    
    @classmethod
    def build_model(self):
        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train.shape[1],), kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.2),
            Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dropout(0.2),
            Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=0.01, l2=0.01)),
            Dense(1)
        ])

        self.model.compile(optimizer='rmsprop', loss='mean_squared_error', metrics=['mse'])

        self.history = self.model.fit(self.X_train, 
                            self.y_train, 
                            epochs=300, 
                            batch_size=64, 
                            validation_split=0.25, 
                            verbose=0)
        return None
    
    @classmethod   
    def evaluate(self):
        self.score = self.model.evaluate(self.X_test, self.y_test, batch_size=128)
        return None
    
    @classmethod
    def show_performance(self):
        # Plot training & validation loss values
        plt.plot(self.history.history['mse'])
        plt.plot(self.history.history['val_mse'])
        plt.title('Model Mean Squared Error')
        plt.ylabel('MSE')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()

        # Plot training & validation loss values
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        return None