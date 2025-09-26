import os
import pickle
from src.data_validation.model_parameters import compare_models
import pandas as pd 

if __name__ == "__main__":
    
    with open('data/data_processed/census.pkl', 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    print(f"Data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    compare_models(X_train, y_train, X_test, y_test)
    