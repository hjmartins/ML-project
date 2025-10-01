import os
import pickle
from src.data_validation.model_parameters import compare_models
import pandas as pd 
import src.data_validation.cross_validation as cv
import notebook.statics as tn

if __name__ == "__main__":
    
    with open('data/data_processed/census.pkl', 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    print(f"Data loaded. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}, X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    compare_models(X_train, y_train, X_test, y_test)
    random_forest, svm, knn, logistic_regression, mlp, gradient_boosting = cv.cross_validation(10, 5, X_train, X_test, y_train, y_test)
    tn.plot_model_performance(random_forest, svm, knn, logistic_regression, mlp, gradient_boosting)
    tn.teste_hipotese(random_forest, svm, knn, logistic_regression, mlp, gradient_boosting)