import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle


'''Carregar dados → Função separada (load_data).

Definir modelos e parâmetros → Um dicionário central (get_models_and_params), fácil de editar.

Execução do GridSearch → Função que roda em todos os modelos e salva os resultados.

Main → Onde você chama as funções.'''


def load_data():
    with open('../data/data_processed/census.pkl', 'rb') as f:
        X_train, y_train, X_test, y_test = pickle.load(f)
    return X_train, y_train, X_test, y_test

def get_models_and_params():
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(),
            'params': {
                'criterion': ['gini', 'entropy'],
                'n_estimators': [10, 40, 100, 150],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 5, 10]
            }
        },
        'SVM': {
            'model': SVC(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 10, 15, 20],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'sag', 'saga'],
                'penalty': ['l1', 'l2', 'elasticnet', 'none']
            }
        }
    }
    return models

def grid_search(X_train, y_train, X_test, y_test):
    results = []    
    models = get_models_and_params()
    for model_name, x in models.items():
        print(f"Running GridSearchCV for {model_name}...")
        grid = GridSearchCV(estimator=x['model'], param_grid=x['params'], cv=5, n_jobs=-1, scoring='accuracy')
        grid.fit(X_train, y_train)
        print(f"Best parameters for {model_name}: {grid.best_params_}")
        print(f"Best cross-validation score for {model_name}: {grid.best_score_}")
        results.append({
            'model': model_name,
            'best_params': grid.best_params_,
            'best_score': grid.best_score_
        })
    return results

def compare_models(X_train, y_train, X_test, y_test):
    results = grid_search(X_train, y_train, X_test, y_test)
    results_df = pd.DataFrame(results)
    print(results_df)
    results_df.to_csv('../reports/model_comparison.csv', index=False)