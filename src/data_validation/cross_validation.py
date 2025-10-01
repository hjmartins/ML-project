import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.exceptions import NotFittedError
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import pickle
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import GradientBoostingClassifier

def cross_validation(n,fd, X_train, X_test, y_train, y_test):
    random_forest = []
    svm = []
    knn = []
    logistic_regression = []
    mlp = []
    gradient_boosting = []

    for i in range(n):
        kf = KFold(n_splits=fd, shuffle=True, random_state=i)
        rf = RandomForestClassifier(criterion='gini', n_estimators=100, min_samples_split=2, min_samples_leaf=1)
        svm_model = SVC(C=1, kernel='rbf', gamma='scale')
        knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski')
        log_reg = LogisticRegression(C=1, max_iter=1000, solver='liblinear')
        mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', max_iter=200)
        gradient = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)

        score_rf = cross_val_score(rf, X_train, y_train, cv=kf, scoring='accuracy')
        score_svm = cross_val_score(svm_model, X_train, y_train, cv=kf, scoring='accuracy')
        score_knn = cross_val_score(knn_model, X_train, y_train, cv=kf, scoring='accuracy')
        score_log = cross_val_score(log_reg, X_train, y_train, cv=kf, scoring='accuracy')
        score_mlp = cross_val_score(mlp_model, X_train, y_train, cv=kf, scoring='accuracy')
        score_gradient = cross_val_score(gradient, X_train, y_train, cv=kf, scoring='accuracy')

        random_forest.append(score_rf.mean())
        svm.append(score_svm.mean())
        knn.append(score_knn.mean())
        logistic_regression.append(score_log.mean())
        mlp.append(score_mlp.mean())
        gradient_boosting.append(score_gradient.mean())
    return random_forest, svm, knn, logistic_regression, mlp, gradient_boosting 

