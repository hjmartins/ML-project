import sys
from preprocess import load_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sys.path.append("../src")
# Load the dataset
df = load_data(r'C:/Users/hjmar/Documents/Vscode/ML-project/data/census.csv')

# Display the first few rows of the dataframe
print(df.head())

# Get a summary of the dataframe
print(df.describe())

# Check for missing values
print(df.isnull().sum())    

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    print(f"Value counts for {col}:")
    print(df[col].value_counts())
    print("\n")