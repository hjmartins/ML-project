# Handling missing values, outliers, and inconsistencies.
# Transforming data into a suitable format for model training.

from preprocess import load_data
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def data_stats(df):
    # see the data
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.duplicated().sum())
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"Value counts for {col}:")
        print(df[col].value_counts())
        print("\n")

    mask = (df == ' ?').sum()
    mask = mask[mask > 0]
    print(mask)
    print(f"Total de valores ausentes: {(df == ' ?').sum().sum()}")


    with open ('../reports/data/cleaning_report.txt', 'w') as f:
        f.write("Cleaning Report\n")
        f.write(df.head().to_string() + "\n")
        f.write("====================\n")
        f.write(df.describe().to_string() + "\n")
        f.write(df.info() + "\n")
        f.write("================\n")
        f.write(df.isnull().sum().to_string() + "\n")
        f.write("================\n")
        f.write(df.duplicated().sum().to_string() + "\n")
        f.write("================\n")
        for col in categorical_cols:
            f.write(f"Value counts for {col}:\n")
            f.write(df[col].value_counts().to_string() + "\n")
        f.write("====================\n")
        f.write(mask.to_string() + "\n")
        f.write(f"Total de valores ausentes: {(df == ' ?').sum().sum()}\n")
        f.write("====================\n")
#not finished
def cleaning(df):
    
    df.replace(' ?', np.nan, inplace=True)
    df.dropna(inplace=True)

    
    df.drop_duplicates(inplace=True)

    
    df = pd.get_dummies(df, drop_first=True)

    return df