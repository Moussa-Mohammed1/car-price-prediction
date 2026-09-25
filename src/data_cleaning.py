import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

def handle_missing_values(df: pd.DataFrame):
    df = df.copy()

    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())

    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    return df 

def remove_duplicates(df: pd.DataFrame):
    return df.drop_duplicates().reset_index(drop=True)

def remove_outliers_iqr(df: pd.DataFrame, columns):
    df = df.copy()

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    return df.reset_index(drop=True)

def map_owner(df: pd.DataFrame):
    owner_mapping = {
        'Test Drive Car': 0,
        'First Owner': 1,
        'Second Owner': 2,
        'Third Owner': 3,
        'Fourth & Above Owner': 4
    }

    df = df.copy()
    df['owner'] = df['owner'].map(owner_mapping)
    return df

def drop_unnecessary(df: pd.DataFrame, columns):
    df = df.drop(columns=columns)
    return df 