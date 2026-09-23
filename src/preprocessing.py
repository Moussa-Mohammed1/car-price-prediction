import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def encode_categorical(df: pd.DataFrame, columns):
    return pd.get_dummies(df, columns=columns, drop_first=True)

def split_data(df: pd.DataFrame, target='selling_price', test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def scale_features(X_train, X_test, numeric_cols):
    scaler = StandardScaler()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    return X_train, X_test, scaler
