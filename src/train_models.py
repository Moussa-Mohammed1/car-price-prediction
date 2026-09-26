import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def build_pipeline(model, numeric_cols, X_train):
    from sklearn.compose import ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[('num', StandardScaler(), numeric_cols)],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def train_all_models(X_train, y_train, X_test, y_test, numeric_cols):
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestClassifier(),
        'XGBoost': XGBRegressor(),
        'SVR': SVR()
    }

    results = {}
    trained_models = {}

    for name, model in models.items():
        print(f"Entrainement: {name}")
        pipeline = build_pipeline(model, numeric_cols, X_train)
        pipeline.fit(X_train, y_train)
        scores = evaluate_model(pipeline, X_test, y_test)
        results[name] = scores
        trained_models[name] = pipeline
        print(f"scores: {scores}")
        results_df = pd.DataFrame(results).T
    return results_df, trained_models
    