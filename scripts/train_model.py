import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("../data/processed_github_repo_data.csv")

# Define target and drop unnecessary columns
target = 'stargazers_count'
drop_cols = ['id', 'watchers', 'watchers_count', 'created_at', 'updated_at', 'pushed_at']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=["object", "bool"]).columns.tolist()
numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Define preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder='passthrough'
)

# Define models to evaluate
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize tracking variables
best_score = -np.inf
best_model = None

# Train and evaluate each model
for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    print(f"Model: {name}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print("-" * 30)

    if r2 > best_score:
        best_score = r2
        best_model = pipeline

# Save best model in models directory
dump(best_model, "../models/best_model.pkl")
print(f"\nBest model saved to: ../models/best_model.pkl")
print(f"Best model is {type(best_model.named_steps['regressor']).__name__} with R² = {best_score:.4f}")

