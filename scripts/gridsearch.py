import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from joblib import dump
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("../data/processed_github_repo_data.csv")

# Define target and drop unnecessary columns
target = 'stargazers_count'
drop_cols = ['id']
df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True, errors='ignore')

# Separate features and target
X = df.drop(columns=[target])
y = df[target]

# Print the list of features being used
print("\nFeatures used for training:")
print(X.columns.tolist())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create the model
model = RandomForestRegressor(random_state=42)

# Grid of hyperparameters
param_grid = {
    'n_estimators': [50, 100, 200, 500, 1000],
    'max_depth': [None, 2, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],
    'bootstrap': [True, False],
    'ccp_alpha': [0.0, 0.001, 0.005, 0.01, 0.05]
}

# Perform gridsearch
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, verbose=2, scoring='r2')
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"  R²:   {r2:.4f}")
print(f"  MAE:  {mae:.4f}")
print(f"  RMSE: {rmse:.4f}")

# Save best model in models directory
dump(best_model, "../models/best_model.pkl")
print(f"\nBest model saved to: ../models/best_model.pkl")
