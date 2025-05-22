from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os
import numpy as np

app = Flask(__name__)

# Load the model
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "best_model.pkl")
model = joblib.load(MODEL_PATH)

# Model fields (deduplicated)
model_fields = list(dict.fromkeys(model.feature_names_in_))

# Extract language-related fields
language_columns = [col for col in model_fields if col.startswith("language_")]

# Non-feature fields to retain for display
identifier_cols = ['full_name', 'html_url']

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()
    df = pd.DataFrame(input_data)

    # Step 1: Retain metadata for display
    df_meta = df[identifier_cols] if all(col in df.columns for col in identifier_cols) else pd.DataFrame()

    # Step 2: One-hot encode language column (if present)
    if 'language' in df.columns:
        lang_dummies = pd.get_dummies(df['language'], prefix='language')
        df = pd.concat([df.drop(columns=['language']), lang_dummies], axis=1)

    # Step 3: Automatically compute log_forks and log_issues if possible
    if 'forks_count' in df.columns:
        df['log_forks'] = np.log1p(df['forks_count'])  # log(1 + forks_count)
    if 'open_issues_count' in df.columns:
        df['log_issues'] = np.log1p(df['open_issues_count'])  # log(1 + open_issues_count)

    # Step 4: Add missing language_XXX columns with 0
    for lang_col in language_columns:
        if lang_col not in df.columns:
            df[lang_col] = 0

    # Step 5: Fill missing feature fields with 0
    for field in model_fields:
        if field not in df.columns:
            df[field] = 0

    # Step 6: Reorder columns to match model input
    df_features = df[model_fields]

    # Step 7: Model prediction
    df['predicted_stars'] = model.predict(df_features)

    # Step 8: Merge prediction with metadata BEFORE sorting
    if not df_meta.empty:
        df_result = pd.concat([df, df_meta], axis=1)
    else:
        df_result = df

    # Step 9: Sort and assign rank
    df_sorted = df_result.sort_values(by='predicted_stars', ascending=False).reset_index(drop=True)
    df_sorted['rank'] = df_sorted.index + 1

    # Step 10: Return selected fields
    if not df_meta.empty:
        df_sorted = df_sorted[['full_name', 'html_url', 'predicted_stars', 'rank']]
    else:
        df_sorted = df_sorted[['predicted_stars', 'rank']]

    return jsonify(df_sorted.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
