import pandas as pd

df = pd.read_csv('raw_github_repo_data.csv')

print(df.columns[df.nunique(dropna=False) == 1])

print("--------------- First Row Original of Data ---------------")
for column, data in zip(df.columns, df.iloc[0]):
    print(column, ": ", data)

# Drop watchers and watchers_count columns (identical to target column)
df_processed = df.drop(['watchers', 'watchers_count'], axis=1)
# Drop url columns
df_processed = df_processed.drop(columns=df.filter(regex='url$').columns, axis=1)
# Drop non-quantifieable columns
df_processed = df_processed.drop(['node_id', 'name', 'full_name', 'owner', 'description', 'homepage', 'license', 'topics', 'default_branch', 'permissions', 'custom_properties', 'template_repository', 'organization'], axis=1)
# Drop column with same values for all data
df_processed = df_processed.drop(df_processed.columns[df_processed.nunique(dropna=False) == 1], axis=1)

# Translate language column to one hot encoding
for language in df_processed['language'].dropna().unique().tolist():
    df_processed['language_' + language] = (df_processed['language'] == language).astype(int)
df_processed = df_processed.drop('language', axis=1)

# Convert boolean values to their respective integer value
df_processed[df_processed.select_dtypes('bool').columns] = df_processed.select_dtypes('bool').astype(int)

# Convert time into their numerical value in seconds
df_processed['created_at'] = pd.to_datetime(df_processed['created_at']).astype('int64') // 10**9
df_processed['updated_at'] = pd.to_datetime(df_processed['updated_at']).astype('int64') // 10**9
df_processed['pushed_at'] = pd.to_datetime(df_processed['pushed_at']).astype('int64') // 10**9

print("--------------- First Row of Processed Data ---------------")
for column, data in zip(df_processed.columns, df_processed.iloc[0]):
    print(column, ": ", data)

# Save to CSV
filename = 'processed_github_repo_data.csv'
df_processed.to_csv(filename, index=False)
print(f"Saved {len(df_processed)} processed repositories to '{filename}'")