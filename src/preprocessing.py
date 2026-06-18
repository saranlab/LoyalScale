import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import joblib

def load_data(filepath: str) -> pd.DataFrame:
    """Loads the dataset from the CSV file path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the raw Telco Churn DataFrame.
    - Handles spaces in 'TotalCharges' by casting to float and filling NaNs with 0.0 (since tenure is 0).
    - Encodes the target 'Churn' to binary integer values (Yes -> 1, No -> 0).
    - Drops the 'customerID' column.
    """
    df_clean = df.copy()
    
    # 1. Handle TotalCharges missing values
    # Replace empty string or whitespace with NaN, then convert to float
    df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # New customers with tenure = 0 have no TotalCharges. Impute with 0.0.
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(0.0)
    
    # 2. Encode target 'Churn' (if exists in df)
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        
    # 3. Drop customerID if exists
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        
    return df_clean

def get_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    """
    Returns a ColumnTransformer containing:
    - Numeric transformation: Imputation followed by standard scaling.
    - Categorical transformation: Imputation followed by One-Hot encoding.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def save_pipeline_artifacts(X_train_proc: pd.DataFrame, X_test_proc: pd.DataFrame, 
                            y_train: pd.Series, y_test: pd.Series, 
                            preprocessor: ColumnTransformer, output_dir: str = 'processed_data'):
    """Saves the preprocessed datasets and the fitted preprocessor object to disk."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save datasets
    X_train_proc.to_csv(os.path.join(output_dir, 'X_train_processed.csv'), index=False)
    X_test_proc.to_csv(os.path.join(output_dir, 'X_test_processed.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    
    # Save the fitted preprocessor
    joblib.dump(preprocessor, os.path.join(output_dir, 'preprocessor.joblib'))
    print(f"Preprocessed data and pipeline artifacts saved to '{output_dir}/'")
