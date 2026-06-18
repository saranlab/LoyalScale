import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from mapie.classification import SplitConformalClassifier

# Base directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = r"C:\Users\Saran\Documents\forMock\mock_churn_data"
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data')

os.makedirs(OUTPUT_DIR, exist_ok=True)

INDUSTRIES = ['telecom', 'saas', 'retail', 'banking', 'ecommerce', 'education', 'healthcare', 'hospitality', 'insurance', 'utilities']

def get_feature_types(df):
    """Dynamically determine numeric and categorical features."""
    cols_to_exclude = ['customer_id', 'industry', 'churn_probability', 'churned']
    features = [c for c in df.columns if c not in cols_to_exclude]
    
    numeric_features = []
    categorical_features = []
    
    # Register all binary flags that need One-Hot Encoding
    binary_categorical_cols = [
        'autopay_enabled', 'device_financed', 'international_roaming', 'onboarding_completed',
        'free_shipping_member', 'primary_provider_assigned', 'smart_meter_enabled', 'paperless_billing', 'move_flag_90d'
    ]
    
    for col in features:
        if df[col].dtype == 'object' or df[col].dtype.name == 'category':
            categorical_features.append(col)
        else:
            # Check if integer but behaves like a categorical feature (e.g. low cardinality binary)
            if df[col].nunique() <= 2 and col in binary_categorical_cols:
                categorical_features.append(col)
            else:
                numeric_features.append(col)
                
    return numeric_features, categorical_features

def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )

def train_industry(industry):
    print(f"\n==================== Training {industry.upper()} Model ====================")
    
    # 1. Load data
    train_path = os.path.join(DATA_DIR, 'train', f'{industry}_churn_train.csv')
    val_path = os.path.join(DATA_DIR, 'val', f'{industry}_churn_val.csv')
    test_path = os.path.join(DATA_DIR, 'test', f'{industry}_churn_test_features.csv')
    test_ans_path = os.path.join(DATA_DIR, 'test_answer_key', f'{industry}_churn_test_answer_key.csv')
    
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    df_test = pd.read_csv(test_path)
    df_test_ans = pd.read_csv(test_ans_path)
    
    # Identify target
    y_train = df_train['churned'].values.ravel()
    y_val = df_val['churned'].values.ravel()
    y_test = df_test_ans['churned'].values.ravel()
    
    # Determine features
    numeric_features, categorical_features = get_feature_types(df_train)
    print(f"Numerical features ({len(numeric_features)}): {numeric_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Drop irrelevant columns
    cols_to_drop = ['customer_id', 'industry', 'churn_probability', 'churned']
    X_train_raw = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
    X_val_raw = df_val.drop(columns=[c for c in cols_to_drop if c in df_val.columns])
    X_test_raw = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
    
    # 2. Fit and apply preprocessing
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test_raw)
    
    # Get column names after transformation
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
    all_transformed_cols = numeric_features + encoded_cat_cols
    
    # Make DataFrames
    X_train = pd.DataFrame(X_train_trans, columns=all_transformed_cols)
    X_val = pd.DataFrame(X_val_trans, columns=all_transformed_cols)
    X_test = pd.DataFrame(X_test_trans, columns=all_transformed_cols)
    
    # Save processed splits for debugging/conformal stats
    X_train.to_csv(os.path.join(OUTPUT_DIR, f'X_train_processed_{industry}.csv'), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, f'X_test_processed_{industry}.csv'), index=False)
    pd.DataFrame(y_train, columns=['churned']).to_csv(os.path.join(OUTPUT_DIR, f'y_train_{industry}.csv'), index=False)
    pd.DataFrame(y_test, columns=['churned']).to_csv(os.path.join(OUTPUT_DIR, f'y_test_{industry}.csv'), index=False)
    
    # 3. Model Hyperparameter Tuning
    print(f"Tuning base estimators for {industry}...")
    
    # Train fit split for randomized search
    X_fit, X_tuning_cal, y_fit, y_tuning_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # XGBoost
    xgb_clf = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
    xgb_grid = {
        'n_estimators': [100, 150],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.08, 0.12],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    xgb_search = RandomizedSearchCV(xgb_clf, xgb_grid, n_iter=8, cv=3, random_state=42, n_jobs=-1)
    xgb_search.fit(X_fit, y_fit)
    xgb_best = xgb_search.best_estimator_
    
    # LightGBM
    lgb_clf = lgb.LGBMClassifier(random_state=42, verbose=-1)
    lgb_grid = {
        'n_estimators': [100, 150],
        'max_depth': [4, 5, 6],
        'learning_rate': [0.05, 0.08, 0.12],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    lgb_search = RandomizedSearchCV(lgb_clf, lgb_grid, n_iter=8, cv=3, random_state=42, n_jobs=-1)
    lgb_search.fit(X_fit, y_fit)
    lgb_best = lgb_search.best_estimator_
    
    # CatBoost
    cb_clf = cb.CatBoostClassifier(random_state=42, verbose=0)
    cb_grid = {
        'iterations': [100, 150],
        'depth': [4, 5, 6],
        'learning_rate': [0.05, 0.08, 0.12],
        'subsample': [0.8, 1.0]
    }
    cb_search = RandomizedSearchCV(cb_clf, cb_grid, n_iter=8, cv=3, random_state=42, n_jobs=-1)
    cb_search.fit(X_fit, y_fit)
    cb_best = cb_search.best_estimator_
    
    # 4. Stacking Classifier Ensemble
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    
    ensemble = StackingClassifier(
        estimators=[
            ('xgb', xgb_best),
            ('lgb', lgb_best),
            ('cb', cb_best)
        ],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )
    # Fit the ensemble on the entire training set
    ensemble.fit(X_train, y_train)
    
    # 5. Conformal Calibration (MAPIE) using the separate validation set
    print(f"Calibrating conformal predictor using separate validation set ({len(X_val)} rows)...")
    confidence_levels = [0.80, 0.85, 0.90, 0.95]
    mapie_model = SplitConformalClassifier(estimator=ensemble, confidence_level=confidence_levels, prefit=True)
    mapie_model.conformalize(X_val, y_val)
    
    # 6. Evaluate
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    
    print(f"\n--- {industry.upper()} Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Save artifacts
    preprocessor_path = os.path.join(OUTPUT_DIR, f'preprocessor_{industry}.joblib')
    model_path = os.path.join(OUTPUT_DIR, f'model_{industry}.joblib')
    mapie_model_path = os.path.join(OUTPUT_DIR, f'mapie_model_{industry}.joblib')
    
    joblib.dump(preprocessor, preprocessor_path)
    joblib.dump(ensemble, model_path)
    joblib.dump(mapie_model, mapie_model_path)
    
    print(f"Saved preprocessor to: {preprocessor_path}")
    print(f"Saved model to: {model_path}")
    print(f"Saved mapie model to: {mapie_model_path}")

def main():
    for ind in INDUSTRIES:
        train_industry(ind)
    print("\nAll industries trained successfully!")

if __name__ == '__main__':
    main()
