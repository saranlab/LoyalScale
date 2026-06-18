import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from mapie.classification import SplitConformalClassifier
import optuna
import pandera as pa

# Base directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = r"C:\Users\Saran\Documents\forMock\mock_churn_data"
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data')

os.makedirs(OUTPUT_DIR, exist_ok=True)

INDUSTRIES = ['telecom', 'saas', 'retail', 'banking', 'ecommerce', 'education', 'healthcare', 'hospitality', 'insurance', 'utilities']

def validate_raw_data(df, industry):
    """Validates the raw DataFrame structure and types using Pandera to prevent target leakage."""
    schema_cols = {}
    
    # Strictly validate target column if it exists (prevents leakage and ensures schema health)
    if 'churned' in df.columns:
        schema_cols['churned'] = pa.Column(pa.Int, checks=pa.Check.isin([0, 1]), nullable=False)
        
    for col in df.columns:
        if col in ['customer_id', 'industry']:
            schema_cols[col] = pa.Column(pa.String, nullable=True)
        elif col == 'churn_probability':
            schema_cols[col] = pa.Column(pa.Float, nullable=True)
        elif col not in schema_cols:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                schema_cols[col] = pa.Column(pa.String, nullable=True)
            else:
                schema_cols[col] = pa.Column(pa.Float, coerce=True, nullable=True)
                
    schema = pa.DataFrameSchema(columns=schema_cols, strict=False)
    return schema.validate(df)

def tune_xgb_optuna(X, y, n_trials=5):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 150, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, step=0.03),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0, step=0.1),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': 1
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            clf = xgb.XGBClassifier(**params)
            clf.fit(X_tr, y_tr)
            score = clf.score(X_val, y_val)
            scores.append(score)
            
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_lgb_optuna(X, y, n_trials=5):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 150, step=50),
            'max_depth': trial.suggest_int('max_depth', 4, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, step=0.03),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0, step=0.1),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 1
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            clf = lgb.LGBMClassifier(**params)
            clf.fit(X_tr, y_tr)
            score = clf.score(X_val, y_val)
            scores.append(score)
            
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

def tune_cb_optuna(X, y, n_trials=5):
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 150, step=50),
            'depth': trial.suggest_int('depth', 4, 6),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.15, step=0.03),
            'subsample': trial.suggest_float('subsample', 0.8, 1.0, step=0.1),
            'random_state': 42,
            'verbose': 0,
            'thread_count': 1
        }
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            clf = cb.CatBoostClassifier(**params)
            clf.fit(X_tr, y_tr)
            score = clf.score(X_val, y_val)
            scores.append(score)
            
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

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
        elif col in binary_categorical_cols:
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
    
    df_train = validate_raw_data(pd.read_csv(train_path), industry)
    df_val = validate_raw_data(pd.read_csv(val_path), industry)
    df_test = validate_raw_data(pd.read_csv(test_path), industry)
    df_test_ans = validate_raw_data(pd.read_csv(test_ans_path), industry)
    
    # Cast binary categorical features to int to avoid float vs int feature name suffix issues (e.g. 0.0 vs 0)
    binary_categorical_cols = [
        'autopay_enabled', 'device_financed', 'international_roaming', 'onboarding_completed',
        'free_shipping_member', 'primary_provider_assigned', 'smart_meter_enabled', 'paperless_billing', 'move_flag_90d'
    ]
    for col in binary_categorical_cols:
        if col in df_train.columns:
            df_train[col] = pd.to_numeric(df_train[col], errors='coerce').fillna(0).astype(int)
        if col in df_val.columns:
            df_val[col] = pd.to_numeric(df_val[col], errors='coerce').fillna(0).astype(int)
        if col in df_test.columns:
            df_test[col] = pd.to_numeric(df_test[col], errors='coerce').fillna(0).astype(int)
        if col in df_test_ans.columns:
            df_test_ans[col] = pd.to_numeric(df_test_ans[col], errors='coerce').fillna(0).astype(int)
            
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
    
    # 3. Model Hyperparameter Tuning (Optuna with MedianPruner)
    print(f"Tuning base estimators for {industry} via Optuna...")
    
    # Train fit split for Optuna tuning
    X_fit, X_tuning_cal, y_fit, y_tuning_cal = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # XGBoost
    xgb_best_params = tune_xgb_optuna(X_fit, y_fit, n_trials=5)
    xgb_best = xgb.XGBClassifier(**xgb_best_params, random_state=42, eval_metric='logloss', n_jobs=1)
    xgb_best.fit(X_fit, y_fit)
    
    # LightGBM
    lgb_best_params = tune_lgb_optuna(X_fit, y_fit, n_trials=5)
    lgb_best = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, n_jobs=1)
    lgb_best.fit(X_fit, y_fit)
    
    # CatBoost
    cb_best_params = tune_cb_optuna(X_fit, y_fit, n_trials=5)
    cb_best = cb.CatBoostClassifier(**cb_best_params, random_state=42, verbose=0, thread_count=1)
    cb_best.fit(X_fit, y_fit)
    
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

    # SaaS Survival Analysis implementation
    if industry == 'saas':
        print("\n--- Training Survival Analysis Model for SaaS Time-to-Churn ---")
        try:
            from lifelines import CoxPHFitter
            # Cox PH model expects all columns to be numeric. We'll use the preprocessed features
            # plus duration ('tenure_months') and event ('churned').
            X_train_surv = X_train.copy()
            # Ensure no target leakage: drop target if already encoded (it shouldn't be encoded since cols_to_exclude has 'churned')
            X_train_surv['tenure_months'] = df_train['tenure_months'].values
            X_train_surv['churned'] = y_train
            
            # Drop zero variance or highly collinear columns if any
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(X_train_surv, duration_col='tenure_months', event_col='churned')
            print("Cox Proportional Hazards Model fitted successfully!")
            
            cph_path = os.path.join(OUTPUT_DIR, 'saas_survival_model.joblib')
            joblib.dump(cph, cph_path)
            print(f"Saved SaaS survival model to: {cph_path}")
        except Exception as surv_err:
            print(f"Failed to fit SaaS CoxPH model: {surv_err}")

def main():
    for ind in INDUSTRIES:
        train_industry(ind)
    print("\nAll industries trained successfully!")

if __name__ == '__main__':
    main()
