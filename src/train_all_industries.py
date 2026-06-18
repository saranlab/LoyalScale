import os
import sys

# Base directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

import pandas as pd
import numpy as np
import joblib
import optuna
import pandera as pa
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from mapie.classification import SplitConformalClassifier

DATA_DIR = r"C:\Users\Saran\Documents\forMock\mock_churn_data"
OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data')

os.makedirs(OUTPUT_DIR, exist_ok=True)

INDUSTRIES = ['telecom', 'saas', 'retail', 'banking', 'ecommerce', 'education', 'healthcare', 'hospitality', 'insurance', 'utilities']

# Define feature categorization for strict type checking
CONTINUOUS_FEATURES = {
    'monthly_spend_usd', 'discount_pct', 'data_usage_gb_30d', 'dropped_calls_30d',
    'avg_basket_usd', 'returns_90d', 'avg_balance_usd', 'products_count',
    'cart_abandon_rate', 'avg_order_value_usd', 'return_rate', 'completion_rate',
    'logins_30d', 'assignments_late_90d', 'appointments_12m', 'missed_appointments_12m',
    'avg_nightly_rate_usd', 'review_rating', 'reward_points_balance', 'cancellations_12m',
    'premium_usd', 'claims_24m', 'policy_count', 'renewal_days_remaining',
    'avg_monthly_usage', 'outages_12m', 'feature_adoption_score'
}

DISCRETE_FEATURES = {
    'signup_year', 'age', 'tenure_months', 'support_tickets_90d', 'complaints_90d',
    'nps_score', 'days_since_last_activity', 'late_payments_12m', 'seats_purchased',
    'active_users_30d', 'integrations_connected', 'admin_logins_30d', 'visits_90d',
    'coupons_used_90d', 'mobile_logins_30d', 'overdrafts_12m', 'branch_visits_90d',
    'orders_180d', 'app_sessions_30d', 'courses_enrolled', 'advisor_contacts_90d',
    'portal_logins_90d', 'care_gap_count', 'stays_12m', 'agent_contact_90d',
    'network_complaints_90d'
}

BINARY_FEATURES = {
    'autopay_enabled', 'device_financed', 'international_roaming', 'onboarding_completed',
    'free_shipping_member', 'primary_provider_assigned', 'smart_meter_enabled',
    'paperless_billing', 'move_flag_90d'
}

STRING_FEATURES = {
    'region', 'customer_segment', 'contract_type', 'acquisition_channel', 'plan_type',
    'loyalty_tier', 'store_preference', 'account_type', 'program_type', 'plan_category',
    'membership_level', 'policy_type', 'service_type'
}

def get_pandera_schema(industry: str) -> pa.DataFrameSchema:
    """
    Creates a strict Pandera validation schema based on the industry's expected schema.
    Enforces appropriate data types (String, Float) and value ranges.
    """
    from src.nlp_mapper import INDUSTRY_SCHEMAS
    
    expected_features = INDUSTRY_SCHEMAS.get(industry, [])
    schema_cols = {}
    
    # Target column is required to be binary integer (0 or 1)
    schema_cols['churned'] = pa.Column(
        pa.Int, 
        checks=pa.Check.isin([0, 1]), 
        coerce=True, 
        nullable=False,
        description="Binary churn indicator (target)"
    )
    
    # Enforce correct types and values for each expected feature
    for feature in expected_features:
        if feature in CONTINUOUS_FEATURES or feature in DISCRETE_FEATURES or feature in BINARY_FEATURES:
            schema_cols[feature] = pa.Column(pa.Float, coerce=True, nullable=True)
        elif feature in STRING_FEATURES:
            schema_cols[feature] = pa.Column(pa.String, coerce=True, nullable=True)
        else:
            # Fallback type coercion
            schema_cols[feature] = pa.Column(pa.Float, coerce=True, nullable=True)
            
    return pa.DataFrameSchema(columns=schema_cols, strict=True)

def validate_raw_data(df: pd.DataFrame, industry: str) -> pd.DataFrame:
    """
    Performs data cleaning to eliminate target leakage and applies the strict Pandera schema.
    """
    df_clean = df.copy()
    
    # 1. Eliminate Target Leakage: Drop raw identifiers, metadata, and proxy targets
    leaking_cols = ['churn_probability', 'customer_id', 'industry', 'accidental_raw_id', 'id', 'customerID']
    df_clean = df_clean.drop(columns=[col for col in leaking_cols if col in df_clean.columns], errors='ignore')
    
    # 2. Get the validation schema
    schema = get_pandera_schema(industry)
    
    # If target is missing (e.g. test features dataset), dynamically adapt validation schema
    if 'churned' not in df_clean.columns:
        schema = pa.DataFrameSchema(
            columns={k: v for k, v in schema.columns.items() if k != 'churned'},
            strict=True
        )
        
    # Ensure all expected features are present in the df to prevent validation failures (fill with NaN)
    for col in schema.columns:
        if col not in df_clean.columns:
            df_clean[col] = np.nan
            
    # Reorder columns to match the validation schema order exactly
    df_clean = df_clean[list(schema.columns.keys())]
            
    # 3. Perform validation
    return schema.validate(df_clean)

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
    """Creates a modular scikit-learn ColumnTransformer for preprocessing."""
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

def tune_stacking_optuna(X_raw, y, numeric_features, categorical_features, n_trials=5):
    """
    Simultaneously tunes hyperparameters for XGBoost, LightGBM, and CatBoost inside a
    single unified Optuna study. Prevents feature leakage by executing fit/transform 
    within each fold partition separately.
    """
    def objective(trial):
        # XGBoost parameters
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 150, step=50),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 6),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.05, 0.20, step=0.05),
            'subsample': trial.suggest_float('xgb_subsample', 0.7, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.7, 1.0, step=0.1),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': 1  # Prevent CPU/thread oversubscription inside Optuna
        }
        
        # LightGBM parameters
        lgb_params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 150, step=50),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 6),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.05, 0.20, step=0.05),
            'subsample': trial.suggest_float('lgb_subsample', 0.7, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.7, 1.0, step=0.1),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 1  # Prevent CPU/thread oversubscription inside Optuna
        }
        
        # CatBoost parameters
        cb_params = {
            'iterations': trial.suggest_int('cb_iterations', 50, 150, step=50),
            'depth': trial.suggest_int('cb_depth', 3, 6),
            'learning_rate': trial.suggest_float('cb_learning_rate', 0.05, 0.20, step=0.05),
            'subsample': trial.suggest_float('cb_subsample', 0.7, 1.0, step=0.1),
            'random_state': 42,
            'verbose': 0,
            'thread_count': 1  # Prevent CPU/thread oversubscription inside Optuna
        }
        
        # 3-Fold Cross Validation
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw, y)):
            X_tr_raw, X_val_raw = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # FIT ColumnTransformer ONLY on the training fold (Strict Leakage Prevention)
            preprocessor = build_preprocessor(numeric_features, categorical_features)
            X_tr = preprocessor.fit_transform(X_tr_raw)
            X_val = preprocessor.transform(X_val_raw)
            
            xgb_clf = xgb.XGBClassifier(**xgb_params)
            lgb_clf = lgb.LGBMClassifier(**lgb_params)
            cb_clf = cb.CatBoostClassifier(**cb_params)
            
            # Fit Stacking ensemble
            ensemble = StackingClassifier(
                estimators=[('xgb', xgb_clf), ('lgb', lgb_clf), ('cb', cb_clf)],
                final_estimator=LogisticRegression(),
                cv=3,
                n_jobs=1
            )
            
            ensemble.fit(X_tr, y_tr)
            y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)
            
            # Prune poor trials early using MedianPruner
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return float(np.mean(scores))
        
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=n_trials)
    return study.best_params

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
    
    # Drop irrelevant target/leakage columns
    cols_to_drop = ['customer_id', 'industry', 'churn_probability', 'churned']
    X_train_raw = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
    X_val_raw = df_val.drop(columns=[c for c in cols_to_drop if c in df_val.columns])
    X_test_raw = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
    
    # 2. Hyperparameter Tuning using Unified Optuna Stacking study
    print(f"Running unified Optuna tuning for {industry} base estimators...")
    
    # Split train raw split to prevent leakage during tuning
    X_fit_raw, X_tuning_cal_raw, y_fit, y_tuning_cal = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    best_params = tune_stacking_optuna(X_fit_raw, y_fit, numeric_features, categorical_features, n_trials=5)
    
    # Extract optimal hyperparameter dictionaries
    xgb_best_params = {
        'n_estimators': best_params['xgb_n_estimators'],
        'max_depth': best_params['xgb_max_depth'],
        'learning_rate': best_params['xgb_learning_rate'],
        'subsample': best_params['xgb_subsample'],
        'colsample_bytree': best_params['xgb_colsample_bytree'],
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': 1
    }
    lgb_best_params = {
        'n_estimators': best_params['lgb_n_estimators'],
        'max_depth': best_params['lgb_max_depth'],
        'learning_rate': best_params['lgb_learning_rate'],
        'subsample': best_params['lgb_subsample'],
        'colsample_bytree': best_params['lgb_colsample_bytree'],
        'random_state': 42,
        'verbose': -1,
        'n_jobs': 1
    }
    cb_best_params = {
        'iterations': best_params['cb_iterations'],
        'depth': best_params['cb_depth'],
        'learning_rate': best_params['cb_learning_rate'],
        'subsample': best_params['cb_subsample'],
        'random_state': 42,
        'verbose': 0,
        'thread_count': 1
    }
    
    # 3. Fit and apply final preprocessing (fit preprocessor ONLY on X_train_raw)
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_trans = preprocessor.fit_transform(X_train_raw)
    X_val_trans = preprocessor.transform(X_val_raw)
    X_test_trans = preprocessor.transform(X_test_raw)
    
    # Retrieve column names dynamically after OHE
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
    
    # 4. Train final Stacking Classifier Ensemble
    xgb_best = xgb.XGBClassifier(**xgb_best_params)
    lgb_best = lgb.LGBMClassifier(**lgb_best_params)
    cb_best = cb.CatBoostClassifier(**cb_best_params)
    
    ensemble = StackingClassifier(
        estimators=[('xgb', xgb_best), ('lgb', lgb_best), ('cb', cb_best)],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )
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

    # 7. Framework for Survival Analysis (SaaS & Telecom Deployment)
    if industry in ['saas', 'telecom']:
        print(f"\n--- Training Survival Analysis Model for {industry.upper()} Time-to-Churn ---")
        try:
            from lifelines import CoxPHFitter
            
            # Prepare survival data: use preprocessed features + tenure & churn target
            X_train_surv = X_train.copy()
            X_train_surv['tenure_months'] = df_train['tenure_months'].values
            X_train_surv['churned'] = y_train
            
            # Drop zero variance columns to prevent convergence/singular matrix failures
            non_surv_cols = [c for c in X_train_surv.columns if c not in ['tenure_months', 'churned']]
            zero_var_cols = [c for c in non_surv_cols if X_train_surv[c].var() == 0]
            if zero_var_cols:
                print(f"Dropping zero-variance features for survival: {zero_var_cols}")
                X_train_surv = X_train_surv.drop(columns=zero_var_cols)
                
            # Fit Cox Proportional Hazards model with L2 regularization
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(X_train_surv, duration_col='tenure_months', event_col='churned')
            print("Cox Proportional Hazards Model fitted successfully!")
            
            # Print top features based on Hazard Ratios
            summary_df = cph.summary.sort_values(by='p')
            print("Survival Model Hazard Ratios (Top 5 significant features):")
            print(summary_df[['coef', 'exp(coef)', 'p']].head(5))
            
            # Save survival model to disk
            cph_path = os.path.join(OUTPUT_DIR, f'{industry}_survival_model.joblib')
            joblib.dump(cph, cph_path)
            print(f"Saved {industry.upper()} survival model to: {cph_path}")
            
            # NOTE: Conformal Calibration Note
            # The survival curves outputted by `cph.predict_survival_function(X)` can be evaluated at 
            # milestone intervals (e.g. 12, 24, 36 months) to yield a churn risk trajectory over time,
            # which can subsequently be fed to our SplitConformalClassifier to produce calibrated 
            # prediction intervals for time-to-churn milestones.
        except Exception as surv_err:
            print(f"Failed to fit {industry} CoxPH model: {surv_err}")

def main():
    for ind in INDUSTRIES:
        train_industry(ind)
    print("\nAll industries trained successfully!")

if __name__ == '__main__':
    main()
