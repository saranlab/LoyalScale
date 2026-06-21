import os
import sys
import logging

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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from mapie.classification import SplitConformalClassifier

# Setup production logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LoyalScalePipeline")

# 1. Dynamic Path Resolution
DATA_DIR = os.getenv('CHURN_DATA_DIR')
if not DATA_DIR:
    default_path = os.path.join(BASE_DIR, 'mock_churn_data')
    parent_dir = os.path.dirname(BASE_DIR)
    sibling_path = os.path.join(parent_dir, 'forMock', 'mock_churn_data')
    if os.path.exists(sibling_path):
        DATA_DIR = sibling_path
    else:
        DATA_DIR = default_path

logger.info(f"Resolved DATA_DIR dynamically to: {DATA_DIR}")

OUTPUT_DIR = os.path.join(BASE_DIR, 'processed_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

INDUSTRIES = ['telecom', 'saas', 'retail', 'banking', 'ecommerce', 'education', 'healthcare', 'hospitality', 'insurance', 'utilities']

class CalibrationQualityException(Exception):
    """Custom exception raised when empirical coverage deviates from target level by > 5%."""
    pass

class DataFrameCaster(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to convert a preprocessed numpy array back to 
    a Pandas DataFrame with correct dtypes, enabling native categorical handling 
    for LightGBM and CatBoost downstream.
    """
    def __init__(self, numeric_features, categorical_features, to_string=False):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.to_string = to_string
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        cols = self.numeric_features + self.categorical_features
        df = pd.DataFrame(X, columns=cols)
        
        # Enforce correct numerical dtypes
        for col in self.numeric_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # Enforce correct categorical dtypes
        for col in self.categorical_features:
            if self.to_string:
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype('category')
        return df

class TypeCaster(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to convert specific columns of a pandas DataFrame
    to a given data type (e.g. str for CatBoost, category for LightGBM).
    """
    def __init__(self, columns, to_type=str):
        self.columns = columns
        self.to_type = to_type
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].astype(self.to_type)
        return X

class SklearnCatBoostWrapper(BaseEstimator, ClassifierMixin):
    """
    Custom scikit-learn wrapper for CatBoostClassifier to ensure compatibility
    with sklearn.base.clone by storing parameters as attributes without modification.
    """
    def __init__(self, iterations=100, depth=6, learning_rate=0.03, subsample=None, random_state=None, verbose=0, thread_count=1, cat_features=None):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.verbose = verbose
        self.thread_count = thread_count
        self.cat_features = cat_features
        self.model_ = None
        
    def fit(self, X, y):
        params = {
            'iterations': self.iterations,
            'depth': self.depth,
            'learning_rate': self.learning_rate,
            'random_seed': self.random_state,
            'verbose': self.verbose,
            'thread_count': self.thread_count,
            'cat_features': self.cat_features
        }
        if self.subsample is not None:
            params['subsample'] = self.subsample
        self.model_ = cb.CatBoostClassifier(**params)
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        return self
        
    def predict(self, X):
        return self.model_.predict(X)
        
    def predict_proba(self, X):
        return self.model_.predict_proba(X)

def get_pandera_schema(industry: str, df: pd.DataFrame = None) -> pa.DataFrameSchema:
    """
    Creates a flexible Pandera validation schema based on the industry's expected schema
    and dynamic data types determined from the provided DataFrame.
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
    
    for feature in expected_features:
        if df is not None and feature in df.columns:
            # Determine type dynamically based on the DataFrame's parsed data type
            col_dtype = df[feature].dtype
            if pd.api.types.is_numeric_dtype(col_dtype):
                schema_cols[feature] = pa.Column(pa.Float, coerce=True, nullable=True)
            elif pd.api.types.is_string_dtype(col_dtype) or pd.api.types.is_object_dtype(col_dtype):
                schema_cols[feature] = pa.Column(pa.String, coerce=True, nullable=True)
            else:
                schema_cols[feature] = pa.Column(pa.Float, coerce=True, nullable=True)
        else:
            # Fallback based on column name heuristic
            is_str = any(term in feature.lower() for term in ['region', 'segment', 'type', 'channel', 'tier', 'preference', 'level', 'category'])
            if is_str:
                schema_cols[feature] = pa.Column(pa.String, coerce=True, nullable=True)
            else:
                schema_cols[feature] = pa.Column(pa.Float, coerce=True, nullable=True)
                
    return pa.DataFrameSchema(columns=schema_cols, strict=False)

def validate_raw_data(df: pd.DataFrame, industry: str) -> pd.DataFrame:
    """
    Performs data cleaning to eliminate target leakage and applies the flexible Pandera schema.
    """
    df_clean = df.copy()
    
    # 1. Eliminate Target Leakage: Drop raw identifiers, metadata, and proxy targets
    leaking_cols = ['churn_probability', 'customer_id', 'industry', 'accidental_raw_id', 'id', 'customerID']
    df_clean = df_clean.drop(columns=[col for col in leaking_cols if col in df_clean.columns], errors='ignore')
    
    # 2. Get the validation schema
    schema = get_pandera_schema(industry, df_clean)
    
    # If target is missing (e.g. test features dataset), dynamically adapt validation schema
    if 'churned' not in df_clean.columns:
        schema = pa.DataFrameSchema(
            columns={k: v for k, v in schema.columns.items() if k != 'churned'},
            strict=False
        )
        
    # Ensure all expected features are present in the df to prevent validation failures (fill with NaN)
    for col in schema.columns:
        if col not in df_clean.columns:
            df_clean[col] = np.nan
            
    # Note: Hardcoded column reordering is completely removed to allow flexibility.
    
    # 3. Perform validation
    return schema.validate(df_clean)

def get_feature_types(df, industry):
    """
    Dynamically determine numeric and categorical features.
    Strictly excludes 'tenure_months' from the classification features matrix (X)
    for survival-mode subscription industries (saas, telecom) to prevent target leakage.
    """
    cols_to_exclude = ['customer_id', 'industry', 'churn_probability', 'churned']
    if industry in ['saas', 'telecom']:
        cols_to_exclude.append('tenure_months')
        
    from src.nlp_mapper import INDUSTRY_SCHEMAS
    expected_features = INDUSTRY_SCHEMAS.get(industry, [])
    
    # Select only columns present in the DataFrame and expected for the given industry
    features = [c for c in df.columns if c not in cols_to_exclude and c in expected_features]
    
    categorical_names = {
        'region', 'customer_segment', 'contract_type', 'acquisition_channel', 'plan_type',
        'loyalty_tier', 'store_preference', 'account_type', 'program_type', 'plan_category',
        'membership_level', 'policy_type', 'service_type'
    }
    
    numeric_features = []
    categorical_features = []
    
    for col in features:
        if col in categorical_names:
            categorical_features.append(col)
        else:
            numeric_features.append(col)
                
    return numeric_features, categorical_features

def build_preprocessor(numeric_features, categorical_features):
    """
    Creates a preprocessor that imputes and scales numeric features,
    and imputes categorical features without One-Hot Encoding.
    This allows raw categorical features to be passed directly to downstream estimators.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='missing'))
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

def tune_stacking_optuna(X_raw, y, numeric_features, categorical_features, industry, n_trials=3):
    """
    Simultaneously tunes hyperparameters for XGBoost, LightGBM, and CatBoost inside a
    single unified Optuna study with resilient SQLite database storage.
    Uses a single validation split for outer evaluation to guarantee 3x computational speedups.
    """
    db_path = os.path.abspath(os.path.join(OUTPUT_DIR, f"optuna_study_{industry}.db"))
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except Exception as e:
            logger.warning(f"Could not remove old database {db_path}: {e}")

    storage_url = f"sqlite:///{db_path}"
    study_name = f"optuna_study_{industry}"

    # Split the tuning data once (Strict Leakage Protection)
    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )

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
            'n_jobs': 1
        }
        
        # LightGBM parameters
        lgb_params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 150, step=50),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 6),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.05, 0.20, step=0.05),
            'subsample': trial.suggest_float('lgb_subsample', 0.7, 1.0, step=0.1),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0, step=0.1),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 1
        }
        
        # CatBoost parameters
        cb_params = {
            'iterations': trial.suggest_int('cb_iterations', 50, 150, step=50),
            'depth': trial.suggest_int('cb_depth', 3, 6),
            'learning_rate': trial.suggest_float('cb_learning_rate', 0.05, 0.20, step=0.05),
            'subsample': trial.suggest_float('cb_subsample', 0.7, 1.0, step=0.1),
            'random_state': 42,
            'verbose': 0,
            'thread_count': 1
        }
        
        # Preprocessor for this fold only
        preprocessor = build_preprocessor(numeric_features, categorical_features)
        
        # Base models
        xgb_clf = xgb.XGBClassifier(**xgb_params)
        lgb_clf = lgb.LGBMClassifier(**lgb_params)
        cb_clf = SklearnCatBoostWrapper(
            iterations=cb_params['iterations'],
            depth=cb_params['depth'],
            learning_rate=cb_params['learning_rate'],
            subsample=cb_params['subsample'],
            random_state=cb_params['random_state'],
            verbose=cb_params['verbose'],
            thread_count=cb_params['thread_count'],
            cat_features=categorical_features
        )
        
        # Base estimators wrapped in pipelines with DataFrameCasters to preserve types
        xgb_pipe = Pipeline([
            ('caster', DataFrameCaster(numeric_features, categorical_features, to_string=True)),
            ('ohe', ColumnTransformer([('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)], remainder='passthrough')),
            ('xgb', xgb_clf)
        ])
        
        lgb_pipe = Pipeline([
            ('caster', DataFrameCaster(numeric_features, categorical_features, to_string=False)),
            ('lgb', lgb_clf)
        ])
        
        cb_pipe = Pipeline([
            ('caster', DataFrameCaster(numeric_features, categorical_features, to_string=True)),
            ('cb', cb_clf)
        ])
        
        ensemble = StackingClassifier(
            estimators=[('xgb', xgb_pipe), ('lgb', lgb_pipe), ('cb', cb_pipe)],
            final_estimator=LogisticRegression(),
            cv=3,
            n_jobs=1
        )
        
        # Unified fold pipeline
        fold_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('ensemble', ensemble)
        ])
        
        fold_pipeline.fit(X_tr_raw, y_tr)
        y_pred_proba = fold_pipeline.predict_proba(X_val_raw)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        return float(score)
        
    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='maximize',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner()
        )
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
    except Exception as e:
        logger.error(f"Optuna study hyperparameter optimization failed for {industry}: {str(e)}")
        raise RuntimeError(f"Failed Optuna tuning phase: {e}")

def clean_mapped_features(df, industry):
    """
    Cleans and casts raw values mapped from real-world datasets to match expected schemas.
    """
    from src.nlp_mapper import INDUSTRY_SCHEMAS
    from src.feature_bridge import DEFAULT_VALUES
    
    categorical_names = {
        'region', 'customer_segment', 'contract_type', 'acquisition_channel', 'plan_type',
        'loyalty_tier', 'store_preference', 'account_type', 'program_type', 'plan_category',
        'membership_level', 'policy_type', 'service_type'
    }
    
    cleaned_df = pd.DataFrame()
    if 'churned' in df.columns:
        cleaned_df['churned'] = df['churned']
    if 'customer_id' in df.columns:
        cleaned_df['customer_id'] = df['customer_id']
        
    for col in INDUSTRY_SCHEMAS[industry]:
        val_series = df.get(col)
        if val_series is None:
            # Column was not mapped/present → filled with default value
            cleaned_df[col] = [DEFAULT_VALUES.get(col)] * len(df)
            continue
            
        if col == 'signup_year':
            def extract_year(x):
                try:
                    return int(pd.to_datetime(x).year)
                except:
                    try:
                        return int(float(x))
                    except:
                        return int(DEFAULT_VALUES.get('signup_year', 2024))
            cleaned_df[col] = val_series.map(extract_year)
            
        elif col in categorical_names:
            # Leave categoricals as strings
            cleaned_df[col] = val_series.fillna(DEFAULT_VALUES.get(col, 'unknown')).astype(str)
            
        else:
            # Numeric/Discrete/Continuous/Binary columns
            # Handle yes/no binary strings first if they are mapped to numeric/binary features
            if val_series.dtype == object:
                # Map common binary strings
                binary_map = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, '1': 1, '0': 0, '1.0': 1, '0.0': 0}
                mapped_series = val_series.astype(str).str.lower().str.strip().map(binary_map)
                # If it successfully mapped at least some values
                if not mapped_series.dropna().empty:
                    val_series = mapped_series
                    
            cleaned_df[col] = pd.to_numeric(val_series, errors='coerce').fillna(DEFAULT_VALUES.get(col, 0.0))
            
    return cleaned_df

def load_and_map_real_world_dataset(file_path, industry):
    """
    Loads a real-world CSV dataset, detects target column, maps features using NLP semantic mapping,
    and splits the dataset into stratified train (70%), validation (15%), and test (15%) splits.
    """
    df_raw = pd.read_csv(file_path)
    
    # 1. Identify target column using synonyms
    from src.nlp_mapper import map_columns_nlp, clean_name, SYNONYMS, map_target_values
    target_syns = SYNONYMS.get('churned', []) + ['churned', 'churn', 'exited', 'class', 'target', 'label']
    target_syns_clean = [clean_name(s) for s in target_syns]
    
    target_col = None
    original_target_col = None
    for col in df_raw.columns:
        if clean_name(col) in target_syns_clean:
            original_target_col = col
            target_col = 'churned'
            break
            
    if not target_col:
        raise ValueError(f"Could not detect target churn column in {file_path}")
        
    df_mapped = df_raw.copy()
    
    # Clean target labels using nlp_mapper helper
    df_mapped['churned'] = map_target_values(df_raw[original_target_col])
    
    if original_target_col != 'churned':
        df_mapped = df_mapped.drop(columns=[original_target_col], errors='ignore')
        
    # 2. Map other columns using map_columns_nlp
    headers = [c for c in df_mapped.columns if c != 'churned']
    mapping = map_columns_nlp(headers, industry)
    df_mapped = df_mapped.rename(columns=mapping)
    df_mapped = df_mapped.loc[:, ~df_mapped.columns.duplicated()]
    
    # 3. Clean and type-cast mapped columns to conform to schemas
    df_mapped = clean_mapped_features(df_mapped, industry)
    
    # Keep only target and expected schema features (plus customer_id if present)
    from src.nlp_mapper import INDUSTRY_SCHEMAS
    expected_features = INDUSTRY_SCHEMAS.get(industry, [])
    cols_to_keep = ['churned']
    for col in df_mapped.columns:
        if col in expected_features or col == 'customer_id':
            cols_to_keep.append(col)
    df_mapped = df_mapped[cols_to_keep]
    
    # Drop rows with missing target
    df_mapped = df_mapped.dropna(subset=['churned'])
    df_mapped['churned'] = df_mapped['churned'].astype(int)
    
    # Split into train (70%), validation (15%), test (15%) splits
    from sklearn.model_selection import train_test_split
    df_train_raw, df_temp = train_test_split(df_mapped, test_size=0.3, random_state=42, stratify=df_mapped['churned'])
    df_val_raw, df_test_raw = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=df_temp['churned'])
    
    # Create test answer key (with customer_id, churn_probability, churned)
    df_test_ans_raw = df_test_raw.copy()
    if 'churn_probability' not in df_test_ans_raw.columns:
        df_test_ans_raw['churn_probability'] = 0.0
        
    return df_train_raw, df_val_raw, df_test_raw, df_test_ans_raw

def train_industry(industry):
    logger.info(f"==================== Training {industry.upper()} Model ====================")
    
    # Check for real-world datasets in BASE_DIR first for generalization
    real_world_files = {
        'telecom': os.path.join(BASE_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        'banking': os.path.join(BASE_DIR, 'Bank_Churn_Modelling.csv'),
        'saas': os.path.join(BASE_DIR, 'SaaS_customer_subscription_churn_usage_patterns.csv'),
        'ecommerce': os.path.join(BASE_DIR, 'E Commerce Dataset(E Comm).csv')
    }
    
    loaded_real_world = False
    use_mock_only = os.getenv('USE_MOCK_ONLY', 'true').lower() == 'true'
    if not use_mock_only and industry in real_world_files and os.path.exists(real_world_files[industry]):
        try:
            logger.info(f"Loading real-world dataset for {industry} from {real_world_files[industry]}...")
            df_train_raw, df_val_raw, df_test_raw, df_test_ans_raw = load_and_map_real_world_dataset(
                real_world_files[industry], industry
            )
            loaded_real_world = True
            logger.info(f"Successfully loaded and mapped real-world data splits. Train shape: {df_train_raw.shape}")
        except Exception as e:
            logger.warning(f"Failed to load/map real-world dataset for {industry}: {e}. Falling back to mock data.")
            
    if not loaded_real_world:
        # Load mock data
        train_path = os.path.join(DATA_DIR, 'train', f'{industry}_churn_train.csv')
        val_path = os.path.join(DATA_DIR, 'val', f'{industry}_churn_val.csv')
        test_path = os.path.join(DATA_DIR, 'test', f'{industry}_churn_test_features.csv')
        test_ans_path = os.path.join(DATA_DIR, 'test_answer_key', f'{industry}_churn_test_answer_key.csv')
        
        df_train_raw = pd.read_csv(train_path)
        df_val_raw = pd.read_csv(val_path)
        df_test_raw = pd.read_csv(test_path)
        df_test_ans_raw = pd.read_csv(test_ans_path)
        logger.info(f"Loaded raw datasets. Train shape: {df_train_raw.shape}, Val shape: {df_val_raw.shape}")
    
    df_train = validate_raw_data(df_train_raw, industry)
    df_val = validate_raw_data(df_val_raw, industry)
    df_test = validate_raw_data(df_test_raw, industry)
    df_test_ans = validate_raw_data(df_test_ans_raw, industry)
    
    # Dynamically find binary columns and ensure they are clean/numeric if appropriate
    # (e.g. columns with <= 2 unique values)
    for col in df_train.columns:
        if col in ['churned', 'customer_id', 'industry']:
            continue
        # Check if column has 2 or fewer unique values
        if df_train[col].nunique(dropna=True) <= 2:
            try:
                unique_vals = set(df_train[col].dropna().unique())
                # If values are binary-like
                if unique_vals.issubset({0, 1, 0.0, 1.0, '0', '1', True, False, 'True', 'False', 'yes', 'no', 'Yes', 'No'}):
                    def cast_binary(series):
                        s = series.astype(str).str.lower().str.strip()
                        return s.map({'true': 1, '1': 1, '1.0': 1, 'yes': 1, 'false': 0, '0': 0, '0.0': 0, 'no': 0}).fillna(0).astype(int)
                    
                    df_train[col] = cast_binary(df_train[col])
                    if col in df_val.columns:
                        df_val[col] = cast_binary(df_val[col])
                    if col in df_test.columns:
                        df_test[col] = cast_binary(df_test[col])
                    if col in df_test_ans.columns:
                        df_test_ans[col] = cast_binary(df_test_ans[col])
            except Exception:
                pass
            
    # Identify target
    y_train = df_train['churned'].values.ravel()
    y_val = df_val['churned'].values.ravel()
    y_test = df_test_ans['churned'].values.ravel()
    
    # Determine features
    numeric_features, categorical_features = get_feature_types(df_train, industry)
    logger.info(f"Numerical features ({len(numeric_features)}): {numeric_features}")
    logger.info(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    
    # Drop irrelevant target/leakage columns (tenure_months strictly dropped from Classification for saas & telecom)
    cols_to_drop = ['customer_id', 'industry', 'churn_probability', 'churned']
    if industry in ['saas', 'telecom']:
        cols_to_drop.append('tenure_months')
    X_train_raw = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns])
    X_val_raw = df_val.drop(columns=[c for c in cols_to_drop if c in df_val.columns])
    X_test_raw = df_test.drop(columns=[c for c in cols_to_drop if c in df_test.columns])
    
    # Hyperparameter Tuning using Unified Optuna Stacking study on complete X_train_raw
    logger.info(f"Running unified Optuna tuning study for {industry} base estimators...")
    best_params = tune_stacking_optuna(X_train_raw, y_train, numeric_features, categorical_features, industry, n_trials=3)
    logger.info(f"Tuning finished. Best params: {best_params}")
    
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
        'colsample_bytree': best_params['colsample_bytree'],
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
    
    xgb_best = xgb.XGBClassifier(**xgb_best_params)
    lgb_best = lgb.LGBMClassifier(**lgb_best_params)
    cb_best = SklearnCatBoostWrapper(
        iterations=cb_best_params['iterations'],
        depth=cb_best_params['depth'],
        learning_rate=cb_best_params['learning_rate'],
        subsample=cb_best_params['subsample'],
        random_state=cb_best_params['random_state'],
        verbose=cb_best_params['verbose'],
        thread_count=cb_best_params['thread_count'],
        cat_features=categorical_features
    )
    
    # Base estimators wrapped in pipelines with DataFrameCasters to preserve types
    xgb_pipe = Pipeline([
        ('caster', DataFrameCaster(numeric_features, categorical_features, to_string=True)),
        ('ohe', ColumnTransformer([('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_features)], remainder='passthrough')),
        ('xgb', xgb_best)
    ])
    
    lgb_pipe = Pipeline([
        ('caster', DataFrameCaster(numeric_features, categorical_features, to_string=False)),
        ('lgb', lgb_best)
    ])
    
    cb_pipe = Pipeline([
        ('caster', DataFrameCaster(numeric_features, categorical_features, to_string=True)),
        ('cb', cb_best)
    ])
    
    # Final stacking ensemble fit uses cv=5 and parallel n_jobs=-1
    ensemble = StackingClassifier(
        estimators=[('xgb', xgb_pipe), ('lgb', lgb_pipe), ('cb', cb_pipe)],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )
    
    # Global unified pipeline
    preprocessor = build_preprocessor(numeric_features, categorical_features)
    clf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('ensemble', ensemble)
    ])
    
    logger.info("Fitting final StackingClassifier unified pipeline on X_train_raw...")
    clf_pipeline.fit(X_train_raw, y_train)
    
    # Conformal Calibration (MAPIE) using the completely isolated validation set
    logger.info(f"Calibrating conformal predictor using separate validation set ({len(X_val_raw)} rows)...")
    confidence_levels = [0.80, 0.85, 0.90, 0.95]
    try:
        mapie_model = SplitConformalClassifier(estimator=clf_pipeline, confidence_level=confidence_levels, prefit=True)
        mapie_model.conformalize(X_val_raw, y_val)
    except Exception as conformal_err:
        logger.error(f"MAPIE Conformal calibration failed: {str(conformal_err)}")
        raise RuntimeError(f"Failed conformalization phase: {conformal_err}")
    
    # Quality Guardrail Check (dev deviation < 5%)
    logger.info("Checking conformal calibration empirical coverage guardrails...")
    _, y_pis = mapie_model.predict_set(X_val_raw)
    for idx, target_lvl in enumerate(confidence_levels):
        empirical_cov = np.mean(y_pis[np.arange(len(y_val)), y_val, idx])
        deviation = abs(empirical_cov - target_lvl)
        logger.info(f"Level {target_lvl:.2f} -> Empirical Coverage: {empirical_cov:.4f} (deviation: {deviation:.4f})")
        if deviation > 0.05:
            raise CalibrationQualityException(
                f"Conformal calibration quality check failed for {industry}. "
                f"Target coverage: {target_lvl:.2f}, Empirical coverage: {empirical_cov:.4f} (Deviation: {deviation:.4f} > 5%)"
            )
    
    logger.info("Conformal coverage guardrails passed successfully!")
    
    # Evaluate
    y_pred = clf_pipeline.predict(X_test_raw)
    y_pred_proba = clf_pipeline.predict_proba(X_test_raw)[:, 1]
    
    logger.info(f"--- {industry.upper()} Evaluation Results ---")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
    # Save processed splits for debugging/conformal stats (matching legacy output structure)
    X_train_trans = clf_pipeline.named_steps['preprocessor'].transform(X_train_raw)
    X_test_trans = clf_pipeline.named_steps['preprocessor'].transform(X_test_raw)
    X_train_proc_df = pd.DataFrame(X_train_trans, columns=numeric_features + categorical_features)
    X_test_proc_df = pd.DataFrame(X_test_trans, columns=numeric_features + categorical_features)
    
    X_train_proc_df.to_csv(os.path.join(OUTPUT_DIR, f'X_train_processed_{industry}.csv'), index=False)
    X_test_proc_df.to_csv(os.path.join(OUTPUT_DIR, f'X_test_processed_{industry}.csv'), index=False)
    pd.DataFrame(y_train, columns=['churned']).to_csv(os.path.join(OUTPUT_DIR, f'y_train_{industry}.csv'), index=False)
    pd.DataFrame(y_test, columns=['churned']).to_csv(os.path.join(OUTPUT_DIR, f'y_test_{industry}.csv'), index=False)

    # Save raw and preprocessed train/val/test data as augmented files to keep cache unified and consistent
    raw_train_df = X_train_raw.copy()
    raw_train_df['churned'] = y_train
    raw_train_df.to_csv(os.path.join(OUTPUT_DIR, f'{industry}_augmented_raw.csv'), index=False)
    X_train_proc_df.to_csv(os.path.join(OUTPUT_DIR, f'{industry}_augmented_X.csv'), index=False)
    pd.DataFrame(y_train, columns=['churned']).to_csv(os.path.join(OUTPUT_DIR, f'{industry}_augmented_y.csv'), index=False)
    
    # Save validation set
    X_val_trans = clf_pipeline.named_steps['preprocessor'].transform(X_val_raw)
    X_val_proc_df = pd.DataFrame(X_val_trans, columns=numeric_features + categorical_features)
    raw_val_df = X_val_raw.copy()
    raw_val_df['churned'] = y_val
    
    raw_val_df.to_csv(os.path.join(OUTPUT_DIR, f'{industry}_augmented_val_raw.csv'), index=False)
    X_val_proc_df.to_csv(os.path.join(OUTPUT_DIR, f'{industry}_augmented_val_X.csv'), index=False)
    pd.DataFrame(y_val, columns=['churned']).to_csv(os.path.join(OUTPUT_DIR, f'{industry}_augmented_val_y.csv'), index=False)
    
    # Save artifacts
    preprocessor_path = os.path.join(OUTPUT_DIR, f'preprocessor_{industry}.joblib')
    model_path = os.path.join(OUTPUT_DIR, f'model_{industry}.joblib')
    mapie_model_path = os.path.join(OUTPUT_DIR, f'mapie_model_{industry}.joblib')
    
    # Save step preprocessor separately for backwards compatibility
    joblib.dump(clf_pipeline.named_steps['preprocessor'], preprocessor_path)
    joblib.dump(clf_pipeline, model_path)
    joblib.dump(mapie_model, mapie_model_path)
    
    logger.info(f"Saved preprocessor step to: {preprocessor_path}")
    logger.info(f"Saved global pipeline model to: {model_path}")
    logger.info(f"Saved conformal mapie model to: {mapie_model_path}")

    # Framework for Survival Analysis (SaaS & Telecom Deployment)
    if industry in ['saas', 'telecom']:
        logger.info(f"--- Training Survival Analysis Model for {industry.upper()} Time-to-Churn ---")
        try:
            from lifelines import CoxPHFitter
            
            # Prepare survival data: use preprocessed features + tenure & churn target
            X_train_surv = X_train_proc_df.copy()
            # Enforce correct numerical dtypes
            for col in numeric_features:
                X_train_surv[col] = pd.to_numeric(X_train_surv[col], errors='coerce')
            
            # Restore tenure_months specifically for survival duration target
            X_train_surv['tenure_months'] = df_train_raw['tenure_months'].values
            X_train_surv['churned'] = y_train
            
            # One-Hot Encode categorical features for CoxPHFitter
            if categorical_features:
                X_train_surv = pd.get_dummies(X_train_surv, columns=categorical_features, drop_first=True, dtype=float)
            
            # Drop zero variance columns to prevent convergence/singular matrix failures
            non_surv_cols = [c for c in X_train_surv.columns if c not in ['tenure_months', 'churned']]
            zero_var_cols = [c for c in non_surv_cols if X_train_surv[c].var() == 0]
            if zero_var_cols:
                logger.info(f"Dropping zero-variance features for survival: {zero_var_cols}")
                X_train_surv = X_train_surv.drop(columns=zero_var_cols)
                
            # Fit Cox Proportional Hazards model with L2 regularization
            cph = CoxPHFitter(penalizer=0.1)
            cph.fit(X_train_surv, duration_col='tenure_months', event_col='churned')
            logger.info("Cox Proportional Hazards Model fitted successfully!")
            
            # Print top features based on Hazard Ratios
            summary_df = cph.summary.sort_values(by='p')
            logger.info("Survival Model Hazard Ratios (Top 5 significant features):")
            logger.info(summary_df[['coef', 'exp(coef)', 'p']].head(5).to_string())
            
            # Save survival model to disk
            cph_path = os.path.join(OUTPUT_DIR, f'{industry}_survival_model.joblib')
            joblib.dump(cph, cph_path)
            logger.info(f"Saved {industry.upper()} survival model to: {cph_path}")
        except Exception as surv_err:
            logger.error(f"Failed to fit {industry} CoxPH model: {surv_err}")

def main():
    logger.info("Starting LoyalScale Multi-Industry Model Training Pipeline...")
    for ind in INDUSTRIES:
        try:
            train_industry(ind)
        except Exception as ind_err:
            logger.error(f"Failed training loop for industry {ind}: {str(ind_err)}")
            
    logger.info("All industries training loop executed!")

if __name__ == '__main__':
    main()
