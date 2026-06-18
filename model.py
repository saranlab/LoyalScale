import os
import sys
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from mapie.classification import SplitConformalClassifier
import joblib
import optuna

# Setup production logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("LoyalScaleTelecomModel")

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
        for col in self.numeric_features:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in self.categorical_features:
            if self.to_string:
                df[col] = df[col].astype(str)
            else:
                df[col] = df[col].astype('category')
        return df

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

def load_preprocessed_data(data_dir: str = 'processed_data'):
    """Loads preprocessed training and testing data from directory."""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_processed.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_processed.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    return X_train, X_test, y_train, y_test

def tune_stacking_optuna(X, y, numeric_features, categorical_features, n_trials=5):
    """
    Tunes hyperparameters for XGBoost, LightGBM, and CatBoost simultaneously using a
    single unified Optuna study. Maximizes the cross-validated ROC-AUC of the
    final StackingClassifier ensemble.
    """
    def objective(trial):
        xgb_params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 150, step=50),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 5),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.05, 0.2, step=0.05),
            'random_state': 42,
            'eval_metric': 'logloss',
            'n_jobs': 1
        }
        
        lgb_params = {
            'n_estimators': trial.suggest_int('lgb_n_estimators', 50, 150, step=50),
            'max_depth': trial.suggest_int('lgb_max_depth', 3, 5),
            'learning_rate': trial.suggest_float('lgb_learning_rate', 0.05, 0.2, step=0.05),
            'random_state': 42,
            'verbose': -1,
            'n_jobs': 1
        }
        
        cb_params = {
            'iterations': trial.suggest_int('cb_iterations', 50, 150, step=50),
            'depth': trial.suggest_int('cb_depth', 3, 5),
            'learning_rate': trial.suggest_float('cb_learning_rate', 0.05, 0.2, step=0.05),
            'random_state': 42,
            'verbose': 0,
            'thread_count': 1
        }
        
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            xgb_clf = xgb.XGBClassifier(**xgb_params)
            lgb_clf = lgb.LGBMClassifier(**lgb_params)
            cb_clf = SklearnCatBoostWrapper(
                iterations=cb_params['iterations'],
                depth=cb_params['depth'],
                learning_rate=cb_params['learning_rate'],
                subsample=cb_params.get('subsample'),
                random_state=cb_params['random_state'],
                verbose=cb_params['verbose'],
                thread_count=cb_params['thread_count'],
                cat_features=categorical_features
            )
            
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
            ensemble.fit(X_tr, y_tr)
            y_pred_proba = ensemble.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)
            
            trial.report(score, fold)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
        return float(np.mean(scores))

    try:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(objective, n_trials=n_trials)
        return study.best_params
    except Exception as e:
        logger.error(f"Optuna tuning failed in model.py: {e}")
        raise RuntimeError(e)

def train_and_evaluate():
    data_dir = 'processed_data'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Preprocessed data directory '{data_dir}' not found. "
            "Please run 'run_pipeline.py' first to prepare the data."
        )
    
    logger.info("=== Loading Preprocessed Data ===")
    X_train_raw, X_test_raw, y_train, y_test = load_preprocessed_data(data_dir)
    logger.info(f"X_train Shape: {X_train_raw.shape}, X_test Shape: {X_test_raw.shape}")
    
    # Identify numeric and categorical columns from the dataset
    # In run_pipeline.py, the columns are numeric features followed by expanded OHE categories.
    # To treat them natively, since the processed files are already One-Hot encoded,
    # let's reconstruct the features or treat them as raw.
    # Wait, the processed files X_train_processed.csv from run_pipeline.py are already OHE!
    # If they are already OHE, we cannot run native categoricals on them.
    # But to maintain full architectural consistency with train_all_industries.py, 
    # we can define the numeric and categorical columns based on the original features.
    # Wait! The original columns in run_pipeline.py are:
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # If the processed files are already OHE, let's treat all columns in X_train_processed as numeric
    # for the caster (categorical_features = []) to keep it simple and avoid OHE mismatch.
    categorical_features = []
    
    # Split training set into fit and calibration sets for Conformal Prediction
    logger.info("\n=== Splitting Train Data for Conformal Calibration ===")
    X_train_fit, X_calib, y_train_fit, y_calib = train_test_split(
        X_train_raw, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    logger.info(f"Train Fit Shape: {X_train_fit.shape}, Calibration Shape: {X_calib.shape}")
    
    # 1. Hyperparameter Tuning for base estimators (Unified Optuna study)
    logger.info("\n=== Unified Stacking Hyperparameter Tuning (Optuna) ===")
    best_params = tune_stacking_optuna(X_train_fit, y_train_fit, numeric_features, categorical_features, n_trials=5)
    logger.info(f"Best Tuned Study Parameters: {best_params}")
    
    xgb_best_params = {
        'n_estimators': best_params['xgb_n_estimators'],
        'max_depth': best_params['xgb_max_depth'],
        'learning_rate': best_params['xgb_learning_rate'],
        'random_state': 42,
        'eval_metric': 'logloss',
        'n_jobs': 1
    }
    lgb_best_params = {
        'n_estimators': best_params['lgb_n_estimators'],
        'max_depth': best_params['lgb_max_depth'],
        'learning_rate': best_params['lgb_learning_rate'],
        'random_state': 42,
        'verbose': -1,
        'n_jobs': 1
    }
    cb_best_params = {
        'iterations': best_params['cb_iterations'],
        'depth': best_params['cb_depth'],
        'learning_rate': best_params['cb_learning_rate'],
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
        subsample=cb_best_params.get('subsample'),
        random_state=cb_best_params['random_state'],
        verbose=cb_best_params['verbose'],
        thread_count=cb_best_params['thread_count'],
        cat_features=categorical_features
    )
    
    # Wrap base estimators in pipelines with DataFrameCasters to preserve types
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
    
    # 2. Stacking Classifier ensemble
    logger.info("\n=== Training Stacking Ensemble Classifier ===")
    ensemble = StackingClassifier(
        estimators=[('xgb', xgb_pipe), ('lgb', lgb_pipe), ('cb', cb_pipe)],
        final_estimator=LogisticRegression(),
        cv=5,
        n_jobs=-1
    )
    ensemble.fit(X_train_fit, y_train_fit)
    
    # 3. Conformal Calibration with MAPIE (multiple confidence levels)
    logger.info("\n=== Calibrating Conformal Predictor (MAPIE) ===")
    try:
        confidence_levels = [0.80, 0.85, 0.90, 0.95]
        mapie_model = SplitConformalClassifier(estimator=ensemble, confidence_level=confidence_levels, prefit=True)
        mapie_model.conformalize(X_calib, y_calib)
    except Exception as e:
        logger.error(f"Conformal calibration failed in model.py: {e}")
        raise RuntimeError(e)
    
    # Evaluate base ensemble
    y_pred = ensemble.predict(X_test_raw)
    y_pred_proba = ensemble.predict_proba(X_test_raw)[:, 1]
    
    logger.info("\n--- Base Ensemble Model Evaluation ---")
    logger.info(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    logger.info("\n--- Classification Report ---")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    # Evaluate conformal predictions across all levels
    y_pred_mapie, y_pis = mapie_model.predict_set(X_test_raw)
    logger.info("\n--- Conformal Evaluation ---")
    for idx, lvl in enumerate(confidence_levels):
        coverage = np.mean([y_pis[i, y_test[i], idx] for i in range(len(y_test))])
        set_sizes = np.sum(y_pis[:, :, idx], axis=1)
        avg_set_size = np.mean(set_sizes)
        logger.info(f"Confidence Level {lvl*100}%:")
        logger.info(f"  Empirical Coverage: {coverage*100:.2f}%")
        logger.info(f"  Average Set Size: {avg_set_size:.2f}")
        
    # Save artifacts
    model_path = os.path.join(data_dir, 'model.joblib')
    mapie_model_path = os.path.join(data_dir, 'mapie_model.joblib')
    
    joblib.dump(ensemble, model_path)
    joblib.dump(mapie_model, mapie_model_path)
    logger.info(f"\nEnsemble model saved to '{model_path}'")
    logger.info(f"MAPIE Conformal predictor saved to '{mapie_model_path}'")

if __name__ == '__main__':
    train_and_evaluate()
