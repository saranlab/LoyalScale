import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from mapie.classification import SplitConformalClassifier
import joblib
import optuna

def load_preprocessed_data(data_dir: str = 'processed_data'):
    """Loads preprocessed training and testing data from directory."""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train_processed.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test_processed.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    return X_train, X_test, y_train, y_test

def tune_xgb_optuna(X, y, n_trials=5):
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, step=0.05),
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
            'n_estimators': trial.suggest_int('n_estimators', 50, 150, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, step=0.05),
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
            'iterations': trial.suggest_int('iterations', 50, 150, step=50),
            'depth': trial.suggest_int('depth', 3, 5),
            'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.2, step=0.05),
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

def train_and_evaluate():
    data_dir = 'processed_data'
    if not os.path.exists(data_dir):
        raise FileNotFoundError(
            f"Preprocessed data directory '{data_dir}' not found. "
            "Please run 'run_pipeline.py' first to prepare the data."
        )
    
    print("=== Loading Preprocessed Data ===")
    X_train, X_test, y_train, y_test = load_preprocessed_data(data_dir)
    print(f"X_train Shape: {X_train.shape}")
    print(f"X_test Shape: {X_test.shape}")
    
    # Split training set into fit and calibration sets for Conformal Prediction
    print("\n=== Splitting Train Data for Conformal Calibration ===")
    X_train_fit, X_calib, y_train_fit, y_calib = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"Train Fit Shape: {X_train_fit.shape}, Calibration Shape: {X_calib.shape}")
    
    # 1. Hyperparameter Tuning for base estimators
    print("\n=== Hyperparameter Tuning for XGBoost (Optuna) ===")
    xgb_best_params = tune_xgb_optuna(X_train_fit, y_train_fit, n_trials=5)
    xgb_best = xgb.XGBClassifier(**xgb_best_params, random_state=42, eval_metric='logloss', n_jobs=1)
    xgb_best.fit(X_train_fit, y_train_fit)
    print("Best XGBoost Params:", xgb_best_params)
    
    print("\n=== Hyperparameter Tuning for LightGBM (Optuna) ===")
    lgb_best_params = tune_lgb_optuna(X_train_fit, y_train_fit, n_trials=5)
    lgb_best = lgb.LGBMClassifier(**lgb_best_params, random_state=42, verbose=-1, n_jobs=1)
    lgb_best.fit(X_train_fit, y_train_fit)
    print("Best LightGBM Params:", lgb_best_params)
    
    print("\n=== Hyperparameter Tuning for CatBoost (Optuna) ===")
    cb_best_params = tune_cb_optuna(X_train_fit, y_train_fit, n_trials=5)
    cb_best = cb.CatBoostClassifier(**cb_best_params, random_state=42, verbose=0, thread_count=1)
    cb_best.fit(X_train_fit, y_train_fit)
    print("Best CatBoost Params:", cb_best_params)
    
    # 2. Voting Classifier soft ensemble
    print("\n=== Training Voting Ensemble Classifier ===")
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb_best),
            ('lgb', lgb_best),
            ('cb', cb_best)
        ],
        voting='soft'
    )
    ensemble.fit(X_train_fit, y_train_fit)
    
    # 3. Conformal Calibration with MAPIE (multiple confidence levels)
    print("\n=== Calibrating Conformal Predictor (MAPIE) ===")
    confidence_levels = [0.80, 0.85, 0.90, 0.95]
    mapie_model = SplitConformalClassifier(estimator=ensemble, confidence_level=confidence_levels, prefit=True)
    mapie_model.conformalize(X_calib, y_calib)
    
    # Evaluate base ensemble
    y_pred = ensemble.predict(X_test)
    y_pred_proba = ensemble.predict_proba(X_test)[:, 1]
    
    print("\n--- Base Ensemble Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print("\n--- Classification Report ---")
    print(classification_report(y_test, y_pred))
    
    # Evaluate conformal predictions across all levels
    y_pred_mapie, y_pis = mapie_model.predict_set(X_test)
    print("\n--- Conformal Evaluation ---")
    for idx, lvl in enumerate(confidence_levels):
        coverage = np.mean([y_pis[i, y_test[i], idx] for i in range(len(y_test))])
        set_sizes = np.sum(y_pis[:, :, idx], axis=1)
        avg_set_size = np.mean(set_sizes)
        print(f"Confidence Level {lvl*100}%:")
        print(f"  Empirical Coverage: {coverage*100:.2f}%")
        print(f"  Average Set Size: {avg_set_size:.2f}")
        
    # Save artifacts
    model_path = os.path.join(data_dir, 'model.joblib')
    mapie_model_path = os.path.join(data_dir, 'mapie_model.joblib')
    
    joblib.dump(ensemble, model_path)
    joblib.dump(mapie_model, mapie_model_path)
    print(f"\nEnsemble model saved to '{model_path}'")
    print(f"MAPIE Conformal predictor saved to '{mapie_model_path}'")

if __name__ == '__main__':
    train_and_evaluate()
