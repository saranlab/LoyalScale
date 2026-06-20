import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb

# Append current working directory to sys.path
cwd = os.getcwd()
if cwd not in sys.path:
    sys.path.append(cwd)

from src.train_all_industries import get_feature_types, build_preprocessor, DataFrameCaster, SklearnCatBoostWrapper
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from mapie.classification import SplitConformalClassifier

industry = 'saas'
DATA_DIR = os.getenv('CHURN_DATA_DIR')
if not DATA_DIR:
    DATA_DIR = os.path.join(cwd, 'mock_churn_data')
    if not os.path.exists(DATA_DIR):
        DATA_DIR = os.path.join(os.path.dirname(cwd), 'forMock', 'mock_churn_data')

print(f"Loading data from {DATA_DIR}...")
train_path = os.path.join(DATA_DIR, 'train', f'{industry}_churn_train.csv')
val_path = os.path.join(DATA_DIR, 'val', f'{industry}_churn_val.csv')

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)

# Mock some columns
cols_to_drop = ['customer_id', 'industry', 'churn_probability', 'churned']
if industry in ['saas', 'telecom']:
    cols_to_drop.append('tenure_months')

X_train_raw = df_train.drop(columns=[c for c in cols_to_drop if c in df_train.columns], errors='ignore')
X_val_raw = df_val.drop(columns=[c for c in cols_to_drop if c in df_val.columns], errors='ignore')
y_train = df_train['churned'].values.ravel()
y_val = df_val['churned'].values.ravel()

# Determine features dynamically
numeric_features, categorical_features = get_feature_types(X_train_raw, industry)

print("Building stacking pipeline estimators...")
xgb_clf = xgb.XGBClassifier(n_estimators=10, max_depth=3, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss', n_jobs=1)
lgb_clf = lgb.LGBMClassifier(n_estimators=10, max_depth=3, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=1)
cb_clf = SklearnCatBoostWrapper(iterations=10, depth=3, learning_rate=0.08, subsample=0.8, random_state=42, verbose=0, thread_count=1, cat_features=categorical_features)

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
    n_jobs=-1
)

preprocessor = build_preprocessor(numeric_features, categorical_features)
clf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('caster', DataFrameCaster(numeric_features, categorical_features, to_string=False)),
    ('ensemble', ensemble)
])

print("Fitting global pipeline...")
clf_pipeline.fit(X_train_raw, y_train)

print("Calibrating mapie model...")
confidence_levels = [0.80, 0.85, 0.90, 0.95]
mapie_model = SplitConformalClassifier(estimator=clf_pipeline, confidence_level=confidence_levels, prefit=True)
mapie_model.conformalize(X_val_raw, y_val)

print("SUCCESS: Conformal calibration succeeded without any errors!")
