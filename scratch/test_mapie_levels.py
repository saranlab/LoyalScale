import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from mapie.classification import SplitConformalClassifier
import xgboost as xgb

X_train = pd.read_csv('processed_data/X_train_processed.csv')
y_train = pd.read_csv('processed_data/y_train.csv').values.ravel()

# Split train for fitting and calibration
X_train_fit, X_calib, y_train_fit, y_calib = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

model = xgb.XGBClassifier(
    n_estimators=10, # small for quick test
    max_depth=3,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train_fit, y_train_fit)

# Pass multiple confidence levels!
levels = [0.80, 0.85, 0.90, 0.95]
try:
    mapie_model = SplitConformalClassifier(estimator=model, confidence_level=levels, prefit=True)
    mapie_model.conformalize(X_calib, y_calib)
    print("Success conformalizing with multiple levels!")
    y_pred, y_pis = mapie_model.predict_set(X_calib.iloc[:5])
    print("y_pis shape:", y_pis.shape) # Expected: (5, 2, 4) because 5 samples, 2 classes, 4 levels
    for idx, lvl in enumerate(levels):
        print(f"\nLevel: {lvl * 100}% (index {idx})")
        print(y_pis[:, :, idx])
except Exception as e:
    print("Error training with list of confidence levels:", str(e))
