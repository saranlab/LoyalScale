import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load preprocessed data and models
X_test = pd.read_csv('processed_data/X_test_processed.csv')
y_test = pd.read_csv('processed_data/y_test.csv').values.ravel()
model = joblib.load('processed_data/model.joblib')

# Wait, let's train a model with multiple levels to see the exact numbers!
from mapie.classification import SplitConformalClassifier
from sklearn.model_selection import train_test_split

X_train = pd.read_csv('processed_data/X_train_processed.csv')
y_train = pd.read_csv('processed_data/y_train.csv').values.ravel()

# Split train for fitting and calibration
X_train_fit, X_calib, y_train_fit, y_calib = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

levels = [0.80, 0.85, 0.90, 0.95]
mapie_model = SplitConformalClassifier(estimator=model, confidence_level=levels, prefit=True)
mapie_model.conformalize(X_calib, y_calib)

y_pred, y_pis = mapie_model.predict_set(X_test)
# y_pis shape: (n_samples, n_classes, n_confidence_levels)

print("=== Telecom Conformal Statistics ===")
for idx, lvl in enumerate(levels):
    print(f"\n--- Confidence Level: {lvl * 100}% (alpha={1-lvl:.2f}) ---")
    
    # Coverage
    coverage = np.mean([y_pis[i, y_test[i], idx] for i in range(len(y_test))])
    print(f"Empirical Coverage: {coverage * 100:.2f}%")
    
    # Set distributions
    # class 0: Retained, class 1: Churned
    set_contents = []
    retained_count = 0
    churned_count = 0
    uncertain_count = 0
    empty_count = 0
    
    for i in range(len(y_test)):
        in_0 = y_pis[i, 0, idx]
        in_1 = y_pis[i, 1, idx]
        if in_0 and in_1:
            uncertain_count += 1
        elif in_0:
            retained_count += 1
        elif in_1:
            churned_count += 1
        else:
            empty_count += 1
            
    total = len(y_test)
    print(f"Total test samples: {total}")
    print(f"  Confident Retained (Set: [Retained]): {retained_count} ({retained_count/total*100:.2f}%)")
    print(f"  Confident Churn (Set: [Churned]): {churned_count} ({churned_count/total*100:.2f}%)")
    print(f"  Uncertain / Monitoring (Set: [Retained, Churned]): {uncertain_count} ({uncertain_count/total*100:.2f}%)")
    print(f"  Empty Set: {empty_count} ({empty_count/total*100:.2f}%)")
    
    # Business Simulation:
    # Let's assume:
    # - Average value of a retained customer = $500
    # - Cost of active monitoring / customer outreach = $50
    # - If we outreach to a Confident Churner, we have a 40% chance of retaining them (saving $500 - $50 = $450).
    # - If we outreach to an Uncertain customer, we have a 60% chance of retaining them if they were going to churn (saving $500 - $50 = $450).
    # Let's count how many actual churners are in each set, and compute cost/benefit:
    # Outreach target 1: Confident Churn (we target all of them)
    # Outreach target 2: Uncertain (we monitor them, costing $50 each, but we prevent some churn)
    # Let's compute:
    # - Monitoring/Outreach Cost: (churned_count + uncertain_count) * $50
    # - Churners correctly identified and saved:
    #   - For Confident Churn: number of actual churners in this set * 0.40 * $500
    #   - For Uncertain: number of actual churners in this set * 0.20 * $500
    # Let's print this simulation.
    actual_churn_in_churned = sum([1 for i in range(total) if y_test[i] == 1 and y_pis[i, 1, idx] and not y_pis[i, 0, idx]])
    actual_churn_in_uncertain = sum([1 for i in range(total) if y_test[i] == 1 and y_pis[i, 1, idx] and y_pis[i, 0, idx]])
    
    cost = (churned_count + uncertain_count) * 50
    savings = (actual_churn_in_churned * 0.40 * 500) + (actual_churn_in_uncertain * 0.20 * 500)
    net_benefit = savings - cost
    print(f"  Simulation - Cost: ${cost:,}, Savings: ${savings:,.0f}, Net Benefit: ${net_benefit:,.0f}")
