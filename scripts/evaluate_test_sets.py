import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Add workspace to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.train_all_industries import DataFrameCaster, SklearnCatBoostWrapper, TypeCaster, CalibrationQualityException, validate_raw_data, get_feature_types

def evaluate_all():
    mock_dir = r"C:\Users\Saran\Documents\forMock\mock_churn_data"
    output_dir = os.path.join(BASE_DIR, 'processed_data')
    
    industries = ['telecom', 'saas', 'retail', 'banking', 'ecommerce', 'education', 'healthcare', 'hospitality', 'insurance', 'utilities']
    
    results = []
    
    print("=== Model Predictive Performance Validation against Test Answer Keys ===")
    print(f"{'Industry':<15} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'Test Size':<10}")
    print("-" * 55)
    
    for ind in industries:
        model_path = os.path.join(output_dir, f'model_{ind}.joblib')
        test_feat_path = os.path.join(mock_dir, 'test', f'{ind}_churn_test_features.csv')
        test_ans_path = os.path.join(mock_dir, 'test_answer_key', f'{ind}_churn_test_answer_key.csv')
        
        if not os.path.exists(model_path):
            print(f"{ind:<15} | Model file not found at {model_path}")
            continue
        if not os.path.exists(test_feat_path) or not os.path.exists(test_ans_path):
            print(f"{ind:<15} | Test data files not found in mock folder")
            continue
            
        try:
            # Load model pipeline
            model = joblib.load(model_path)
            
            # Load test datasets
            df_feat = pd.read_csv(test_feat_path)
            df_ans = pd.read_csv(test_ans_path)
            
            # Merge to align target and features by customer_id if present, otherwise assume matching order
            if 'customer_id' in df_feat.columns and 'customer_id' in df_ans.columns:
                df_test = pd.merge(df_feat, df_ans, on='customer_id', how='inner')
            else:
                df_test = df_feat.copy()
                df_test['churned'] = df_ans['churned']
                df_test['churn_probability'] = df_ans.get('churn_probability', 0.0)
            
            # Validate raw features to align schemas
            df_test_validated = validate_raw_data(df_test, ind)
            
            # Select features (drop leakage / id / target columns)
            cols_to_drop = ['customer_id', 'industry', 'churn_probability', 'churned']
            if ind in ['saas', 'telecom']:
                cols_to_drop.append('tenure_months')
                
            X_test = df_test_validated.drop(columns=[c for c in cols_to_drop if c in df_test_validated.columns], errors='ignore')
            y_test = df_test_validated['churned'].values
            
            # Run predictions
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)
            
            print(f"{ind:<15} | {acc:<10.4f} | {auc:<10.4f} | {len(y_test):<10}")
            results.append({
                'industry': ind,
                'accuracy': acc,
                'roc_auc': auc,
                'size': len(y_test)
            })
            
        except Exception as e:
            print(f"{ind:<15} | Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            
    if results:
        mean_acc = np.mean([r['accuracy'] for r in results])
        mean_auc = np.mean([r['roc_auc'] for r in results])
        print("-" * 55)
        print(f"{'Mean':<15} | {mean_acc:<10.4f} | {mean_auc:<10.4f} |")

if __name__ == '__main__':
    evaluate_all()
