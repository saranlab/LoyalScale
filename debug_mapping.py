import os
import sys
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
from src.nlp_mapper import map_columns_nlp, detect_industry

files = {
    'telecom': 'WA_Fn-UseC_-Telco-Customer-Churn.csv',
    'banking': 'Bank_Churn_Modelling.csv',
    'saas': 'SaaS_customer_subscription_churn_usage_patterns.csv',
    'ecommerce': 'E Commerce Dataset(E Comm).csv'
}

for ind, filename in files.items():
    filepath = os.path.join(CURRENT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    df = pd.read_csv(filepath, nrows=2)
    headers = df.columns.tolist()
    det = detect_industry(headers)
    mapping = map_columns_nlp(headers, ind)
    print(f"\nExpected: {ind} | Detected: {det}")
    print("Mapping:")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")
