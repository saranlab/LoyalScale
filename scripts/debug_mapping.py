"""
Utility script to test RapidFuzz fuzzy column mapping against benchmark datasets.
"""

import os
import sys
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.fuzzy_mapper import map_columns_fuzzy_simple, detect_industry_fuzzy

files = {
    'telecom': 'WA_Fn-UseC_-Telco-Customer-Churn.csv',
    'banking': 'Bank_Churn_Modelling.csv',
    'saas': 'SaaS_customer_subscription_churn_usage_patterns.csv',
    'ecommerce': 'E Commerce Dataset(E Comm).csv'
}

for ind, filename in files.items():
    filepath = os.path.join(BASE_DIR, 'data', filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        continue
    df = pd.read_csv(filepath, nrows=2)
    headers = df.columns.tolist()
    det = detect_industry_fuzzy(headers)
    mapping = map_columns_fuzzy_simple(headers, ind)
    print(f"\nExpected Sector: {ind.upper()} | Detected: {det.upper()}")
    print("Resolved Schema Mapping:")
    for k, v in mapping.items():
        print(f"  {k} -> {v}")
