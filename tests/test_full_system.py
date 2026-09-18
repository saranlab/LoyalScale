"""
LoyalScale Automated Integration Test Suite
===========================================
Validates end-to-end functionality across:
1. Tab 1: Single Diagnostic & Case-Based Reasoning (k-NN)
2. Tab 2: Batch CSV Audit & Fuzzy Schema Alignment Engine (RapidFuzz)
3. Tab 3: What-If Commercial Retention Simulator Model Sensitivity
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.train_all_industries import (
    DataFrameCaster, SklearnCatBoostWrapper, TypeCaster,
    CalibrationQualityException, clean_mapped_features
)
sys.modules['__main__'].DataFrameCaster = DataFrameCaster
sys.modules['__main__'].SklearnCatBoostWrapper = SklearnCatBoostWrapper
sys.modules['__main__'].TypeCaster = TypeCaster
sys.modules['__main__'].CalibrationQualityException = CalibrationQualityException

from src.fuzzy_mapper import (
    INDUSTRY_SCHEMAS, detect_industry_fuzzy, map_columns_fuzzy, map_columns_fuzzy_simple
)
from src.instance_explainer import InstanceExplainer

class TestLoyalScaleSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_dir = os.path.join(BASE_DIR, 'processed_data')
        cls.explainer = InstanceExplainer(cls.data_dir)

    def test_01_single_diagnostic_and_cbr(self):
        """Validates ensemble predictions, conformal tiers, and CBR across 4 benchmark sectors."""
        for ind in ['telecom', 'banking', 'saas', 'ecommerce']:
            model_path = os.path.join(self.data_dir, f'model_{ind}.joblib')
            mapie_path = os.path.join(self.data_dir, f'mapie_model_{ind}.joblib')
            
            self.assertTrue(os.path.exists(model_path), f"Missing model_{ind}.joblib")
            self.assertTrue(os.path.exists(mapie_path), f"Missing mapie_model_{ind}.joblib")
            
            model = joblib.load(model_path)
            mapie = joblib.load(mapie_path)
            
            schema_cols = INDUSTRY_SCHEMAS[ind]
            sample = {col: 0 for col in schema_cols}
            sample['tenure_months'] = 6
            sample['monthly_spend_usd'] = 95.0
            sample['contract_type'] = 'Month-to-month' if ind == 'telecom' else ('Basic' if ind == 'saas' else 'annual')
            sample['autopay_enabled'] = 0
            sample['support_tickets_90d'] = 3
            sample['nps_score'] = 4
            sample['signup_year'] = 2023
            sample['age'] = 40
            sample['region'] = 'West'
            sample['customer_segment'] = 'standard'
            
            df_sample = pd.DataFrame([sample])
            prob = float(model.predict_proba(df_sample)[0, 1])
            self.assertTrue(0.0 <= prob <= 1.0, f"Probability out of bounds: {prob}")
            
            _, pset = mapie.predict_set(df_sample)
            in_0 = bool(pset[0, 0, 2])
            in_1 = bool(pset[0, 1, 2])
            self.assertTrue(in_0 or in_1, "Conformal set cannot be empty under finite sample guarantee")
            
            precedents = self.explainer.explain_instance(df_sample, ind, top_k=3)
            self.assertEqual(len(precedents), 3, f"Expected 3 CBR cases for {ind}, got {len(precedents)}")
            for p in precedents:
                self.assertIn('similarity_pct', p)
                self.assertIn('actual_outcome', p)

    def test_02_batch_audit_and_fuzzy_schema(self):
        """Validates RapidFuzz automatic sector detection, column mapping, and live batch inference."""
        raw_telco_csv = os.path.join(BASE_DIR, 'WA_Fn-UseC_-Telco-Customer-Churn.csv')
        self.assertTrue(os.path.exists(raw_telco_csv), "Missing Telco benchmark CSV")
        
        demo_df = pd.read_csv(raw_telco_csv).head(50)
        detected_ind = detect_industry_fuzzy(demo_df.columns.tolist())
        self.assertEqual(detected_ind, 'telecom', f"Expected telecom, got {detected_ind}")
        
        detailed_map = map_columns_fuzzy(demo_df.columns.tolist(), detected_ind)
        mapped_cols = [v['target'] for v in detailed_map.values() if v['target']]
        self.assertGreaterEqual(len(mapped_cols), 7, "Fuzzy mapping failed to resolve standard attributes")
        
        simple_map = {k: v['target'] for k, v in detailed_map.items() if v['target']}
        df_clean = clean_mapped_features(demo_df.rename(columns=simple_map), detected_ind)
        
        m_telco = joblib.load(os.path.join(self.data_dir, 'model_telecom.joblib'))
        mapie_telco = joblib.load(os.path.join(self.data_dir, 'mapie_model_telecom.joblib'))
        
        batch_features = df_clean[[c for c in INDUSTRY_SCHEMAS['telecom'] if c in df_clean.columns]]
        batch_probs = m_telco.predict_proba(batch_features)[:, 1]
        _, batch_psets = mapie_telco.predict_set(batch_features)
        
        self.assertEqual(len(batch_probs), 50)
        self.assertEqual(len(batch_psets), 50)

    def test_03_whatif_simulator_sensitivity(self):
        """Validates retention levers sensitivity on the Stacking meta-learner."""
        m_telco = joblib.load(os.path.join(self.data_dir, 'model_telecom.joblib'))
        
        base_dict = {col: 0 for col in INDUSTRY_SCHEMAS['telecom']}
        base_dict['tenure_months'] = 4
        base_dict['monthly_spend_usd'] = 115.0
        base_dict['contract_type'] = 'Month-to-month'
        base_dict['autopay_enabled'] = 0
        base_dict['support_tickets_90d'] = 3
        base_dict['nps_score'] = 3
        
        interv_dict = base_dict.copy()
        interv_dict['contract_type'] = 'Two year'
        interv_dict['monthly_spend_usd'] = 115.0 * 0.8
        interv_dict['autopay_enabled'] = 1
        
        df_sim = pd.DataFrame([base_dict, interv_dict])
        sim_probs = m_telco.predict_proba(df_sim)[:, 1]
        prob_before = float(sim_probs[0])
        prob_after = float(sim_probs[1])
        
        self.assertLess(prob_after, prob_before, "Retention levers must reduce churn probability")

if __name__ == '__main__':
    unittest.main(verbosity=2)
