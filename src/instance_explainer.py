"""
LoyalScale Instance-Based Explainer (Case-Based Reasoning via k-NN)
===================================================================
Replaces the mislabeled "Tabular RAG" with technically accurate, statistically sound
Case-Based Reasoning (CBR) / Historical Exemplar Retrieval.

Key principles:
1. Retrieval is performed using Nearest Neighbors on normalized numerical features (StandardScaler).
2. The retrieved historical precedents provide concrete, verifiable case context for human decision-makers.
3. CRUCIAL: It does NOT tamper with or alter the model's calibrated conformal probabilities.
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from src.fuzzy_mapper import INDUSTRY_SCHEMAS

# Standard categoricals to strictly exclude from numerical distance calculations
CATEGORICAL_COLS = {
    'region', 'customer_segment', 'contract_type', 'acquisition_channel', 'plan_type',
    'loyalty_tier', 'store_preference', 'account_type', 'program_type', 'plan_category',
    'membership_level', 'policy_type', 'service_type'
}

class InstanceExplainer:
    def __init__(self, processed_data_dir: str = 'processed_data'):
        self.data_dir = processed_data_dir
        self.cache = {}

    def get_explainer_data(self, industry: str):
        """Loads, standardizes, and caches training features, targets, and raw readable profiles."""
        if industry in self.cache:
            return self.cache[industry]

        raw_path = os.path.join(self.data_dir, f'{industry}_augmented_raw.csv')
        y_path = os.path.join(self.data_dir, f'{industry}_augmented_y.csv')

        # Fallback to general filenames if industry-specific files are missing
        if not os.path.exists(raw_path):
            raw_path = os.path.join(self.data_dir, f'X_train_processed_{industry}.csv')
        if not os.path.exists(raw_path):
            raw_path = os.path.join(self.data_dir, 'telecom_augmented_raw.csv')
        if not os.path.exists(y_path):
            y_path = os.path.join(self.data_dir, f'y_train_{industry}.csv')
        if not os.path.exists(y_path):
            y_path = os.path.join(self.data_dir, 'y_train.csv')

        if not os.path.exists(raw_path):
            return None

        try:
            df_raw = pd.read_csv(raw_path)
            
            if os.path.exists(y_path):
                y_train = pd.read_csv(y_path).iloc[:, 0].astype(int)
            elif 'churned' in df_raw.columns:
                y_train = df_raw['churned'].astype(int)
            else:
                y_train = pd.Series([0] * len(df_raw))

            # Select expected schema features that are strictly non-categorical
            expected_schema = INDUSTRY_SCHEMAS.get(industry, INDUSTRY_SCHEMAS.get('telecom', []))
            numeric_cols = [
                col for col in expected_schema 
                if col in df_raw.columns 
                and col not in CATEGORICAL_COLS 
                and col not in ['customer_id', 'industry', 'churn_probability', 'churned']
            ]
            
            # If no schema match, fallback to numeric dtypes excluding categoricals
            if not numeric_cols:
                numeric_cols = [
                    c for c in df_raw.select_dtypes(include=[np.number]).columns 
                    if c not in CATEGORICAL_COLS and c not in ['customer_id', 'churned']
                ]

            # Coerce all numeric columns to float and fill missing values with median
            X_train_num = df_raw[numeric_cols].apply(pd.to_numeric, errors='coerce')
            medians = X_train_num.median().fillna(0.0)
            X_train_clean = X_train_num.fillna(medians)

            # Fit standard scaler to normalize distance metric space
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_clean)

            # Fit NearestNeighbors
            nn = NearestNeighbors(n_neighbors=min(10, len(X_train_scaled)), metric='euclidean')
            nn.fit(X_train_scaled)

            store = {
                'nn': nn,
                'scaler': scaler,
                'numeric_cols': numeric_cols,
                'medians': medians,
                'y_train': y_train,
                'df_raw': df_raw
            }
            self.cache[industry] = store
            return store
        except Exception as e:
            print(f"Error loading explainer data for {industry}: {e}")
            return None

    def explain_instance(self, feature_df_or_vector, industry: str, top_k: int = 3):
        """
        Finds top_k most similar historical customers and their observed churn outcomes.
        Standardizes query vector into the training feature space.
        Returns a list of dicts with human-interpretable details.
        """
        data = self.get_explainer_data(industry)
        if data is None:
            return []

        nn = data['nn']
        scaler = data['scaler']
        numeric_cols = data['numeric_cols']
        medians = data['medians']
        df_raw = data['df_raw']
        y_train = data['y_train']

        # Construct a clean DataFrame matching exact numeric columns of training space
        if isinstance(feature_df_or_vector, pd.DataFrame):
            query_df = pd.DataFrame(index=feature_df_or_vector.index)
            for c in numeric_cols:
                if c in feature_df_or_vector.columns:
                    query_df[c] = pd.to_numeric(feature_df_or_vector[c], errors='coerce')
                else:
                    query_df[c] = np.nan
            query_df = query_df.fillna(medians)
            query_scaled = scaler.transform(query_df.iloc[0:1])
        else:
            arr = np.asarray(feature_df_or_vector, dtype=float).flatten()
            if len(arr) < len(numeric_cols):
                full_arr = medians.values.copy()
                full_arr[:len(arr)] = arr
                arr = full_arr
            elif len(arr) > len(numeric_cols):
                arr = arr[:len(numeric_cols)]
            query_scaled = scaler.transform(arr.reshape(1, -1))

        k = min(top_k, len(df_raw))
        distances, indices = nn.kneighbors(query_scaled, n_neighbors=k)
        dist = distances[0]
        idxs = indices[0]

        dim_scale = np.sqrt(max(1, len(numeric_cols)))

        precedents = []
        for i, idx in enumerate(idxs):
            # Normalized distance-based similarity score (100% = identical vector)
            norm_dist = dist[i] / dim_scale
            similarity_pct = round(max(5.0, min(99.9, (1.0 / (1.0 + norm_dist)) * 100.0)), 1)
            
            actual_churn = int(y_train.iloc[idx]) if idx < len(y_train) else 0
            raw_row = df_raw.iloc[idx] if idx < len(df_raw) else {}
            
            tenure = raw_row.get('tenure_months', raw_row.get('tenure', 'N/A'))
            spend = raw_row.get('monthly_spend_usd', raw_row.get('MonthlyCharges', 'N/A'))
            contract = raw_row.get('contract_type', raw_row.get('Contract', 'N/A'))
            segment = raw_row.get('customer_segment', raw_row.get('Segment', 'Standard'))

            # Format tenure cleanly
            if isinstance(tenure, (int, float)) and not np.isnan(tenure):
                tenure_str = f"{int(tenure)} mo"
            else:
                tenure_str = str(tenure)

            # Format spend cleanly
            if isinstance(spend, (int, float)) and not np.isnan(spend):
                spend_str = f"${spend:.1f}"
            else:
                spend_str = str(spend)

            precedents.append({
                'rank': i + 1,
                'similarity_pct': similarity_pct,
                'actual_outcome': 'Churned' if actual_churn == 1 else 'Retained',
                'outcome_color': '#EF4444' if actual_churn == 1 else '#10B981',
                'tenure': tenure_str,
                'monthly_spend': spend_str,
                'contract': str(contract).capitalize(),
                'segment': str(segment).capitalize()
            })

        return precedents
