"""
LoyalScale Instance-Based Explainer (Case-Based Reasoning via k-NN)
===================================================================
Replaces the mislabeled "Tabular RAG" with technically accurate, statistically sound
Case-Based Reasoning (CBR) / Historical Exemplar Retrieval.

Key principles:
1. Retrieval is performed using Nearest Neighbors on normalized numerical features.
2. The retrieved historical precedents provide concrete, verifiable case context for human decision-makers.
3. CRUCIAL: It does NOT tamper with or alter the model's calibrated conformal probabilities.
"""

import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

class InstanceExplainer:
    def __init__(self, processed_data_dir: str = 'processed_data'):
        self.data_dir = processed_data_dir
        self.cache = {}

    def get_explainer_data(self, industry: str):
        """Loads and caches training features, targets, and raw readable profiles."""
        if industry in self.cache:
            return self.cache[industry]

        x_path = os.path.join(self.data_dir, f'X_train_processed_{industry}.csv')
        y_path = os.path.join(self.data_dir, f'y_train_{industry}.csv')
        raw_path = os.path.join(self.data_dir, f'{industry}_augmented_raw.csv')

        # Fallback to telecom standalone files if industry-specific files are missing
        if not os.path.exists(x_path):
            x_path = os.path.join(self.data_dir, 'X_train_processed.csv')
            y_path = os.path.join(self.data_dir, 'y_train.csv')

        if not (os.path.exists(x_path) and os.path.exists(y_path)):
            return None

        try:
            X_train = pd.read_csv(x_path)
            y_train = pd.read_csv(y_path).iloc[:, 0].astype(int)

            if os.path.exists(raw_path):
                df_raw = pd.read_csv(raw_path)
            else:
                df_raw = X_train.copy()
                df_raw['churned'] = y_train.values

            # Filter only numeric columns for Euclidean distance calculations
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            X_train_numeric = X_train[numeric_cols]

            nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
            nn.fit(X_train_numeric)

            store = {
                'nn': nn,
                'numeric_cols': numeric_cols,
                'X_train': X_train,
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
        Returns a list of dicts with human-interpretable details.
        """
        data = self.get_explainer_data(industry)
        if data is None:
            return []

        nn = data['nn']
        numeric_cols = data['numeric_cols']
        df_raw = data['df_raw']
        y_train = data['y_train']

        # Extract numeric columns matching training space
        if isinstance(feature_df_or_vector, pd.DataFrame):
            available_cols = [c for c in numeric_cols if c in feature_df_or_vector.columns]
            if len(available_cols) == len(numeric_cols):
                query_vec = feature_df_or_vector[numeric_cols].values
            else:
                # If subset or single row vector
                query_vec = feature_df_or_vector.iloc[0:1].select_dtypes(include=[np.number]).values
        else:
            query_vec = np.asarray(feature_df_or_vector)

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        # Slice to length of numeric_cols if dimension mismatch
        if query_vec.shape[1] > len(numeric_cols):
            query_vec = query_vec[:, :len(numeric_cols)]

        distances, indices = nn.kneighbors(query_vec, n_neighbors=min(top_k, len(df_raw)))
        dist = distances[0]
        idxs = indices[0]

        precedents = []
        for i, idx in enumerate(idxs):
            similarity_pct = round((1.0 / (1.0 + dist[i])) * 100.0, 1)
            actual_churn = int(y_train.iloc[idx])

            raw_row = df_raw.iloc[idx] if idx < len(df_raw) else {}
            
            tenure = raw_row.get('tenure_months', raw_row.get('tenure', 'N/A'))
            spend = raw_row.get('monthly_spend_usd', raw_row.get('MonthlyCharges', 'N/A'))
            contract = raw_row.get('contract_type', raw_row.get('Contract', 'N/A'))
            segment = raw_row.get('customer_segment', raw_row.get('Segment', 'Standard'))

            precedents.append({
                'rank': i + 1,
                'similarity_pct': similarity_pct,
                'actual_outcome': 'Churned' if actual_churn == 1 else 'Retained',
                'outcome_color': '#EF4444' if actual_churn == 1 else '#10B981',
                'tenure': f"{tenure} mo" if tenure != 'N/A' else 'N/A',
                'monthly_spend': f"${spend:.1f}" if isinstance(spend, (int, float)) else str(spend),
                'contract': str(contract),
                'segment': str(segment)
            })

        return precedents
