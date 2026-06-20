"""
LoyalScale Intelligent Feature Bridge
======================================

Transforms arbitrary real-world datasets with unknown column schemas into the
expected industry feature schema using a 3-tier approach:

  Tier 1: NLP-based column name mapping (cosine similarity + synonyms)
  Tier 2: Correlation-based statistical imputation for unmapped numeric features
  Tier 3: Proxy model ensemble for very low schema overlap scenarios

Usage:
    from src.feature_bridge import FeatureBridge

    bridge = FeatureBridge(industry='saas')  # or None for auto-detect
    df_bridged, bridge_report, column_mapping, proxy_scores = bridge.transform(df_user)
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Base directory setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from src.nlp_mapper import detect_industry, map_columns_nlp, INDUSTRY_SCHEMAS, SYNONYMS

logger = logging.getLogger("LoyalScaleFeatureBridge")

# ─────────────────────────────────────────────────────────────────────
# Feature type sets (canonical source — shared with train_all_industries)
# ─────────────────────────────────────────────────────────────────────
CONTINUOUS_FEATURES = {
    'monthly_spend_usd', 'discount_pct', 'data_usage_gb_30d', 'dropped_calls_30d',
    'avg_basket_usd', 'returns_90d', 'avg_balance_usd', 'products_count',
    'cart_abandon_rate', 'avg_order_value_usd', 'return_rate', 'completion_rate',
    'logins_30d', 'assignments_late_90d', 'appointments_12m', 'missed_appointments_12m',
    'avg_nightly_rate_usd', 'review_rating', 'reward_points_balance', 'cancellations_12m',
    'premium_usd', 'claims_24m', 'policy_count', 'renewal_days_remaining',
    'avg_monthly_usage', 'outages_12m', 'feature_adoption_score'
}

DISCRETE_FEATURES = {
    'signup_year', 'age', 'tenure_months', 'support_tickets_90d', 'complaints_90d',
    'nps_score', 'days_since_last_activity', 'late_payments_12m', 'seats_purchased',
    'active_users_30d', 'integrations_connected', 'admin_logins_30d', 'visits_90d',
    'coupons_used_90d', 'mobile_logins_30d', 'overdrafts_12m', 'branch_visits_90d',
    'orders_180d', 'app_sessions_30d', 'courses_enrolled', 'advisor_contacts_90d',
    'portal_logins_90d', 'care_gap_count', 'stays_12m', 'agent_contact_90d',
    'network_complaints_90d'
}

BINARY_FEATURES = {
    'autopay_enabled', 'device_financed', 'international_roaming', 'onboarding_completed',
    'free_shipping_member', 'primary_provider_assigned', 'smart_meter_enabled',
    'paperless_billing', 'move_flag_90d'
}

STRING_FEATURES = {
    'region', 'customer_segment', 'contract_type', 'acquisition_channel', 'plan_type',
    'loyalty_tier', 'store_preference', 'account_type', 'program_type', 'plan_category',
    'membership_level', 'policy_type', 'service_type'
}

# ─────────────────────────────────────────────────────────────────────
# Default fallback values (canonical source — imported by views.py)
# ─────────────────────────────────────────────────────────────────────
DEFAULT_VALUES = {
    # Common
    'signup_year': 2024, 'region': 'West', 'customer_segment': 'standard',
    'age': 40, 'tenure_months': 24, 'contract_type': 'annual',
    'monthly_spend_usd': 100.0, 'discount_pct': 0.05, 'autopay_enabled': 0,
    'support_tickets_90d': 1, 'complaints_90d': 0, 'nps_score': 8,
    'days_since_last_activity': 15, 'late_payments_12m': 0,
    'acquisition_channel': 'organic',
    # Telecom
    'plan_type': 'prepaid', 'data_usage_gb_30d': 10.0, 'dropped_calls_30d': 1,
    'network_complaints_90d': 0, 'device_financed': 0, 'international_roaming': 0,
    # SaaS
    'seats_purchased': 5, 'active_users_30d': 4, 'feature_adoption_score': 0.7,
    'integrations_connected': 3, 'admin_logins_30d': 10, 'onboarding_completed': 1,
    # Retail
    'loyalty_tier': 'silver', 'visits_90d': 6, 'avg_basket_usd': 50.0,
    'returns_90d': 0, 'coupons_used_90d': 2, 'store_preference': 'online',
    # Banking
    'account_type': 'checking', 'avg_balance_usd': 5000.0, 'products_count': 2,
    'mobile_logins_30d': 12, 'overdrafts_12m': 0, 'branch_visits_90d': 1,
    # eCommerce
    'orders_180d': 5, 'cart_abandon_rate': 0.3, 'avg_order_value_usd': 50.0,
    'return_rate': 0.05, 'app_sessions_30d': 10, 'free_shipping_member': 0,
    # Education
    'program_type': 'professional', 'courses_enrolled': 2, 'completion_rate': 0.8,
    'logins_30d': 15, 'assignments_late_90d': 0, 'advisor_contacts_90d': 1,
    # Healthcare
    'plan_category': 'standard', 'appointments_12m': 3, 'missed_appointments_12m': 0,
    'portal_logins_90d': 4, 'care_gap_count': 0, 'primary_provider_assigned': 1,
    # Hospitality
    'membership_level': 'member', 'stays_12m': 2, 'avg_nightly_rate_usd': 120.0,
    'review_rating': 4.5, 'reward_points_balance': 1000, 'cancellations_12m': 0,
    # Insurance
    'policy_type': 'auto', 'premium_usd': 150.0, 'claims_24m': 0,
    'policy_count': 1, 'agent_contact_90d': 1, 'renewal_days_remaining': 180,
    # Utilities
    'service_type': 'electricity', 'avg_monthly_usage': 350.0, 'outages_12m': 1,
    'smart_meter_enabled': 1, 'paperless_billing': 1, 'move_flag_90d': 0
}

# ─────────────────────────────────────────────────────────────────────
# Target column synonyms & value mapping
# ─────────────────────────────────────────────────────────────────────
TARGET_SYNONYMS = {
    'churn': 'churned', 'Churn': 'churned', 'churned': 'churned',
    'Churned': 'churned', 'exited': 'churned', 'Exited': 'churned',
    'is_churned': 'churned', 'IsChurned': 'churned',
    'attrition': 'churned', 'Attrition_Flag': 'churned',
    'target': 'churned', 'label': 'churned',
    'churn_label': 'churned', 'churn_flag': 'churned',
}

TARGET_VALUE_MAP = {
    'Yes': 1, 'No': 0, 'yes': 1, 'no': 0,
    'True': 1, 'False': 0, 'true': 1, 'false': 0,
    'YES': 1, 'NO': 0,
    'Existing Customer': 0, 'Attrited Customer': 1,
    '1': 1, '0': 0, 1: 1, 0: 0,
}

# ─────────────────────────────────────────────────────────────────────
# Data directory resolution (matches train_all_industries.py logic)
# ─────────────────────────────────────────────────────────────────────
def _resolve_data_dir():
    """Resolve the mock churn training data directory."""
    data_dir = os.getenv('CHURN_DATA_DIR')
    if data_dir and os.path.exists(data_dir):
        return data_dir
    
    fallback_paths = [
        os.path.join(BASE_DIR, 'mock_churn_data'),
        os.path.join(os.path.dirname(BASE_DIR), 'forMock', 'mock_churn_data')
    ]
    for path in fallback_paths:
        if os.path.exists(path):
            return path
    return fallback_paths[0]


# ═════════════════════════════════════════════════════════════════════
# Tier 2: Correlation-Based Statistical Imputer
# ═════════════════════════════════════════════════════════════════════
class CorrelationImputer:
    """
    Learns inter-feature correlations from training data to predict
    missing feature values using available ones.

    For each numeric feature in the schema, fits a Ridge regression
    using all other numeric features as predictors. At inference time,
    predicts missing features from whatever features ARE available.

    Categorical features fall back to the training data's mode.
    """

    def __init__(self, industry):
        self.industry = industry
        self.numeric_models = {}   # {feature: {'model': Ridge, 'predictors': [cols], 'medians': {col: median}}}
        self.categorical_modes = {}
        self.feature_medians = {}
        self.is_fitted = False

    def fit(self, df_train):
        """Fit imputation models from training data."""
        schema = INDUSTRY_SCHEMAS.get(self.industry, [])

        # Classify schema features by type
        numeric_cols = [
            c for c in schema
            if (c in CONTINUOUS_FEATURES or c in DISCRETE_FEATURES or c in BINARY_FEATURES)
            and c in df_train.columns
        ]
        categorical_cols = [
            c for c in schema
            if c in STRING_FEATURES and c in df_train.columns
        ]

        # Store medians for numeric features (used as both imputation fallback and predictor fill)
        for col in numeric_cols:
            val = pd.to_numeric(df_train[col], errors='coerce').median()
            self.feature_medians[col] = float(val) if not pd.isna(val) else 0.0

        # Store modes for categorical features
        for col in categorical_cols:
            mode = df_train[col].mode()
            self.categorical_modes[col] = str(mode.iloc[0]) if not mode.empty else DEFAULT_VALUES.get(col, 'unknown')

        # Fit Ridge regression for each numeric feature using all OTHER numeric features
        for target_col in numeric_cols:
            predictor_cols = [c for c in numeric_cols if c != target_col]
            if len(predictor_cols) < 2:
                continue

            X = df_train[predictor_cols].apply(pd.to_numeric, errors='coerce')
            # Fill NaNs with medians for stable fitting
            for pc in predictor_cols:
                X[pc] = X[pc].fillna(self.feature_medians.get(pc, 0))

            y = pd.to_numeric(df_train[target_col], errors='coerce').fillna(self.feature_medians.get(target_col, 0))

            try:
                model = Ridge(alpha=1.0)
                model.fit(X, y)

                # Store predictor medians for filling at inference
                predictor_medians = {pc: self.feature_medians.get(pc, 0) for pc in predictor_cols}

                self.numeric_models[target_col] = {
                    'model': model,
                    'predictors': predictor_cols,
                    'predictor_medians': predictor_medians
                }
            except Exception as e:
                logger.warning(f"CorrelationImputer: Could not fit model for {target_col}: {e}")

        self.is_fitted = True
        logger.info(
            f"CorrelationImputer fitted for {self.industry}: "
            f"{len(self.numeric_models)} numeric models, "
            f"{len(self.categorical_modes)} categorical modes"
        )
        return self

    def impute(self, df_mapped, available_cols, missing_cols):
        """
        Predict missing feature values from available features.

        Args:
            df_mapped: DataFrame with NLP-mapped columns (may not have all schema cols)
            available_cols: List of schema feature names that ARE present in df_mapped
            missing_cols: List of schema feature names that are MISSING

        Returns:
            (imputed_values, impute_sources) — dict of {col: values}, dict of {col: source_label}
        """
        imputed = {}
        impute_sources = {}

        available_numeric = set(available_cols) & (CONTINUOUS_FEATURES | DISCRETE_FEATURES | BINARY_FEATURES)

        for col in missing_cols:
            # Try correlation-based prediction for numeric features
            if col in self.numeric_models:
                model_info = self.numeric_models[col]

                # Check how many predictor columns are actually available from user data
                available_preds = [p for p in model_info['predictors'] if p in available_numeric]

                if len(available_preds) >= 2:
                    # Build the prediction input matrix
                    n_rows = len(df_mapped)
                    X_input = pd.DataFrame(index=range(n_rows))

                    for pc in model_info['predictors']:
                        if pc in available_cols and pc in df_mapped.columns:
                            X_input[pc] = pd.to_numeric(
                                df_mapped[pc].values, errors='coerce'
                            )
                            X_input[pc] = X_input[pc].fillna(model_info['predictor_medians'].get(pc, 0))
                        else:
                            # Predictor not available from user → use training median
                            X_input[pc] = model_info['predictor_medians'].get(pc, 0)

                    predictions = model_info['model'].predict(X_input)

                    # Clip to reasonable range (non-negative for count/score features)
                    if col in DISCRETE_FEATURES or col in BINARY_FEATURES:
                        predictions = np.clip(predictions, 0, None).round().astype(int)
                    elif col in CONTINUOUS_FEATURES:
                        predictions = np.clip(predictions, 0, None)

                    imputed[col] = predictions
                    impute_sources[col] = 'correlation'
                    continue

            # Fallback: training median for numeric, mode for categorical
            if col in self.feature_medians:
                imputed[col] = self.feature_medians[col]
                impute_sources[col] = 'training_median'
            elif col in self.categorical_modes:
                imputed[col] = self.categorical_modes[col]
                impute_sources[col] = 'training_mode'
            else:
                imputed[col] = DEFAULT_VALUES.get(col, 0)
                impute_sources[col] = 'static_default'

        return imputed, impute_sources


# ═════════════════════════════════════════════════════════════════════
# Tier 3: Proxy Model Bridge
# ═════════════════════════════════════════════════════════════════════
class ProxyModelBridge:
    """
    When schema overlap is very low, trains a lightweight XGBoost directly
    on the user's raw features → churn target. The proxy model's predicted
    probabilities serve as a blending signal with the main industry model.

    This captures information from user columns that have NO semantic
    equivalent in the industry schema (e.g., CashbackAmount, WarehouseToHome).
    """

    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.auc_score = None
        self.numeric_cols = []

    def fit(self, df_user, target_col='churned'):
        """Train a quick XGBoost on user's raw numeric features → target."""
        if target_col not in df_user.columns:
            logger.warning("ProxyModelBridge: No target column found. Skipping.")
            return self

        y = df_user[target_col].values
        X = df_user.drop(columns=[target_col], errors='ignore')

        # Select only numeric columns for the proxy model
        self.numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(self.numeric_cols) < 2:
            logger.warning("ProxyModelBridge: Not enough numeric features. Skipping.")
            return self

        X_numeric = X[self.numeric_cols].fillna(0)

        # Ensure we have enough samples and both classes
        if len(X_numeric) < 50 or len(np.unique(y)) < 2:
            logger.warning("ProxyModelBridge: Insufficient data for proxy model training.")
            return self

        try:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_numeric, y, test_size=0.2, random_state=42, stratify=y
            )

            self.model = xgb.XGBClassifier(
                n_estimators=50, max_depth=3, learning_rate=0.1,
                random_state=42, eval_metric='logloss', n_jobs=1,
                use_label_encoder=False
            )
            self.model.fit(X_tr, y_tr)

            y_pred_proba = self.model.predict_proba(X_val)[:, 1]
            self.auc_score = float(roc_auc_score(y_val, y_pred_proba))
            self.is_fitted = True

            logger.info(
                f"ProxyModelBridge trained successfully. "
                f"AUC: {self.auc_score:.4f}, Features: {len(self.numeric_cols)}"
            )
        except Exception as e:
            logger.error(f"ProxyModelBridge training failed: {e}")

        return self

    def predict_proba(self, df_user):
        """Return proxy churn probabilities for the user's raw data."""
        if not self.is_fitted or self.model is None:
            return None

        available = [c for c in self.numeric_cols if c in df_user.columns]
        if len(available) < 2:
            return None

        X = pd.DataFrame(index=range(len(df_user)))
        for col in self.numeric_cols:
            if col in df_user.columns:
                X[col] = pd.to_numeric(df_user[col], errors='coerce').fillna(0).values
            else:
                X[col] = 0

        return self.model.predict_proba(X)[:, 1]


# ═════════════════════════════════════════════════════════════════════
# Cache for fitted CorrelationImputers (one per industry)
# ═════════════════════════════════════════════════════════════════════
_CORRELATION_IMPUTER_CACHE = {}


def _get_correlation_imputer(industry):
    """Lazily fit and cache a CorrelationImputer for the given industry."""
    global _CORRELATION_IMPUTER_CACHE

    if industry in _CORRELATION_IMPUTER_CACHE:
        return _CORRELATION_IMPUTER_CACHE[industry]

    data_dir = _resolve_data_dir()
    train_path = os.path.join(data_dir, 'train', f'{industry}_churn_train.csv')

    if not os.path.exists(train_path):
        logger.warning(f"Training data not found for {industry}: {train_path}")
        return None

    try:
        df_train = pd.read_csv(train_path)
        imputer = CorrelationImputer(industry)
        imputer.fit(df_train)
        _CORRELATION_IMPUTER_CACHE[industry] = imputer
        return imputer
    except Exception as e:
        logger.error(f"Failed to fit CorrelationImputer for {industry}: {e}")
        return None


# ═════════════════════════════════════════════════════════════════════
# Main Orchestrator: FeatureBridge
# ═════════════════════════════════════════════════════════════════════
class FeatureBridge:
    """
    Orchestrator that transforms arbitrary user DataFrames into the expected
    industry feature schema using a 3-tier approach:

      Tier 1: NLP column mapping (cosine similarity + synonyms)
      Tier 2: Correlation-based statistical imputation
      Tier 3: Proxy model ensemble (low overlap + target available)

    Usage:
        bridge = FeatureBridge(industry='saas')  # or None for auto-detect
        df_bridged, report, mapping, proxy_scores = bridge.transform(df_user)
    """

    def __init__(self, industry=None):
        self.industry = industry
        self.proxy_model = None

    def transform(self, df_user):
        """
        Transform a user DataFrame into the industry schema.

        Args:
            df_user: Raw user DataFrame with arbitrary columns.

        Returns:
            df_bridged: DataFrame with columns matching INDUSTRY_SCHEMAS[industry]
            bridge_report: Dict with mapping transparency and quality metrics
            column_mapping: Dict {user_col: schema_col}
            proxy_scores: np.array of proxy churn probabilities or None
        """
        headers = df_user.columns.tolist()

        # ── Tier 1: Industry Detection & NLP Column Mapping ──────────
        if self.industry is None:
            self.industry = detect_industry(headers)
            logger.info(f"FeatureBridge: Auto-detected industry as '{self.industry}'")

        # NLP mapping
        column_mapping = map_columns_nlp(headers, self.industry)

        # Detect and map target column
        detected_target_col = None
        for user_col in headers:
            if user_col in TARGET_SYNONYMS and user_col not in column_mapping:
                column_mapping[user_col] = TARGET_SYNONYMS[user_col]
                detected_target_col = user_col
                break

        # Apply column renaming
        df_mapped = df_user.rename(columns=column_mapping)

        # Convert target values (Yes/No → 1/0)
        if 'churned' in df_mapped.columns and df_mapped['churned'].dtype == object:
            df_mapped['churned'] = df_mapped['churned'].map(TARGET_VALUE_MAP).fillna(0).astype(int)
        elif 'churned' in df_mapped.columns:
            df_mapped['churned'] = pd.to_numeric(df_mapped['churned'], errors='coerce').fillna(0).astype(int)

        # Handle signup_date → signup_year extraction
        if 'signup_year' in df_mapped.columns:
            sample = df_mapped['signup_year'].dropna().iloc[0] if not df_mapped['signup_year'].dropna().empty else None
            if sample is not None and isinstance(sample, str) and '-' in sample:
                try:
                    df_mapped['signup_year'] = pd.to_datetime(df_mapped['signup_year'], errors='coerce').dt.year
                except Exception:
                    pass

        # ── Compute Schema Overlap ───────────────────────────────────
        schema_features = INDUSTRY_SCHEMAS[self.industry]
        mapped_features = [col for col in schema_features if col in df_mapped.columns]
        missing_features = [col for col in schema_features if col not in df_mapped.columns]
        schema_overlap = len(mapped_features) / len(schema_features) if schema_features else 0.0

        logger.info(
            f"FeatureBridge [{self.industry}]: Schema overlap {schema_overlap:.0%} "
            f"({len(mapped_features)}/{len(schema_features)} features mapped, "
            f"{len(missing_features)} missing)"
        )

        # ── Tier 2: Correlation-Based Imputation ─────────────────────
        impute_sources = {}
        correlation_imputer = _get_correlation_imputer(self.industry)

        if correlation_imputer and correlation_imputer.is_fitted and missing_features:
            imputed_values, impute_sources = correlation_imputer.impute(
                df_mapped, mapped_features, missing_features
            )
            for col, values in imputed_values.items():
                df_mapped[col] = values
                if col not in impute_sources:
                    impute_sources[col] = 'correlation'
        else:
            for col in missing_features:
                df_mapped[col] = DEFAULT_VALUES.get(col, 0)
                impute_sources[col] = 'static_default'

        # ── Tier 3: Proxy Model (low overlap + target available) ─────
        proxy_scores = None
        proxy_auc = None

        if (schema_overlap < 0.40
                and 'churned' in df_mapped.columns
                and len(df_user) >= 50):

            logger.info(
                f"FeatureBridge: Low schema overlap ({schema_overlap:.0%}). "
                f"Training proxy model on user's raw features..."
            )

            # Prepare user data for proxy model (use original columns, not mapped)
            df_proxy = df_user.copy()
            if detected_target_col and detected_target_col in df_proxy.columns:
                df_proxy = df_proxy.rename(columns={detected_target_col: 'churned'})
                if df_proxy['churned'].dtype == object:
                    df_proxy['churned'] = df_proxy['churned'].map(TARGET_VALUE_MAP).fillna(0).astype(int)
                else:
                    df_proxy['churned'] = pd.to_numeric(df_proxy['churned'], errors='coerce').fillna(0).astype(int)

            # Drop non-numeric ID columns
            id_cols = [c for c in df_proxy.columns
                       if any(kw in c.lower() for kw in ['id', 'name', 'email', 'phone', 'address', 'date'])]
            df_proxy = df_proxy.drop(columns=id_cols, errors='ignore')

            self.proxy_model = ProxyModelBridge()
            self.proxy_model.fit(df_proxy, target_col='churned')

            if self.proxy_model.is_fitted:
                # Predict on original user data (not mapped)
                df_predict = df_user.drop(columns=id_cols, errors='ignore')
                proxy_scores = self.proxy_model.predict_proba(df_predict)
                proxy_auc = self.proxy_model.auc_score

        # ── Build Final Schema-Ordered DataFrame ─────────────────────
        df_bridged = pd.DataFrame()
        for col in schema_features:
            if col in df_mapped.columns:
                df_bridged[col] = df_mapped[col].values
            else:
                df_bridged[col] = DEFAULT_VALUES.get(col, 0)
                impute_sources[col] = 'static_default'

        # ── Build Bridge Report ──────────────────────────────────────
        correlation_count = sum(1 for s in impute_sources.values() if s == 'correlation')
        median_count = sum(1 for s in impute_sources.values() if s == 'training_median')
        mode_count = sum(1 for s in impute_sources.values() if s == 'training_mode')
        default_count = sum(1 for s in impute_sources.values() if s == 'static_default')

        bridge_report = {
            'detected_industry': self.industry,
            'schema_overlap_pct': round(schema_overlap * 100, 1),
            'total_schema_features': len(schema_features),
            'mapped_features_count': len(mapped_features),
            'mapped_features': mapped_features,
            'imputed_features_count': len(missing_features),
            'imputation_breakdown': {
                'correlation_imputed': correlation_count,
                'training_median': median_count,
                'training_mode': mode_count,
                'static_default': default_count
            },
            'column_mapping': {k: v for k, v in column_mapping.items() if v in schema_features or v == 'churned'},
            'imputation_sources': impute_sources,
            'missing_features': missing_features,
            'proxy_model': {
                'used': proxy_scores is not None,
                'auc': round(proxy_auc, 4) if proxy_auc is not None else None,
                'reason': (
                    f"Schema overlap ({schema_overlap:.0%}) below 40% threshold"
                    if proxy_scores is not None else None
                )
            },
            'data_quality_warnings': []
        }

        # Generate quality warnings
        if schema_overlap < 0.30:
            bridge_report['data_quality_warnings'].append(
                f"⚠️ Very low schema overlap ({bridge_report['schema_overlap_pct']}%). "
                f"Predictions may be unreliable. Consider a dataset closer to the "
                f"{self.industry} domain."
            )
        elif schema_overlap < 0.50:
            bridge_report['data_quality_warnings'].append(
                f"⚠️ Moderate schema overlap ({bridge_report['schema_overlap_pct']}%). "
                f"{len(missing_features)} features were statistically imputed."
            )

        if correlation_count > 0:
            bridge_report['data_quality_warnings'].append(
                f"ℹ️ {correlation_count} feature(s) were imputed using inter-feature "
                f"correlations learned from the training data."
            )

        if default_count > 3:
            bridge_report['data_quality_warnings'].append(
                f"⚠️ {default_count} feature(s) fell back to static defaults. "
                f"This may reduce prediction accuracy."
            )

        if proxy_scores is not None and proxy_auc is not None:
            if proxy_auc > 0.65:
                bridge_report['data_quality_warnings'].append(
                    f"✅ Proxy model trained on raw user features achieved "
                    f"AUC={proxy_auc:.3f}. Predictions will be blended."
                )
            else:
                bridge_report['data_quality_warnings'].append(
                    f"⚠️ Proxy model AUC is low ({proxy_auc:.3f}). "
                    f"Blending weight will be reduced."
                )

        return df_bridged, bridge_report, column_mapping, proxy_scores

    def transform_single_row(self, row_dict):
        """
        Transform a single row dict into a schema-compliant profile dict.
        Uses the same Tier 1 + Tier 2 pipeline (no proxy model for single rows).

        Args:
            row_dict: Dict of {user_column: value}

        Returns:
            profile: Dict of {schema_column: value} ready for prediction
            bridge_report: Dict with mapping transparency
            column_mapping: Dict {user_col: schema_col}
        """
        df_single = pd.DataFrame([row_dict])
        df_bridged, bridge_report, column_mapping, _ = self.transform(df_single)

        # Convert back to dict for single-row prediction
        profile = df_bridged.iloc[0].to_dict()
        return profile, bridge_report, column_mapping


def blend_predictions(main_probs, proxy_scores, schema_overlap):
    """
    Blend main model probabilities with proxy model scores.

    The blending weight is proportional to schema overlap:
    - High overlap → trust main model more
    - Low overlap  → give more weight to proxy model

    Args:
        main_probs: np.array from main industry model
        proxy_scores: np.array from proxy model (or None)
        schema_overlap: float [0, 1] from bridge report

    Returns:
        np.array of blended probabilities
    """
    if proxy_scores is None:
        return main_probs

    # Alpha = schema_overlap means: more overlap → more weight on main model
    # Clamp alpha between 0.3 and 0.9 to avoid extreme weighting
    alpha = np.clip(schema_overlap, 0.3, 0.9)

    blended = alpha * np.array(main_probs) + (1 - alpha) * np.array(proxy_scores)
    return np.clip(blended, 0, 1)
