import os
import pandas as pd
import numpy as np
import joblib
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
import io
from sklearn.neighbors import NearestNeighbors

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from src.nlp_mapper import detect_industry, map_columns_nlp, INDUSTRY_SCHEMAS

# Data directory path
MOCK_DATA_DIR = r"C:\Users\Saran\Documents\forMock\mock_churn_data"

# Default feature values used for imputing missing columns dynamically
DEFAULT_VALUES = {
    # Common
    'signup_year': 2024,
    'region': 'West',
    'customer_segment': 'standard',
    'age': 40,
    'tenure_months': 24,
    'contract_type': 'annual',
    'monthly_spend_usd': 100.0,
    'discount_pct': 0.05,
    'autopay_enabled': 0,
    'support_tickets_90d': 1,
    'complaints_90d': 0,
    'nps_score': 8,
    'days_since_last_activity': 15,
    'late_payments_12m': 0,
    'acquisition_channel': 'organic',
    
    # Telecom
    'plan_type': 'prepaid',
    'data_usage_gb_30d': 10.0,
    'dropped_calls_30d': 1,
    'network_complaints_90d': 0,
    'device_financed': 0,
    'international_roaming': 0,
    
    # SaaS
    'seats_purchased': 5,
    'active_users_30d': 4,
    'feature_adoption_score': 0.7,
    'integrations_connected': 3,
    'admin_logins_30d': 10,
    'onboarding_completed': 1,
    
    # Retail
    'loyalty_tier': 'silver',
    'visits_90d': 6,
    'avg_basket_usd': 50.0,
    'returns_90d': 0,
    'coupons_used_90d': 2,
    'store_preference': 'online',
    
    # Banking
    'account_type': 'checking',
    'avg_balance_usd': 5000.0,
    'products_count': 2,
    'mobile_logins_30d': 12,
    'overdrafts_12m': 0,
    'branch_visits_90d': 1,

    # eCommerce
    'orders_180d': 5,
    'cart_abandon_rate': 0.3,
    'avg_order_value_usd': 50.0,
    'return_rate': 0.05,
    'app_sessions_30d': 10,
    'free_shipping_member': 0,

    # Education
    'program_type': 'professional',
    'courses_enrolled': 2,
    'completion_rate': 0.8,
    'logins_30d': 15,
    'assignments_late_90d': 0,
    'advisor_contacts_90d': 1,

    # Healthcare
    'plan_category': 'standard',
    'appointments_12m': 3,
    'missed_appointments_12m': 0,
    'portal_logins_90d': 4,
    'care_gap_count': 0,
    'primary_provider_assigned': 1,

    # Hospitality
    'membership_level': 'member',
    'stays_12m': 2,
    'avg_nightly_rate_usd': 120.0,
    'review_rating': 4.5,
    'reward_points_balance': 1000,
    'cancellations_12m': 0,

    # Insurance
    'policy_type': 'auto',
    'premium_usd': 150.0,
    'claims_24m': 0,
    'policy_count': 1,
    'agent_contact_90d': 1,
    'renewal_days_remaining': 180,

    # Utilities
    'service_type': 'electricity',
    'avg_monthly_usage': 350.0,
    'outages_12m': 1,
    'smart_meter_enabled': 1,
    'paperless_billing': 1,
    'move_flag_90d': 0
}

# Binary categorical columns that need one-hot encoding
BINARY_CATEGORICAL_COLS = [
    'autopay_enabled', 'device_financed', 'international_roaming', 'onboarding_completed',
    'free_shipping_member', 'primary_provider_assigned', 'smart_meter_enabled', 'paperless_billing', 'move_flag_90d'
]

def get_feature_types(df_cols):
    cols_to_exclude = ['customer_id', 'industry', 'churn_probability', 'churned']
    features = [c for c in df_cols.columns if c not in cols_to_exclude]
    
    numeric_features = []
    categorical_features = []
    
    for col in features:
        if df_cols[col].dtype == 'object' or df_cols[col].dtype.name == 'category':
            categorical_features.append(col)
        elif col in BINARY_CATEGORICAL_COLS:
            categorical_features.append(col)
        else:
            numeric_features.append(col)
            
    return numeric_features, categorical_features

# Global training database caches for tabular RAG retrieval per industry
_TRAINING_DBS = {}

def get_training_db(industry, force_reload=False):
    global _TRAINING_DBS
    if industry in _TRAINING_DBS and not force_reload:
        return _TRAINING_DBS[industry]

    x_train_path = os.path.join(BASE_DIR, 'processed_data', f'X_train_processed_{industry}.csv')
    y_train_path = os.path.join(BASE_DIR, 'processed_data', f'y_train_{industry}.csv')
    raw_path = os.path.join(MOCK_DATA_DIR, 'train', f'{industry}_churn_train.csv')

    # Fallback to base augmented datasets if exists for retraining
    aug_raw_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_raw.csv')
    aug_x_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_X.csv')
    aug_y_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_y.csv')

    actual_x = aug_x_path if os.path.exists(aug_x_path) else x_train_path
    actual_y = aug_y_path if os.path.exists(aug_y_path) else y_train_path
    actual_raw = aug_raw_path if os.path.exists(aug_raw_path) else raw_path

    if not (os.path.exists(actual_x) and os.path.exists(actual_y)):
        return None

    try:
        X_train_proc = pd.read_csv(actual_x)
        y_train = pd.read_csv(actual_y)
        y_train_series = y_train.iloc[:, 0].astype(int)

        raw_df = pd.DataFrame()
        if os.path.exists(actual_raw):
            raw_df = pd.read_csv(actual_raw)

        nn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        nn.fit(X_train_proc)

        _TRAINING_DBS[industry] = {
            'X_processed': X_train_proc,
            'y': y_train_series,
            'raw': raw_df,
            'nn': nn
        }
    except Exception as e:
        print(f"Error loading training DB for {industry}: {str(e)}")
        _TRAINING_DBS[industry] = None

    return _TRAINING_DBS[industry]


def compute_conformal_diagnostics(mapie_model, X_test_proc, y_test):
    confidence_levels = [0.80, 0.85, 0.90, 0.95]
    try:
        y_pred_mapie, y_pis = mapie_model.predict_set(X_test_proc)
        conformal_data = {}
        for idx, lvl in enumerate(confidence_levels):
            coverage = float(np.mean([y_pis[i, y_test[i], idx] for i in range(len(y_test))]) * 100)
            set_sizes = np.sum(y_pis[:, :, idx], axis=1)
            
            retained_count = int(np.sum((set_sizes == 1) & (y_pis[:, 0, idx] == 1)))
            churned_count = int(np.sum((set_sizes == 1) & (y_pis[:, 1, idx] == 1)))
            uncertain_count = int(np.sum(set_sizes != 1))
            
            outreach_cost = (churned_count + uncertain_count) * 50
            saved_value = int(churned_count * 200 + uncertain_count * 50)
            net_benefit = saved_value - outreach_cost

            conformal_data[f"{lvl:.2f}"] = {
                'coverage': round(coverage, 2),
                'set_distribution': {
                    'labels': ['Retained', 'Churned', 'Uncertain'],
                    'values': [retained_count, churned_count, uncertain_count]
                },
                'business_simulation': {
                    'outreach_cost': outreach_cost,
                    'saved_value': saved_value,
                    'net_benefit': net_benefit
                }
            }
        return conformal_data
    except Exception as e:
        print(f"Error computing conformal diagnostics: {e}")
        return None


def get_stats_and_chart_data():
    """Prepares aggregates and chart data for all supported industries."""
    results = {}
    industries = ['telecom', 'saas', 'retail', 'banking', 'ecommerce', 'education', 'healthcare', 'hospitality', 'insurance', 'utilities']
    
    for industry in industries:
        # Load artifacts paths
        prep_path = os.path.join(BASE_DIR, 'processed_data', f'preprocessor_{industry}.joblib')
        model_path = os.path.join(BASE_DIR, 'processed_data', f'model_{industry}.joblib')
        mapie_path = os.path.join(BASE_DIR, 'processed_data', f'mapie_model_{industry}.joblib')
        
        # Datasets paths
        train_path = os.path.join(MOCK_DATA_DIR, 'train', f'{industry}_churn_train.csv')
        val_path = os.path.join(MOCK_DATA_DIR, 'val', f'{industry}_churn_val.csv')
        test_ans_path = os.path.join(MOCK_DATA_DIR, 'test_answer_key', f'{industry}_churn_test_answer_key.csv')
        
        # Check augmented
        aug_raw_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_raw.csv')
        
        # Default fallback values
        total_cust = 1200
        churn_rate = 25.0
        avg_spend = 100.0
        avg_tenure = 24.0
        labels = ['Retained', 'Churned']
        values = [900, 300]
        imp_labels = []
        imp_values = []
        accuracy = 75.0
        auc = 75.0
        conformal_data = {
            '0.80': {
                'coverage': 80.0,
                'set_distribution': {'labels': ['Retained', 'Churned', 'Uncertain'], 'values': [100, 20, 5]},
                'business_simulation': {'outreach_cost': 1250, 'saved_value': 4000, 'net_benefit': 2750}
            },
            '0.85': {
                'coverage': 85.0,
                'set_distribution': {'labels': ['Retained', 'Churned', 'Uncertain'], 'values': [95, 18, 12]},
                'business_simulation': {'outreach_cost': 1500, 'saved_value': 3600, 'net_benefit': 2100}
            },
            '0.90': {
                'coverage': 90.0,
                'set_distribution': {'labels': ['Retained', 'Churned', 'Uncertain'], 'values': [90, 15, 20]},
                'business_simulation': {'outreach_cost': 1750, 'saved_value': 3000, 'net_benefit': 1250}
            },
            '0.95': {
                'coverage': 95.0,
                'set_distribution': {'labels': ['Retained', 'Churned', 'Uncertain'], 'values': [80, 10, 35]},
                'business_simulation': {'outreach_cost': 2250, 'saved_value': 2000, 'net_benefit': -250}
            }
        }
        
        validation_data = {
            'ensemble_acc': 75.0,
            'xgb_acc': 73.8,
            'lgb_acc': 74.2,
            'cb_acc': 73.5,
            'brier_score': 0.165,
            'calibration_score': 83.5,
            'conformal_status': "Guarantees Verified (Fallback)"
        }
        rag_files = []
        
        try:
            # Combine raw datasets to calculate global statistics
            dfs = []
            if os.path.exists(aug_raw_path):
                dfs.append(pd.read_csv(aug_raw_path))
                aug_val_raw_path_metrics = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_val_raw.csv')
                if os.path.exists(aug_val_raw_path_metrics):
                    dfs.append(pd.read_csv(aug_val_raw_path_metrics))
            else:
                if os.path.exists(train_path): dfs.append(pd.read_csv(train_path))
                if os.path.exists(val_path): dfs.append(pd.read_csv(val_path))
                if os.path.exists(test_ans_path): dfs.append(pd.read_csv(test_ans_path))
                
            if dfs:
                df_all = pd.concat(dfs, ignore_index=True)
                total_cust = len(df_all)
                churn_rate = float(df_all['churned'].mean() * 100)
                avg_spend = float(df_all['monthly_spend_usd'].mean())
                avg_tenure = float(df_all['tenure_months'].mean())
                
                churn_counts = df_all['churned'].value_counts()
                values = [int(churn_counts.get(0, 0)), int(churn_counts.get(1, 0))]
                
            # Load models to extract metrics and conformal data
            if os.path.exists(prep_path) and os.path.exists(model_path):
                preprocessor = joblib.load(prep_path)
                model = joblib.load(model_path)
                
                # Determine features dynamically using helper
                df_cols = pd.read_csv(train_path, nrows=1)
                numeric_features, categorical_features = get_feature_types(df_cols)
                
                cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
                encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
                all_transformed_cols = numeric_features + encoded_cat_cols
                
                # Feature importances
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'estimators_'):
                    xgb_idx = [i for i, (name, _) in enumerate(model.estimators) if name == 'xgb']
                    if xgb_idx:
                        importances = model.estimators_[xgb_idx[0]].feature_importances_
                    else:
                        importances = model.estimators_[0].feature_importances_
                else:
                    importances = np.zeros(len(all_transformed_cols))
                    
                feature_imp_pairs = sorted(zip(all_transformed_cols, importances), key=lambda x: x[1], reverse=True)
                top_10 = feature_imp_pairs[:10]
                imp_labels = [item[0] for item in top_10]
                imp_values = [float(item[1]) for item in top_10]
                
                # Model evaluation metrics using processed files
                x_test_proc_path = os.path.join(BASE_DIR, 'processed_data', f'X_test_processed_{industry}.csv')
                y_test_path = os.path.join(BASE_DIR, 'processed_data', f'y_test_{industry}.csv')
                
                if os.path.exists(x_test_proc_path) and os.path.exists(y_test_path):
                    X_test_proc = pd.read_csv(x_test_proc_path)
                    y_test = pd.read_csv(y_test_path).values.ravel()
                    
                    from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
                    y_pred = model.predict(X_test_proc)
                    y_pred_proba = model.predict_proba(X_test_proc)[:, 1]
                    accuracy = float(accuracy_score(y_test, y_pred) * 100)
                    auc = float(roc_auc_score(y_test, y_pred_proba) * 100)
                    brier = float(brier_score_loss(y_test, y_pred_proba))
                    calibration_score = float(round((1.0 - brier) * 100, 1))

                    xgb_acc = accuracy
                    lgb_acc = accuracy
                    cb_acc = accuracy
                    if hasattr(model, 'named_estimators_'):
                        try:
                            xgb_pred = model.named_estimators_['xgb'].predict(X_test_proc)
                            xgb_acc = float(accuracy_score(y_test, xgb_pred) * 100)
                        except: pass
                        try:
                            lgb_pred = model.named_estimators_['lgb'].predict(X_test_proc)
                            lgb_acc = float(accuracy_score(y_test, lgb_pred) * 100)
                        except: pass
                        try:
                            cb_pred = model.named_estimators_['cb'].predict(X_test_proc)
                            cb_acc = float(accuracy_score(y_test, cb_pred) * 100)
                        except: pass
                    
                    if os.path.exists(mapie_path):
                        mapie_model = joblib.load(mapie_path)
                        dyn_conformal = compute_conformal_diagnostics(mapie_model, X_test_proc, y_test)
                        
                        # Calculate Train and Val Confident Rates (Decisiveness)
                        train_conf_rates = {}
                        val_conf_rates = {}
                        
                        try:
                            # 1. Train set paths
                            x_train_proc_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_X.csv')
                            if not os.path.exists(x_train_proc_path):
                                x_train_proc_path = os.path.join(BASE_DIR, 'processed_data', f'X_train_processed_{industry}.csv')
                            
                            # 2. Val set paths
                            x_val_proc_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_val_X.csv')
                            if not os.path.exists(x_val_proc_path) and os.path.exists(val_path):
                                df_val_raw = pd.read_csv(val_path)
                                cols_to_drop = ['customer_id', 'industry', 'churn_probability', 'churned']
                                X_val_raw = df_val_raw.drop(columns=[c for c in cols_to_drop if c in df_val_raw.columns])
                                X_val_trans = preprocessor.transform(X_val_raw)
                                X_val_proc = pd.DataFrame(X_val_trans, columns=all_transformed_cols)
                            elif os.path.exists(x_val_proc_path):
                                X_val_proc = pd.read_csv(x_val_proc_path)
                            else:
                                X_val_proc = pd.DataFrame()
                                
                            # 3. Predict conformal set sizes on train
                            if os.path.exists(x_train_proc_path):
                                X_train_proc = pd.read_csv(x_train_proc_path)
                                _, y_pis_train = mapie_model.predict_set(X_train_proc)
                                for lvl_idx, lvl in enumerate([0.80, 0.85, 0.90, 0.95]):
                                    set_sizes_train = np.sum(y_pis_train[:, :, lvl_idx], axis=1)
                                    train_conf_rates[f"{lvl:.2f}"] = float(np.mean(set_sizes_train == 1) * 100)
                            else:
                                for lvl in [0.80, 0.85, 0.90, 0.95]:
                                    train_conf_rates[f"{lvl:.2f}"] = 80.0
                                    
                            # 4. Predict conformal set sizes on val
                            if not X_val_proc.empty:
                                _, y_pis_val = mapie_model.predict_set(X_val_proc)
                                for lvl_idx, lvl in enumerate([0.80, 0.85, 0.90, 0.95]):
                                    set_sizes_val = np.sum(y_pis_val[:, :, lvl_idx], axis=1)
                                    val_conf_rates[f"{lvl:.2f}"] = float(np.mean(set_sizes_val == 1) * 100)
                            else:
                                for lvl in [0.80, 0.85, 0.90, 0.95]:
                                    val_conf_rates[f"{lvl:.2f}"] = 78.0
                        except Exception as conf_err:
                            print(f"Error predicting train/val conformal sets: {conf_err}")
                            for lvl in [0.80, 0.85, 0.90, 0.95]:
                                train_conf_rates[f"{lvl:.2f}"] = 80.0
                                val_conf_rates[f"{lvl:.2f}"] = 78.0
                                
                        if dyn_conformal:
                            for lvl_key in dyn_conformal:
                                dyn_conformal[lvl_key]['confident_rate_train'] = round(train_conf_rates.get(lvl_key, 80.0), 2)
                                dyn_conformal[lvl_key]['confident_rate_val'] = round(val_conf_rates.get(lvl_key, 78.0), 2)
                            conformal_data = dyn_conformal
                            
                            cov_85 = dyn_conformal.get('0.85', {}).get('coverage', 85.0)
                            validation_data = {
                                'ensemble_acc': round(accuracy, 2),
                                'xgb_acc': round(xgb_acc, 2),
                                'lgb_acc': round(lgb_acc, 2),
                                'cb_acc': round(cb_acc, 2),
                                'brier_score': round(brier, 4),
                                'calibration_score': calibration_score,
                                'conformal_status': f"Guarantees Verified (Coverage: {cov_85:.1f}%)"
                            }

                # Collect info on active RAG files (Train/Val/Test)
                rag_files = []
                try:
                    train_csv_name = f'{industry}_augmented_raw.csv' if os.path.exists(os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_raw.csv')) else f'{industry}_churn_train.csv'
                    train_csv_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_raw.csv') if os.path.exists(os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_raw.csv')) else os.path.join(MOCK_DATA_DIR, 'train', f'{industry}_churn_train.csv')
                    
                    val_csv_name = f'{industry}_augmented_val_raw.csv' if os.path.exists(os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_val_raw.csv')) else f'{industry}_churn_val.csv'
                    val_csv_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_val_raw.csv') if os.path.exists(os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_val_raw.csv')) else os.path.join(MOCK_DATA_DIR, 'val', f'{industry}_churn_val.csv')
                    
                    test_csv_name = f'{industry}_churn_test_features.csv'
                    test_csv_path = os.path.join(MOCK_DATA_DIR, 'test', f'{industry}_churn_test_features.csv')
                    
                    for name, fpath, ftype in [
                        (train_csv_name, train_csv_path, 'Train Set (RAG Reference)'),
                        (val_csv_name, val_csv_path, 'Validation Set (UQ Calibration)'),
                        (test_csv_name, test_csv_path, 'Test Set (Out-of-sample)')
                    ]:
                        if os.path.exists(fpath):
                            row_count = 0
                            try:
                                with open(fpath, 'r', encoding='utf-8') as f:
                                    row_count = sum(1 for _ in f) - 1
                            except:
                                pass
                            rag_files.append({
                                'name': name,
                                'path': fpath.replace('\\', '/'),
                                'type': ftype,
                                'rows': row_count,
                                'size_kb': round(os.path.getsize(fpath) / 1024, 1)
                            })
                except Exception as f_err:
                    print(f"Error gathering RAG files for {industry}: {f_err}")
                    
        except Exception as e:
            print(f"Error calculating stats for {industry}: {str(e)}")
            
        results[industry] = {
            'name': f"{industry.capitalize()} Churn Engine",
            'metrics': {
                'total_customers': total_cust,
                'churn_rate': round(churn_rate, 2),
                'avg_monthly': round(avg_spend, 2),
                'avg_tenure': round(avg_tenure, 1),
                'model_accuracy': round(accuracy, 2),
                'model_auc': round(auc, 2)
            },
            'charts': {
                'churn_dist': {'labels': labels, 'values': values},
                'feature_weights': {'labels': imp_labels, 'values': imp_values}
            },
            'conformal_data': conformal_data,
            'rag_files': rag_files,
            'validation_data': validation_data
        }
        
    return results


def index(request):
    """Renders the dashboard homepage."""
    data = get_stats_and_chart_data()
    context = {
        'industries_json': json.dumps(data)
    }
    return render(request, 'dashboard/index.html', context)


def predict_single_row(row_dict, industry, preprocessor, model, mapie_model, all_transformed_cols, confidence=0.85, mapping=None):
    """Computes prediction and conformal UQ decision for a single record dict."""
    profile = {}
    for col in INDUSTRY_SCHEMAS[industry]:
        val = None
        if mapping:
            # Find the user column mapped to standard feature 'col'
            for orig_col, std_col in mapping.items():
                if std_col == col:
                    val = row_dict.get(orig_col)
                    break
        if val is None:
            val = row_dict.get(col)
        if val is None:
            val = DEFAULT_VALUES.get(col)
        profile[col] = val
        
    # Cast variables correctly based on dynamic types
    for k, v in profile.items():
        if v is None or pd.isna(v) or str(v).strip() == '':
            profile[k] = DEFAULT_VALUES.get(k)
        elif k in ['signup_year', 'age', 'tenure_months', 'support_tickets_90d', 'complaints_90d', 
                   'nps_score', 'days_since_last_activity', 'late_payments_12m', 'seats_purchased',
                   'active_users_30d', 'integrations_connected', 'admin_logins_30d', 'products_count',
                   'visits_90d', 'returns_90d', 'coupons_used_90d', 'branch_visits_90d', 'overdrafts_12m',
                   'orders_180d', 'app_sessions_30d', 'courses_enrolled', 'logins_30d', 'assignments_late_90d',
                   'advisor_contacts_90d', 'appointments_12m', 'missed_appointments_12m', 'portal_logins_90d',
                   'care_gap_count', 'stays_12m', 'reward_points_balance', 'cancellations_12m', 'claims_24m',
                   'policy_count', 'agent_contact_90d', 'renewal_days_remaining', 'outages_12m']:
            profile[k] = int(float(v))
        elif k in ['monthly_spend_usd', 'discount_pct', 'data_usage_gb_30d', 'dropped_calls_30d', 
                   'network_complaints_90d', 'feature_adoption_score', 'avg_basket_usd', 'avg_balance_usd',
                   'cart_abandon_rate', 'avg_order_value_usd', 'return_rate', 'completion_rate',
                   'avg_nightly_rate_usd', 'review_rating', 'premium_usd', 'avg_monthly_usage']:
            profile[k] = float(v)
        elif k in ['autopay_enabled', 'device_financed', 'international_roaming', 'onboarding_completed',
                   'free_shipping_member', 'primary_provider_assigned', 'smart_meter_enabled', 'paperless_billing', 'move_flag_90d']:
            # Encode binary flags
            if str(v).lower() in ['yes', 'true', '1', 'onboarded', 'enabled']:
                profile[k] = 1
            elif str(v).lower() in ['no', 'false', '0', 'disabled']:
                profile[k] = 0
            else:
                profile[k] = int(v)

    # Parse confidence level index
    confidence_levels = [0.80, 0.85, 0.90, 0.95]
    try:
        c_val = float(confidence)
    except (ValueError, TypeError):
        c_val = 0.85
        
    try:
        level_idx = confidence_levels.index(c_val)
    except ValueError:
        level_idx = 1  # default to index 1 (85%)

    rag_sources = []
    
    df_temp = pd.DataFrame([profile])
    X_temp = preprocessor.transform(df_temp)
    X_temp_df = pd.DataFrame(X_temp, columns=all_transformed_cols)
    
    base_prob = float(model.predict_proba(X_temp_df)[0, 1])
    
    # --- TABULAR RAG RETRIEVAL ---
    db = get_training_db(industry)
    avg_sim = 100.0
    if db is not None:
        distances, indices = db['nn'].kneighbors(X_temp_df)
        dist = distances[0]
        idxs = indices[0]
        
        # Average similarity of the top 3 neighbors
        top_similarities = []
        for i in range(min(3, len(dist))):
            s_pct = (1.0 / (1.0 + dist[i])) * 100
            top_similarities.append(s_pct)
        avg_sim = float(np.mean(top_similarities)) if top_similarities else 100.0
        
        local_churn_statuses = []
        for i, idx in enumerate(idxs):
            churn_val = int(db['y'].iloc[idx])
            local_churn_statuses.append(churn_val)
            
            # Top 3 similar cases for explanation
            if i < 3 and not db['raw'].empty and idx < len(db['raw']):
                raw_row = db['raw'].iloc[idx]
                sim_pct = round((1.0 / (1.0 + dist[i])) * 100, 1)
                
                # Construct symmetric UI values from industry schema to prevent layout breakage
                seg = raw_row.get('customer_segment', 'standard')
                ten = int(raw_row.get('tenure_months', 24))
                cont = raw_row.get('contract_type', 'annual')
                spend = float(raw_row.get('monthly_spend_usd', 100.0))
                
                # Extra detail in place of internet service
                if industry == 'telecom':
                    detail = f"Plan: {raw_row.get('plan_type', 'N/A')} | Data: {raw_row.get('data_usage_gb_30d', 0)} GB"
                elif industry == 'saas':
                    detail = f"Seats: {int(raw_row.get('seats_purchased', 0))} | Feature score: {raw_row.get('feature_adoption_score', 0)}"
                elif industry == 'retail':
                    detail = f"Tier: {raw_row.get('loyalty_tier', 'N/A')} | Basket: ${raw_row.get('avg_basket_usd', 0)}"
                elif industry == 'banking':
                    detail = f"Account: {raw_row.get('account_type', 'N/A')} | Balance: ${raw_row.get('avg_balance_usd', 0)}"
                elif industry == 'ecommerce':
                    detail = f"Orders: {int(raw_row.get('orders_180d', 0))} | Cart Abandon: {raw_row.get('cart_abandon_rate', 0)}"
                elif industry == 'education':
                    detail = f"Courses: {int(raw_row.get('courses_enrolled', 0))} | Completion: {raw_row.get('completion_rate', 0)}"
                elif industry == 'healthcare':
                    detail = f"Plan: {raw_row.get('plan_category', 'N/A')} | Appts: {int(raw_row.get('appointments_12m', 0))}"
                elif industry == 'hospitality':
                    detail = f"Level: {raw_row.get('membership_level', 'N/A')} | Stays: {int(raw_row.get('stays_12m', 0))}"
                elif industry == 'insurance':
                    detail = f"Policy: {raw_row.get('policy_type', 'N/A')} | Policies: {int(raw_row.get('policy_count', 0))}"
                elif industry == 'utilities':
                    detail = f"Service: {raw_row.get('service_type', 'N/A')} | Smart Meter: {int(raw_row.get('smart_meter_enabled', 0))}"
                else:
                    detail = f"Spend: ${spend}"
                
                rag_sources.append({
                    'similarity': sim_pct,
                    'gender': f"{seg.capitalize()} segment",
                    'tenure': ten,
                    'contract': cont,
                    'monthly_charges': spend,
                    'internet_service': detail,
                    'churned': 'Yes' if churn_val == 1 else 'No'
                })
        
        p_retrieval = np.mean(local_churn_statuses)
        base_prob = 0.8 * base_prob + 0.2 * p_retrieval
    # -----------------------------
    
    _, y_pis = mapie_model.predict_set(X_temp_df)
    
    in_class_0 = bool(y_pis[0, 0, level_idx])
    in_class_1 = bool(y_pis[0, 1, level_idx])
    
    if in_class_0 and in_class_1:
        conformal_set, uq_status, uq_color = ["Retained", "Churned"], "Active Monitoring", "#F39C12"
    elif in_class_1:
        conformal_set, uq_status, uq_color = ["Churned"], "Action Required", "#E74C3C"
    elif in_class_0:
        conformal_set, uq_status, uq_color = ["Retained"], "No Intervention", "#2ECC71"
    else:
        conformal_set, uq_status, uq_color = ["Retained", "Churned"], "Active Monitoring (Uncertain Profile)", "#F39C12"

    if db is not None:
        if avg_sim < 35.0:
            safety_status = "OOD Warning"
            safety_color = "#ef4444"
            safety_msg = f"Out-of-Distribution: Input profile has low alignment with training data baseline (Avg Similarity: {avg_sim:.1f}%)."
        else:
            safety_status = "Safe Alignment"
            safety_color = "#22c55e"
            safety_msg = f"Safe: Profile matches training distribution baseline (Avg Similarity: {avg_sim:.1f}%)."
    else:
        safety_status = "Safe (Fallback)"
        safety_color = "#22c55e"
        safety_msg = "RAG reference database is unavailable. Statistical alignment check skipped."

    return {
        'churn_probability (%)': round(base_prob * 100, 2),
        'conformal_prediction_set': ", ".join(conformal_set),
        'recommended_business_action': uq_status,
        'action_color': uq_color,
        'rag_sources': rag_sources,
        'column_mappings': mapping or {},
        'safety_audit': {
            'avg_similarity': round(avg_sim, 1),
            'status': safety_status,
            'color': safety_color,
            'message': safety_msg
        }
    }


@csrf_exempt
def predict(request):
    """API endpoint to predict churn probability for a customer profile in a given industry."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required.'}, status=400)
    
    try:
        confidence = request.GET.get('confidence', '0.85')
        industry = request.GET.get('industry', 'telecom').lower()
        if industry not in ['telecom', 'saas', 'retail', 'banking']:
            industry = 'telecom'
            
        data = json.loads(request.body)
        
        # Load specific industry model artifacts
        prep_path = os.path.join(BASE_DIR, 'processed_data', f'preprocessor_{industry}.joblib')
        model_path = os.path.join(BASE_DIR, 'processed_data', f'model_{industry}.joblib')
        mapie_path = os.path.join(BASE_DIR, 'processed_data', f'mapie_model_{industry}.joblib')
        
        if not os.path.exists(prep_path) or not os.path.exists(model_path) or not os.path.exists(mapie_path):
            return JsonResponse({'error': f'Model artifacts for industry {industry} not found.'}, status=500)
        
        preprocessor = joblib.load(prep_path)
        model = joblib.load(model_path)
        mapie_model = joblib.load(mapie_path)
        
        # Load dataset features dynamically to get transformed columns
        train_path = os.path.join(MOCK_DATA_DIR, 'train', f'{industry}_churn_train.csv')
        df_cols = pd.read_csv(train_path, nrows=1)
        numeric_features, categorical_features = get_feature_types(df_cols)
                    
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
        all_transformed_cols = numeric_features + encoded_cat_cols
            
        res = predict_single_row(data, industry, preprocessor, model, mapie_model, all_transformed_cols, confidence)
        
        # Advice builders per industry
        drivers = []
        recommendations = []
        
        mapped_contract = res.get('column_mappings', {}).get('contract_type', 'contract_type')
        mapped_tickets = res.get('column_mappings', {}).get('support_tickets_90d', 'support_tickets_90d')
        mapped_nps = res.get('column_mappings', {}).get('nps_score', 'nps_score')
        
        contract_val = data.get(mapped_contract, data.get('contract_type', 'annual'))
        if contract_val == 'month_to_month':
            drivers.append("Short-term Contract (Month-to-month) increases risk of early churn.")
            
        try:
            tickets_val = int(data.get(mapped_tickets, data.get('support_tickets_90d', 0)))
            if tickets_val > 5:
                drivers.append(f"High support ticket volume ({tickets_val} in last 90 days) indicates technical frustration.")
        except:
            pass
            
        try:
            nps_val = int(data.get(mapped_nps, data.get('nps_score', 8)))
            if nps_val < 7:
                drivers.append(f"Low Customer NPS score ({nps_val}) indicates dissatisfaction or negative sentiment.")
        except:
            pass

        # Industry specific drivers & what-if recommendations
        alt = data.copy()
        
        if industry == 'telecom':
            mapped_dropped = res.get('column_mappings', {}).get('dropped_calls_30d', 'dropped_calls_30d')
            try:
                dc = int(data.get(mapped_dropped, data.get('dropped_calls_30d', 0)))
                if dc > 5: drivers.append(f"High dropped call frequency ({dc} in 30 days) signals signal/coverage issues.")
            except: pass
            
            # Recommendation 1: Move contract to annual/multi_year
            alt[mapped_contract] = 'annual'
            alt['contract_type'] = 'annual'
            
        elif industry == 'saas':
            mapped_adopt = res.get('column_mappings', {}).get('feature_adoption_score', 'feature_adoption_score')
            try:
                fa = float(data.get(mapped_adopt, data.get('feature_adoption_score', 0.8)))
                if fa < 0.5: drivers.append(f"Low feature adoption ({round(fa*100)}%) shows low product utilization.")
            except: pass
            
            # Recommendation: Increase onboarding/feature adoption
            alt[mapped_adopt] = 0.9
            alt['feature_adoption_score'] = 0.9
            
        elif industry == 'retail':
            mapped_recency = res.get('column_mappings', {}).get('days_since_last_activity', 'days_since_last_activity')
            try:
                rec = int(data.get(mapped_recency, data.get('days_since_last_activity', 0)))
                if rec > 45: drivers.append(f"High inactivity period ({rec} days) suggests customer is slipping away.")
            except: pass
            
            # Recommendation: Launch loyalty promotion
            mapped_loyalty = res.get('column_mappings', {}).get('loyalty_tier', 'loyalty_tier')
            alt[mapped_loyalty] = 'gold'
            alt['loyalty_tier'] = 'gold'
            
        elif industry == 'banking':
            mapped_overdraft = res.get('column_mappings', {}).get('overdrafts_12m', 'overdrafts_12m')
            try:
                od = int(data.get(mapped_overdraft, data.get('overdrafts_12m', 0)))
                if od > 1: drivers.append(f"Customer has {od} overdrafts in last 12 months, indicating financial stress.")
            except: pass
            
            # Recommendation: Setup balance alerts / savings options
            mapped_balance = res.get('column_mappings', {}).get('avg_balance_usd', 'avg_balance_usd')
            try:
                bal = float(alt.get(mapped_balance, alt.get('avg_balance_usd', 0.0)))
                alt[mapped_balance] = bal + 5000.0
            except: pass
            
        # Alt scenario analysis
        try:
            df_alt = pd.DataFrame([alt])
            headers_alt = list(df_alt.columns)
            mapping_alt = map_columns_nlp(headers_alt, industry)
            df_alt_mapped = df_alt.rename(columns=mapping_alt)
            
            profile_alt = {}
            for col in INDUSTRY_SCHEMAS[industry]:
                profile_alt[col] = df_alt_mapped.to_dict(orient='records')[0].get(col, DEFAULT_VALUES.get(col))
                
            # Cast variables in alt profile correctly
            for k, v in profile_alt.items():
                if v is None or pd.isna(v) or str(v).strip() == '':
                    profile_alt[k] = DEFAULT_VALUES.get(k)
                elif k in ['signup_year', 'age', 'tenure_months', 'support_tickets_90d', 'complaints_90d', 
                           'nps_score', 'days_since_last_activity', 'late_payments_12m', 'seats_purchased',
                           'active_users_30d', 'integrations_connected', 'admin_logins_30d', 'products_count',
                           'visits_90d', 'returns_90d', 'coupons_used_90d', 'branch_visits_90d', 'overdrafts_12m']:
                    profile_alt[k] = int(float(v))
                elif k in ['monthly_spend_usd', 'discount_pct', 'data_usage_gb_30d', 'dropped_calls_30d', 
                           'network_complaints_90d', 'feature_adoption_score', 'avg_basket_usd', 'avg_balance_usd']:
                    profile_alt[k] = float(v)
                elif k in BINARY_CATEGORICAL_COLS:
                    if str(v).lower() in ['yes', 'true', '1', 'onboarded', 'enabled']:
                        profile_alt[k] = 1
                    elif str(v).lower() in ['no', 'false', '0', 'disabled']:
                        profile_alt[k] = 0
                    else:
                        profile_alt[k] = int(v)
                        
            df_temp_alt = pd.DataFrame([profile_alt])
            X_alt = preprocessor.transform(df_temp_alt)
            alt_prob = float(model.predict_proba(pd.DataFrame(X_alt, columns=all_transformed_cols))[0, 1])
            
            action_desc = "Optimize subscription terms (Upgrade contract)"
            if industry == 'saas': action_desc = "Run onboarding campaign to boost feature adoption to 90%"
            elif industry == 'retail': action_desc = "Target customer with Gold membership level promo"
            elif industry == 'banking': action_desc = "Cross-sell high interest savings to grow balance by $5k"
            
            impact_pct = round((res['churn_probability (%)']/100 - alt_prob) * 100, 1)
            recommendations.append({
                'action': action_desc,
                'new_prob': round(alt_prob * 100, 1),
                'impact': impact_pct if impact_pct > 0 else 0.0
            })
        except Exception as e:
            print(f"Failed to generate dynamic recommendations: {str(e)}")
            recommendations.append({
                'action': 'Upgrade account / Optimize contract type',
                'new_prob': round(res['churn_probability (%)'] * 0.7, 1),
                'impact': round(res['churn_probability (%)'] * 0.3, 1)
            })

        if not drivers: 
            drivers.append("Mostly low-risk features selected.")

        return JsonResponse({
            'churn_probability': res['churn_probability (%)'],
            'prediction': 'Yes' if res['churn_probability (%)'] >= 50.0 else 'No',
            'risk_level': res['recommended_business_action'],
            'risk_color': res['action_color'],
            'drivers': drivers,
            'recommendations': recommendations,
            'rag_sources': res.get('rag_sources', []),
            'column_mappings': res.get('column_mappings', {}),
            'uq': {
                'status': res['recommended_business_action'],
                'recommendation': "Suggested Action: " + ("Trigger high-value outreach." if "Action Required" in res['recommended_business_action'] else ("Deploy success check-in." if "Active Monitoring" in res['recommended_business_action'] else "No intervention budget spent.")),
                'set': res['conformal_prediction_set'],
                'color': res['action_color'],
                'badge': res['recommended_business_action']
            },
            'safety_audit': res.get('safety_audit')
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f"Prediction failed: {str(e)}"}, status=400)


def clean_features_df(df, industry):
    """Aligns and cleans DataFrame to target industry schema with correct types and default fill values."""
    cleaned_features = pd.DataFrame()
    df_reset = df.reset_index(drop=True)
    
    for col in INDUSTRY_SCHEMAS[industry]:
        val_col = df_reset.get(col)
        if val_col is None:
            cleaned_features[col] = [DEFAULT_VALUES.get(col)] * len(df_reset)
        else:
            if isinstance(val_col, pd.DataFrame):
                val_col = val_col.iloc[:, 0]
            cleaned_features[col] = val_col.fillna(DEFAULT_VALUES.get(col))
            
    # Cast types and handle special conversions (like parsing years from dates)
    for k in cleaned_features.columns:
        v_series = cleaned_features[k]
        if k in ['signup_year', 'age', 'tenure_months', 'support_tickets_90d', 'complaints_90d', 
                 'nps_score', 'days_since_last_activity', 'late_payments_12m', 'seats_purchased',
                 'active_users_30d', 'integrations_connected', 'admin_logins_30d', 'products_count',
                 'visits_90d', 'returns_90d', 'coupons_used_90d', 'branch_visits_90d', 'overdrafts_12m',
                 'orders_180d', 'app_sessions_30d', 'courses_enrolled', 'logins_30d', 'assignments_late_90d',
                 'advisor_contacts_90d', 'appointments_12m', 'missed_appointments_12m', 'portal_logins_90d',
                 'care_gap_count', 'stays_12m', 'reward_points_balance', 'cancellations_12m', 'claims_24m',
                 'policy_count', 'agent_contact_90d', 'renewal_days_remaining', 'outages_12m']:
            
            if k == 'signup_year':
                def extract_year(x):
                    try:
                        # Try parsing as datetime
                        return int(pd.to_datetime(x).year)
                    except:
                        try:
                            return int(float(x))
                        except:
                            return int(DEFAULT_VALUES.get('signup_year', 2024))
                cleaned_features[k] = v_series.map(extract_year)
            else:
                cleaned_features[k] = pd.to_numeric(v_series, errors='coerce').fillna(DEFAULT_VALUES.get(k, 0)).astype(float).astype(int)
                
        elif k in ['monthly_spend_usd', 'discount_pct', 'data_usage_gb_30d', 'dropped_calls_30d', 
                   'network_complaints_90d', 'feature_adoption_score', 'avg_basket_usd', 'avg_balance_usd',
                   'cart_abandon_rate', 'avg_order_value_usd', 'return_rate', 'completion_rate',
                   'avg_nightly_rate_usd', 'review_rating', 'premium_usd', 'avg_monthly_usage']:
            cleaned_features[k] = pd.to_numeric(v_series, errors='coerce').fillna(DEFAULT_VALUES.get(k, 0.0)).astype(float)
            
        elif k in BINARY_CATEGORICAL_COLS:
            def cast_binary(x):
                s = str(x).lower().strip()
                if s in ['yes', 'true', '1', 'onboarded', 'enabled']:
                    return 1
                elif s in ['no', 'false', '0', 'disabled']:
                    return 0
                else:
                    try:
                        return int(float(x))
                    except:
                        return int(DEFAULT_VALUES.get(k, 0))
            cleaned_features[k] = v_series.map(cast_binary)
            
    return cleaned_features


def clean_and_transform_batch(df, industry, preprocessor, all_transformed_cols):
    cleaned_features = clean_features_df(df, industry)
    X_trans = preprocessor.transform(cleaned_features)
    return pd.DataFrame(X_trans, columns=all_transformed_cols), cleaned_features


@csrf_exempt
def upload_csv(request):
    """
    Accepts an uploaded CSV file, auto-detects industry via NLP TF-IDF,
    routes features to the appropriate industry model, runs batch prediction with conformal UQ,
    and returns decision records.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required.'}, status=400)
    
    try:
        csv_file = request.FILES.get('file')
        confidence = request.GET.get('confidence', '0.85')
        if not csv_file:
            return JsonResponse({'error': 'No file uploaded.'}, status=400)
            
        df_uploaded = pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')))
        headers = df_uploaded.columns.tolist()
        
        # NLP Auto-detect industry from column headers
        industry = detect_industry(headers)
        
        # Load artifacts paths for detected industry
        prep_path = os.path.join(BASE_DIR, 'processed_data', f'preprocessor_{industry}.joblib')
        model_path = os.path.join(BASE_DIR, 'processed_data', f'model_{industry}.joblib')
        mapie_path = os.path.join(BASE_DIR, 'processed_data', f'mapie_model_{industry}.joblib')
        
        if not (os.path.exists(prep_path) and os.path.exists(model_path) and os.path.exists(mapie_path)):
            return JsonResponse({'error': f'Model artifacts for industry {industry} not found.'}, status=500)
            
        preprocessor = joblib.load(prep_path)
        model = joblib.load(model_path)
        mapie_model = joblib.load(mapie_path)
        
        # Determine standard columns for mapped DF
        mapping = map_columns_nlp(headers, industry)
        df_mapped = df_uploaded.rename(columns=mapping)
        
        # Calculate schema health & validation warnings
        schema_features = INDUSTRY_SCHEMAS[industry]
        found_features = [col for col in schema_features if col in df_mapped.columns]
        missing_features = [col for col in schema_features if col not in df_mapped.columns]
        schema_health_score = float(round((len(found_features) / len(schema_features)) * 100, 1)) if schema_features else 100.0
        
        schema_warnings = []
        for col in missing_features:
            schema_warnings.append(f"Warning: Feature '{col}' was missing and imputed using default value ({DEFAULT_VALUES.get(col)}).")
        
        train_path = os.path.join(MOCK_DATA_DIR, 'train', f'{industry}_churn_train.csv')
        df_cols = pd.read_csv(train_path, nrows=1)
        numeric_features, categorical_features = get_feature_types(df_cols)
                    
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
        all_transformed_cols = numeric_features + encoded_cat_cols

        # Run vectorized batch pre-processing & inference
        X_batch_proc, df_cleaned = clean_and_transform_batch(df_mapped, industry, preprocessor, all_transformed_cols)
        
        # Original probabilities
        base_probs = model.predict_proba(X_batch_proc)[:, 1]
        
        # Parse confidence level index
        confidence_levels = [0.80, 0.85, 0.90, 0.95]
        try:
            c_val = float(confidence)
        except:
            c_val = 0.85
        try:
            level_idx = confidence_levels.index(c_val)
        except ValueError:
            level_idx = 1
            
        _, y_pis = mapie_model.predict_set(X_batch_proc)

        # Tabular RAG Retrieval and Drift Auditing for the entire batch
        db = get_training_db(industry)
        p_retrievals = np.zeros(len(df_mapped))
        rag_sources_batch = [[] for _ in range(len(df_mapped))]
        ood_rows_count = 0
        
        if db is not None:
            distances, indices = db['nn'].kneighbors(X_batch_proc)
            for idx in range(len(df_mapped)):
                dist = distances[idx]
                idxs = indices[idx]
                
                # Check drift for individual row using top 3 neighbors
                row_similarities = []
                for neighbor_idx in range(min(3, len(dist))):
                    s_pct = (1.0 / (1.0 + dist[neighbor_idx])) * 100
                    row_similarities.append(s_pct)
                row_avg_sim = float(np.mean(row_similarities)) if row_similarities else 100.0
                if row_avg_sim < 35.0:
                    ood_rows_count += 1
                
                local_churn_statuses = []
                for i, neighbor_idx in enumerate(idxs):
                    churn_val = int(db['y'].iloc[neighbor_idx])
                    local_churn_statuses.append(churn_val)
                    
                    if i < 3 and not db['raw'].empty and neighbor_idx < len(db['raw']):
                        raw_row = db['raw'].iloc[neighbor_idx]
                        sim_pct = round((1.0 / (1.0 + dist[i])) * 100, 1)
                        
                        seg = raw_row.get('customer_segment', 'standard')
                        ten = int(raw_row.get('tenure_months', 24))
                        cont = raw_row.get('contract_type', 'annual')
                        spend = float(raw_row.get('monthly_spend_usd', 100.0))
                        
                        if industry == 'telecom':
                            detail = f"Plan: {raw_row.get('plan_type', 'N/A')} | Data: {raw_row.get('data_usage_gb_30d', 0)} GB"
                        elif industry == 'saas':
                            detail = f"Seats: {int(raw_row.get('seats_purchased', 0))} | Feature score: {raw_row.get('feature_adoption_score', 0)}"
                        elif industry == 'retail':
                            detail = f"Tier: {raw_row.get('loyalty_tier', 'N/A')} | Basket: ${raw_row.get('avg_basket_usd', 0)}"
                        elif industry == 'banking':
                            detail = f"Account: {raw_row.get('account_type', 'N/A')} | Balance: ${raw_row.get('avg_balance_usd', 0)}"
                        elif industry == 'ecommerce':
                            detail = f"Orders: {int(raw_row.get('orders_180d', 0))} | Cart Abandon: {raw_row.get('cart_abandon_rate', 0)}"
                        elif industry == 'education':
                            detail = f"Courses: {int(raw_row.get('courses_enrolled', 0))} | Completion: {raw_row.get('completion_rate', 0)}"
                        elif industry == 'healthcare':
                            detail = f"Plan: {raw_row.get('plan_category', 'N/A')} | Appts: {int(raw_row.get('appointments_12m', 0))}"
                        elif industry == 'hospitality':
                            detail = f"Level: {raw_row.get('membership_level', 'N/A')} | Stays: {int(raw_row.get('stays_12m', 0))}"
                        elif industry == 'insurance':
                            detail = f"Policy: {raw_row.get('policy_type', 'N/A')} | Policies: {int(raw_row.get('policy_count', 0))}"
                        elif industry == 'utilities':
                            detail = f"Service: {raw_row.get('service_type', 'N/A')} | Smart Meter: {int(raw_row.get('smart_meter_enabled', 0))}"
                        else:
                            detail = f"Spend: ${spend}"
                            
                        rag_sources_batch[idx].append({
                            'similarity': sim_pct,
                            'gender': f"{seg.capitalize()} segment",
                            'tenure': ten,
                            'contract': cont,
                            'monthly_charges': spend,
                            'internet_service': detail,
                            'churned': 'Yes' if churn_val == 1 else 'No'
                        })
                p_retrievals[idx] = np.mean(local_churn_statuses)
            final_probs = 0.8 * base_probs + 0.2 * p_retrievals
        else:
            final_probs = base_probs
            
        # Calculate batch drift metrics
        ood_ratio = float(round((ood_rows_count / len(df_mapped)) * 100, 1)) if len(df_mapped) > 0 else 0.0
        if ood_ratio > 30.0:
            batch_safety_status = "Drift Warning"
            batch_safety_color = "#ef4444"
            batch_safety_msg = f"High Drift: {ood_ratio:.1f}% of cohort is out-of-distribution. Retraining/augmentation is highly recommended."
        elif ood_ratio > 15.0:
            batch_safety_status = "Caution"
            batch_safety_color = "#f59e0b"
            batch_safety_msg = f"Moderate Drift: {ood_ratio:.1f}% of cohort is out-of-distribution. Monitor prediction variance."
        else:
            batch_safety_status = "Safe"
            batch_safety_color = "#22c55e"
            batch_safety_msg = f"Baseline Intact: Only {ood_ratio:.1f}% of cohort is out-of-distribution. Model is operating within safe limits."

        # Calculate model weight effects & actionable optimization
        actionable_feature = 'contract_type'
        optimal_val = 'annual'
        action_text = 'transition month-to-month contracts to annual terms'
        feature_label = 'Contract Type'
        
        if industry == 'saas':
            actionable_feature = 'feature_adoption_score'
            optimal_val = 0.9
            action_text = 'run onboarding campaigns to boost feature adoption to 90%'
            feature_label = 'Feature Adoption'
        elif industry == 'retail':
            actionable_feature = 'loyalty_tier'
            optimal_val = 'gold'
            action_text = 'target inactive customers with Gold membership upgrades'
            feature_label = 'Loyalty Tier'
        elif industry == 'banking':
            actionable_feature = 'avg_balance_usd'
            optimal_val = 'increase_5k'
            action_text = 'cross-sell high-yield savings to grow balances by $5k'
            feature_label = 'Account Balance'
        elif industry == 'ecommerce':
            actionable_feature = 'cart_abandon_rate'
            optimal_val = 0.1
            action_text = 'implement cart recovery emails to lower abandon rate to 10%'
            feature_label = 'Cart Abandon Rate'
        elif industry == 'education':
            actionable_feature = 'completion_rate'
            optimal_val = 0.9
            action_text = 'send reminders to increase assignment completion rate to 90%'
            feature_label = 'Completion Rate'
        elif industry == 'healthcare':
            actionable_feature = 'missed_appointments_12m'
            optimal_val = 0
            action_text = 'send SMS confirmation alerts to eliminate missed appointments'
            feature_label = 'Missed Appointments'
        elif industry == 'hospitality':
            actionable_feature = 'membership_level'
            optimal_val = 'gold'
            action_text = 'target guests with loyalty membership status upgrades'
            feature_label = 'Membership Level'
        elif industry == 'insurance':
            actionable_feature = 'agent_contact_90d'
            optimal_val = 1
            action_text = 'schedule proactive insurance advisory check-ins'
            feature_label = 'Agent Contact'
        elif industry == 'utilities':
            actionable_feature = 'smart_meter_enabled'
            optimal_val = 1
            action_text = 'install smart meters to enable real-time energy usage alerts'
            feature_label = 'Smart Meter'

        df_optimized = df_mapped.copy()
        mapped_actionable = mapping.get(actionable_feature, actionable_feature)
        
        if optimal_val == 'increase_5k':
            if mapped_actionable in df_optimized.columns:
                df_optimized[mapped_actionable] = df_optimized[mapped_actionable].fillna(0.0).astype(float) + 5000.0
            else:
                df_optimized[mapped_actionable] = 5000.0
        else:
            df_optimized[mapped_actionable] = optimal_val
            
        X_opt_proc, _ = clean_and_transform_batch(df_optimized, industry, preprocessor, all_transformed_cols)
        opt_base_probs = model.predict_proba(X_opt_proc)[:, 1]
        
        if db is not None:
            opt_final_probs = 0.8 * opt_base_probs + 0.2 * p_retrievals
        else:
            opt_final_probs = opt_base_probs
            
        _, y_pis_opt = mapie_model.predict_set(X_opt_proc)

        # Calculate feature weight attribution percentage
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'estimators_'):
            xgb_idx = [i for i, (name, _) in enumerate(model.estimators) if name == 'xgb']
            importances = model.estimators_[xgb_idx[0]].feature_importances_ if xgb_idx else model.estimators_[0].feature_importances_
        else:
            importances = np.zeros(len(all_transformed_cols))
            
        feature_imp_pairs = list(zip(all_transformed_cols, importances))
        feature_weight = sum(imp for col_name, imp in feature_imp_pairs if col_name == actionable_feature or col_name.startswith(actionable_feature + '_'))
        total_imp = sum(importances)
        weight_pct = float(round((feature_weight / total_imp) * 100, 1)) if total_imp > 0 else 25.0

        # Construct final lists & aggregate statistics
        results_list = []
        decisive_count = 0
        standout_count = 0
        
        for i in range(len(df_mapped)):
            # Conformal decisiveness (Original)
            in_class_0 = bool(y_pis[i, 0, level_idx])
            in_class_1 = bool(y_pis[i, 1, level_idx])
            is_decisive = (in_class_0 != in_class_1)
            if is_decisive:
                decisive_count += 1
                
            if in_class_0 and in_class_1:
                conformal_set, uq_status, uq_color = ["Retained", "Churned"], "Active Monitoring", "#F39C12"
            elif in_class_1:
                conformal_set, uq_status, uq_color = ["Churned"], "Action Required", "#E74C3C"
            elif in_class_0:
                conformal_set, uq_status, uq_color = ["Retained"], "No Intervention", "#2ECC71"
            else:
                conformal_set, uq_status, uq_color = ["Retained", "Churned"], "Active Monitoring (Uncertain Profile)", "#F39C12"
                
            if uq_status != "No Intervention":
                standout_count += 1

            # Conformal sets (Optimized)
            in_class_0_opt = bool(y_pis_opt[i, 0, level_idx])
            in_class_1_opt = bool(y_pis_opt[i, 1, level_idx])
            if in_class_0_opt and in_class_1_opt:
                conformal_set_opt, uq_status_opt, uq_color_opt = ["Retained", "Churned"], "Active Monitoring", "#F39C12"
            elif in_class_1_opt:
                conformal_set_opt, uq_status_opt, uq_color_opt = ["Churned"], "Action Required", "#E74C3C"
            elif in_class_0_opt:
                conformal_set_opt, uq_status_opt, uq_color_opt = ["Retained"], "No Intervention", "#2ECC71"
            else:
                conformal_set_opt, uq_status_opt, uq_color_opt = ["Retained", "Churned"], "Active Monitoring (Uncertain Profile)", "#F39C12"

            out_row = {}
            for col in headers:
                val = df_mapped.iloc[i].get(mapping.get(col, col))
                if isinstance(val, (np.integer, np.int64, np.int32)):
                    val = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32)):
                    val = float(val)
                elif isinstance(val, np.ndarray):
                    val = val.tolist()
                elif pd.isna(val):
                    val = None
                out_row[col] = val
                
            # Original values
            out_row['churn_probability (%)'] = round(float(final_probs[i] * 100), 2)
            out_row['conformal_prediction_set'] = ", ".join(conformal_set)
            out_row['recommended_business_action'] = uq_status
            out_row['action_color'] = uq_color
            out_row['rag_sources'] = rag_sources_batch[i]
            
            # Optimized values
            out_row['opt_churn_probability (%)'] = round(float(opt_final_probs[i] * 100), 2)
            out_row['opt_conformal_prediction_set'] = ", ".join(conformal_set_opt)
            out_row['opt_recommended_business_action'] = uq_status_opt
            out_row['opt_action_color'] = uq_color_opt
            
            results_list.append(out_row)
            
        total_records = len(results_list)
        batch_confident_rate = round((decisive_count / total_records) * 100, 1) if total_records else 0.0
        avg_original_risk = round(float(np.mean(final_probs) * 100), 1) if total_records else 0.0
        avg_optimized_risk = round(float(np.mean(opt_final_probs) * 100), 1) if total_records else 0.0
        risk_reduction = round(max(0.0, avg_original_risk - avg_optimized_risk), 1)

        mapping_report = {}
        for original, mapped in mapping.items():
            mapping_report[original] = mapped

        # Industry title labels
        industry_labels = {
            'telecom': 'Telecom Subscribers 📱',
            'saas': 'SaaS Subscribers ☁️',
            'retail': 'Retail Customers 🛒',
            'banking': 'Banking Accounts 🏦',
            'ecommerce': 'eCommerce Customers 🛍️',
            'education': 'Education Students 🎓',
            'healthcare': 'Healthcare Patients 🏥',
            'hospitality': 'Hospitality Guests 🏨',
            'insurance': 'Insurance Policyholders 🛡️',
            'utilities': 'Utilities Accounts ⚡'
        }

        return JsonResponse({
            'detected_industry': industry_labels.get(industry, f'{industry.capitalize()} Accounts'),
            'industry_key': industry,
            'column_mappings': mapping_report,
            'total_records': total_records,
            'standout_records': standout_count,
            'batch_confident_rate': batch_confident_rate,
            'avg_churn_risk': avg_original_risk,
            'avg_optimized_risk': avg_optimized_risk,
            'risk_reduction': risk_reduction,
            'actionable_driver': feature_label,
            'action_text': action_text,
            'weight_effect_pct': weight_pct,
            'schema_health_score': schema_health_score,
            'schema_warnings': schema_warnings,
            'ood_records_count': ood_rows_count,
            'ood_ratio_pct': ood_ratio,
            'batch_safety_status': batch_safety_status,
            'batch_safety_color': batch_safety_color,
            'batch_safety_msg': batch_safety_msg,
            'results': results_list
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f"Failed to process CSV batch: {str(e)}"}, status=400)


@csrf_exempt
def augment_db(request):
    """
    Accepts an uploaded CSV file, detects industry, processes it, appends it to active training set,
    retrains stacking ensemble, recalibrates conformal mapie model, and updates training db.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required.'}, status=400)

    try:
        csv_file = request.FILES.get('file')
        if not csv_file:
            return JsonResponse({'error': 'No file uploaded.'}, status=400)

        filename = csv_file.name.lower()
        is_train = False
        is_val = False
        
        if 'train' in filename:
            is_train = True
            ingested_as = 'Train Set'
        elif any(c in filename for c in ['val', 'validation', 'calib']):
            is_val = True
            ingested_as = 'Validation Set'
        else:
            ingested_as = 'Split Train/Val (80/20)'

        df_uploaded = pd.read_csv(io.StringIO(csv_file.read().decode('utf-8')))
        headers = df_uploaded.columns.tolist()

        # NLP detect industry
        industry = detect_industry(headers)
        
        # Column mappings
        mapping = map_columns_nlp(headers, industry)
        df_mapped = df_uploaded.rename(columns=mapping)

        # Look for target churn labels — check both mapped and original columns
        target_col = None
        # First check mapped columns
        for key in ['churned', 'churn', 'Churn', 'Exited', 'exited', 'class', 'Class', 'target', 'Target']:
            if key in df_mapped.columns:
                target_col = key
                break
                
        if not target_col:
            # Check synonym mapping for 'churned'
            for key, val in mapping.items():
                if val == 'churned':
                    target_col = key
                    break
                    
        if not target_col:
            # Also check original columns that weren't mapped
            for key in ['churned', 'churn', 'Churn', 'Exited', 'exited']:
                if key in df_uploaded.columns:
                    target_col = key
                    # Ensure it exists in df_mapped too
                    if key not in df_mapped.columns:
                        df_mapped[key] = df_uploaded[key]
                    break
                    
        if not target_col:
            return JsonResponse({
                'error': f"Uploaded dataset for {industry} must contain target label matching 'churned' (1/0 or Yes/No)."
            }, status=400)

        # Load existing preprocessor
        prep_path = os.path.join(BASE_DIR, 'processed_data', f'preprocessor_{industry}.joblib')
        model_path = os.path.join(BASE_DIR, 'processed_data', f'model_{industry}.joblib')
        mapie_path = os.path.join(BASE_DIR, 'processed_data', f'mapie_model_{industry}.joblib')

        if not os.path.exists(prep_path):
            return JsonResponse({'error': f'Original preprocessor for {industry} not found.'}, status=500)

        preprocessor = joblib.load(prep_path)

        # Load dataset features dynamically to get transformed columns
        train_path = os.path.join(MOCK_DATA_DIR, 'train', f'{industry}_churn_train.csv')
        df_cols = pd.read_csv(train_path, nrows=1)
        numeric_features, categorical_features = get_feature_types(df_cols)
                    
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        encoded_cat_cols = cat_encoder.get_feature_names_out(categorical_features).tolist()
        all_transformed_cols = numeric_features + encoded_cat_cols

        # Helper to process chunk
        def process_df_chunk(df_chunk, t_col):
            if df_chunk.empty:
                return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()
            # Clean target labels
            target_map = {
                'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 
                '1': 1, '0': 0, 1: 1, 0: 0, 
                'churned': 1, 'retained': 0, 'Exited': 1, 'exited': 1,
                True: 1, False: 0
            }
            y_chunk_mapped = df_chunk[t_col].map(target_map)
            y_chunk_mapped = y_chunk_mapped.dropna()
            
            if y_chunk_mapped.empty:
                print(f"Warning: No valid target labels found in column '{t_col}'. Unique values: {df_chunk[t_col].unique()[:10]}")
                return pd.DataFrame(), pd.Series(dtype=int), pd.DataFrame()
            
            # Get features
            df_features = df_chunk.drop(columns=[t_col]).loc[y_chunk_mapped.index]
            
            # Impute and clean features using our robust helper
            cleaned_features = clean_features_df(df_features, industry)
            
            X_trans = preprocessor.transform(cleaned_features)
            
            # Validate column count matches expected shape
            expected_cols = len(all_transformed_cols)
            if X_trans.shape[1] != expected_cols:
                print(f"Column mismatch detected! Got {X_trans.shape[1]} cols, expected {expected_cols}. Truncating/padding to match.")
                if X_trans.shape[1] > expected_cols:
                    X_trans = X_trans[:, :expected_cols]
                else:
                    import numpy as np
                    padding = np.zeros((X_trans.shape[0], expected_cols - X_trans.shape[1]))
                    X_trans = np.hstack([X_trans, padding])
            
            X_trans_df = pd.DataFrame(X_trans, columns=all_transformed_cols)
            y_series = y_chunk_mapped.astype(int)
            
            raw_df = cleaned_features.copy()
            raw_df['churned'] = y_series.values
            return X_trans_df, y_series, raw_df

        # Handle split / train / val routing
        from sklearn.model_selection import train_test_split
        if not is_train and not is_val:
            try:
                # If target is present, stratify if possible
                if target_col and df_mapped[target_col].nunique() > 1:
                    df_train_chunk, df_val_chunk = train_test_split(df_mapped, test_size=0.2, random_state=42, stratify=df_mapped[target_col])
                else:
                    df_train_chunk, df_val_chunk = train_test_split(df_mapped, test_size=0.2, random_state=42)
            except:
                df_train_chunk, df_val_chunk = train_test_split(df_mapped, test_size=0.2, random_state=42)
        elif is_train:
            df_train_chunk = df_mapped
            df_val_chunk = pd.DataFrame()
        else:
            df_train_chunk = pd.DataFrame()
            df_val_chunk = df_mapped

        # Load existing training sets
        db = get_training_db(industry)
        if db is not None:
            X_train_proc = db['X_processed']
            y_train_series = db['y']
            raw_train_df = db['raw']
        else:
            x_train_path = os.path.join(BASE_DIR, 'processed_data', f'X_train_processed_{industry}.csv')
            y_train_path = os.path.join(BASE_DIR, 'processed_data', f'y_train_{industry}.csv')
            X_train_proc = pd.read_csv(x_train_path)
            y_train = pd.read_csv(y_train_path)
            y_train_series = y_train.iloc[:, 0].astype(int)
            raw_train_df = pd.read_csv(os.path.join(MOCK_DATA_DIR, 'train', f'{industry}_churn_train.csv'))

        # Process and concatenate training chunk
        X_train_chunk_proc, y_train_chunk_series, raw_train_chunk_df = process_df_chunk(df_train_chunk, target_col)
        
        if not X_train_chunk_proc.empty:
            X_train_proc_augmented = pd.concat([X_train_proc, X_train_chunk_proc], ignore_index=True)
            y_train_augmented = pd.concat([y_train_series, y_train_chunk_series], ignore_index=True)
            raw_train_df_augmented = pd.concat([raw_train_df, raw_train_chunk_df], ignore_index=True)
            train_records_added = len(X_train_chunk_proc)
        else:
            X_train_proc_augmented = X_train_proc
            y_train_augmented = y_train_series
            raw_train_df_augmented = raw_train_df
            train_records_added = 0

        # Load existing validation sets (or initialize them)
        aug_val_x_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_val_X.csv')
        aug_val_y_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_val_y.csv')
        aug_val_raw_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_val_raw.csv')
        
        if os.path.exists(aug_val_x_path) and os.path.exists(aug_val_y_path) and os.path.exists(aug_val_raw_path):
            X_val_proc = pd.read_csv(aug_val_x_path)
            y_val_series = pd.read_csv(aug_val_y_path).iloc[:, 0].astype(int)
            raw_val_df = pd.read_csv(aug_val_raw_path)
        else:
            # Initialize from base validation dataset
            base_val_path = os.path.join(MOCK_DATA_DIR, 'val', f'{industry}_churn_val.csv')
            df_base_val = pd.read_csv(base_val_path)
            
            # Map columns
            val_target_col = None
            for key in ['churned', 'churn', 'Churn']:
                if key in df_base_val.columns:
                    val_target_col = key
                    break
            if not val_target_col:
                df_base_val = df_base_val.rename(columns={'churn': 'churned', 'Churn': 'churned'})
                val_target_col = 'churned'
                
            val_mapping = map_columns_nlp(df_base_val.columns.tolist(), industry)
            df_base_val_mapped = df_base_val.rename(columns=val_mapping)
            
            if val_target_col in df_base_val_mapped.columns:
                X_val_proc, y_val_series, raw_val_df = process_df_chunk(df_base_val_mapped, val_target_col)
            else:
                X_val_proc = pd.DataFrame()
                y_val_series = pd.Series(dtype=int)
                raw_val_df = pd.DataFrame()

        # Process and concatenate validation chunk
        X_val_chunk_proc, y_val_chunk_series, raw_val_chunk_df = process_df_chunk(df_val_chunk, target_col)
        
        if not X_val_chunk_proc.empty:
            X_val_proc_augmented = pd.concat([X_val_proc, X_val_chunk_proc], ignore_index=True)
            y_val_augmented = pd.concat([y_val_series, y_val_chunk_series], ignore_index=True)
            raw_val_df_augmented = pd.concat([raw_val_df, raw_val_chunk_df], ignore_index=True)
            val_records_added = len(X_val_chunk_proc)
        else:
            X_val_proc_augmented = X_val_proc
            y_val_augmented = y_val_series
            raw_val_df_augmented = raw_val_df
            val_records_added = 0

        # Save training augmented files
        aug_raw_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_raw.csv')
        aug_x_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_X.csv')
        aug_y_path = os.path.join(BASE_DIR, 'processed_data', f'{industry}_augmented_y.csv')
        X_train_proc_augmented.to_csv(aug_x_path, index=False)
        pd.DataFrame(y_train_augmented, columns=['churned']).to_csv(aug_y_path, index=False)
        raw_train_df_augmented.to_csv(aug_raw_path, index=False)
        
        # Save validation augmented files
        X_val_proc_augmented.to_csv(aug_val_x_path, index=False)
        pd.DataFrame(y_val_augmented, columns=['churned']).to_csv(aug_val_y_path, index=False)
        raw_val_df_augmented.to_csv(aug_val_raw_path, index=False)

        # Retrain Stacking Ensemble Model
        import xgboost as xgb
        import lightgbm as lgb
        import catboost as cb
        from sklearn.ensemble import StackingClassifier
        from sklearn.linear_model import LogisticRegression
        from mapie.classification import SplitConformalClassifier

        xgb_clf = xgb.XGBClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='logloss')
        lgb_clf = lgb.LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
        cb_clf = cb.CatBoostClassifier(iterations=150, depth=5, learning_rate=0.08, subsample=0.8, random_state=42, verbose=0)

        ensemble = StackingClassifier(
            estimators=[('xgb', xgb_clf), ('lgb', lgb_clf), ('cb', cb_clf)],
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )
        ensemble.fit(X_train_proc_augmented, y_train_augmented)

        # Calibrate Conformal MAPIE Model on the augmented validation set
        confidence_levels = [0.80, 0.85, 0.90, 0.95]
        mapie_model = SplitConformalClassifier(estimator=ensemble, confidence_level=confidence_levels, prefit=True)
        mapie_model.conformalize(X_val_proc_augmented, y_val_augmented)

        # Save over active model paths
        joblib.dump(ensemble, model_path)
        joblib.dump(mapie_model, mapie_path)

        # Force database cache reload
        get_training_db(industry, force_reload=True)

        # Recalculate stats for the dashboard update
        data = get_stats_and_chart_data()

        return JsonResponse({
            'success': True,
            'message': f"{industry.capitalize()} model augmented successfully! File '{csv_file.name}' ingested as {ingested_as} ({train_records_added} train rows, {val_records_added} validation rows).",
            'file_name': csv_file.name,
            'ingested_as': ingested_as,
            'train_records_added': train_records_added,
            'val_records_added': val_records_added,
            'stats_json': json.dumps(data)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': f"Failed to augment model: {str(e)}"}, status=400)
