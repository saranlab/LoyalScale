"""
LoyalScale Fuzzy Column Mapping Engine
======================================
Replaces character TF-IDF "NLP" with transparent, deterministic Fuzzy String Matching
(Levenshtein Distance & Token Sort Ratio via RapidFuzz) for schema alignment.
"""

import re
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    import difflib
    HAS_RAPIDFUZZ = False

# Standard industry schemas
INDUSTRY_SCHEMAS = {
    'telecom': [
        'signup_year', 'region', 'customer_segment', 'age', 'tenure_months', 'contract_type', 
        'monthly_spend_usd', 'discount_pct', 'autopay_enabled', 'support_tickets_90d', 
        'complaints_90d', 'nps_score', 'days_since_last_activity', 'late_payments_12m', 
        'acquisition_channel', 'plan_type', 'data_usage_gb_30d', 'dropped_calls_30d', 
        'network_complaints_90d', 'device_financed', 'international_roaming'
    ],
    'saas': [
        'signup_year', 'region', 'customer_segment', 'age', 'tenure_months', 'contract_type', 
        'monthly_spend_usd', 'discount_pct', 'autopay_enabled', 'support_tickets_90d', 
        'complaints_90d', 'nps_score', 'days_since_last_activity', 'late_payments_12m', 
        'acquisition_channel', 'seats_purchased', 'active_users_30d', 'feature_adoption_score', 
        'integrations_connected', 'admin_logins_30d', 'onboarding_completed'
    ],
    'retail': [
        'signup_year', 'region', 'customer_segment', 'age', 'tenure_months', 'contract_type', 
        'monthly_spend_usd', 'discount_pct', 'autopay_enabled', 'support_tickets_90d', 
        'complaints_90d', 'nps_score', 'days_since_last_activity', 'late_payments_12m', 
        'acquisition_channel', 'loyalty_tier', 'visits_90d', 'avg_basket_usd', 'returns_90d', 
        'coupons_used_90d', 'store_preference'
    ],
    'banking': [
        'signup_year', 'region', 'customer_segment', 'age', 'tenure_months', 'contract_type', 
        'monthly_spend_usd', 'discount_pct', 'autopay_enabled', 'support_tickets_90d', 
        'complaints_90d', 'nps_score', 'days_since_last_activity', 'late_payments_12m', 
        'acquisition_channel', 'account_type', 'avg_balance_usd', 'products_count', 
        'mobile_logins_30d', 'overdrafts_12m', 'branch_visits_90d'
    ],
    'ecommerce': [
        'signup_year', 'region', 'customer_segment', 'age', 'tenure_months', 'contract_type', 
        'monthly_spend_usd', 'discount_pct', 'autopay_enabled', 'support_tickets_90d', 
        'complaints_90d', 'nps_score', 'days_since_last_activity', 'late_payments_12m', 
        'acquisition_channel', 'orders_180d', 'cart_abandon_rate', 'avg_order_value_usd', 
        'return_rate', 'app_sessions_30d', 'free_shipping_member'
    ],
    'education': [
        'signup_year', 'region', 'customer_segment', 'age', 'tenure_months', 'contract_type', 
        'monthly_spend_usd', 'discount_pct', 'autopay_enabled', 'support_tickets_90d', 
        'complaints_90d', 'nps_score', 'days_since_last_activity', 'late_payments_12m', 
        'acquisition_channel', 'program_type', 'courses_enrolled', 'completion_rate', 
        'logins_30d', 'assignments_late_90d', 'advisor_contacts_90d'
    ],
    'healthcare': [
        'signup_year', 'region', 'customer_segment', 'age', 'tenure_months', 'contract_type', 
        'monthly_spend_usd', 'discount_pct', 'autopay_enabled', 'support_tickets_90d', 
        'complaints_90d', 'nps_score', 'days_since_last_activity', 'late_payments_12m', 
        'acquisition_channel', 'plan_category', 'appointments_12m', 'missed_appointments_12m', 
        'portal_logins_90d', 'care_gap_count', 'primary_provider_assigned'
    ],
    'hospitality': [
        'signup_year', 'region', 'customer_segment', 'age', 'tenure_months', 'contract_type', 
        'monthly_spend_usd', 'discount_pct', 'autopay_enabled', 'support_tickets_90d', 
        'complaints_90d', 'nps_score', 'days_since_last_activity', 'late_payments_12m', 
        'acquisition_channel', 'membership_level', 'stays_12m', 'avg_nightly_rate_usd', 
        'review_rating', 'reward_points_balance', 'cancellations_12m'
    ],
    'insurance': [
        'signup_year', 'region', 'customer_segment', 'age', 'tenure_months', 'contract_type', 
        'monthly_spend_usd', 'discount_pct', 'autopay_enabled', 'support_tickets_90d', 
        'complaints_90d', 'nps_score', 'days_since_last_activity', 'late_payments_12m', 
        'acquisition_channel', 'policy_type', 'premium_usd', 'claims_24m', 'policy_count', 
        'agent_contact_90d', 'renewal_days_remaining'
    ],
    'utilities': [
        'signup_year', 'region', 'customer_segment', 'age', 'tenure_months', 'contract_type', 
        'monthly_spend_usd', 'discount_pct', 'autopay_enabled', 'support_tickets_90d', 
        'complaints_90d', 'nps_score', 'days_since_last_activity', 'late_payments_12m', 
        'acquisition_channel', 'service_type', 'avg_monthly_usage', 'outages_12m', 
        'smart_meter_enabled', 'paperless_billing', 'move_flag_90d'
    ]
}

# Domain-specific synonym dictionary
SYNONYMS = {
    'churned': ['churn', 'churned', 'exited', 'is_churned', 'attrition', 'attrition_flag', 'churn_label', 'churn_flag', 'target', 'label'],
    'customer_id': ['customer_id', 'customerid', 'user_id', 'userid', 'id', 'cust_id', 'account_id', 'member_id', 'rownumber', 'surname'],
    'signup_year': ['signup_year', 'signup', 'registered_year', 'join_year', 'year_joined', 'registration_year', 'signup_date', 'joining_date'],
    'region': ['region', 'location', 'state', 'country', 'city', 'zone', 'area', 'geography', 'citytier', 'city_tier'],
    'customer_segment': ['customer_segment', 'segment', 'tier', 'customer_tier', 'user_segment', 'gender', 'maritalstatus', 'marital_status'],
    'age': ['age', 'customer_age', 'dob', 'years'],
    'tenure_months': ['tenure_months', 'tenure', 'months', 'months_active', 'duration_months', 'months_with_company', 'time_as_customer'],
    'contract_type': ['contract_type', 'contract', 'subscription_type', 'billing_cycle', 'plan_type', 'preferredpaymentmode', 'preferred_payment_mode'],
    'monthly_spend_usd': ['monthly_spend_usd', 'monthly_spend', 'monthly_charges', 'monthly_fee', 'monthly_cost', 'spend_monthly', 'estimated_salary', 'estimatedsalary', 'cashbackamount', 'cashback_amount'],
    'discount_pct': ['discount_pct', 'discount', 'discount_percent', 'promo_discount', 'discount_applied', 'orderamounthikefromlastyear', 'order_amount_hike'],
    'autopay_enabled': ['autopay_enabled', 'autopay', 'auto_pay', 'automatic_payment', 'hascrcard', 'has_cr_card'],
    'support_tickets_90d': ['support_tickets_90d', 'support_tickets', 'tickets', 'tickets_90d', 'issues_raised', 'support_queries', 'complain'],
    'complaints_90d': ['complaints_90d', 'complaints', 'complaints_count', 'customer_complaints'],
    'nps_score': ['nps_score', 'nps', 'net_promoter_score', 'satisfaction_score', 'satisfactionscore', 'rating', 'credit_score', 'creditscore'],
    'days_since_last_activity': ['days_since_last_activity', 'last_activity', 'recency', 'days_inactive', 'last_login_days', 'last_login_days_ago', 'lastlogindaysago', 'daysincelastorder', 'day_since_last_order'],
    'late_payments_12m': ['late_payments_12m', 'late_payments', 'missed_payments', 'payments_late', 'payment_failures', 'paymentfailures'],
    'acquisition_channel': ['acquisition_channel', 'channel', 'referred_by', 'marketing_channel', 'source', 'preferredlogindevice', 'preferred_login_device'],

    # Telecom
    'plan_type': ['plan_type', 'plan', 'telecom_plan', 'tariff_type', 'internetservice'],
    'data_usage_gb_30d': ['data_usage_gb_30d', 'data_usage', 'data_gb', 'gb_used', 'internet_usage'],
    'dropped_calls_30d': ['dropped_calls_30d', 'dropped_calls', 'call_drops', 'failed_calls'],
    'network_complaints_90d': ['network_complaints_90d', 'network_issues', 'signal_complaints', 'coverage_complaints'],
    'device_financed': ['device_financed', 'financed', 'phone_installment', 'installment_plan'],
    'international_roaming': ['international_roaming', 'roaming', 'intl_roaming', 'international_plan'],

    # SaaS
    'seats_purchased': ['seats_purchased', 'seats', 'seats_count', 'licenses', 'seats_allocated'],
    'active_users_30d': ['active_users_30d', 'active_users', 'users_30d', 'monthly_active_users', 'mau'],
    'feature_adoption_score': ['feature_adoption_score', 'feature_adoption', 'adoption_rate', 'usage_score', 'avg_weekly_usage_hours', 'weekly_usage_hours'],
    'integrations_connected': ['integrations_connected', 'integrations', 'connected_apps', 'plugins_connected'],
    'admin_logins_30d': ['admin_logins_30d', 'admin_logins', 'admin_activity', 'logins_admin'],
    'onboarding_completed': ['onboarding_completed', 'onboarded', 'onboarding_status', 'completed_onboarding'],

    # Banking
    'account_type': ['account_type', 'account', 'banking_plan', 'checking_savings'],
    'avg_balance_usd': ['avg_balance_usd', 'balance', 'average_balance', 'deposits', 'account_balance'],
    'products_count': ['products_count', 'products', 'num_products', 'holdings', 'number_of_products', 'numofproducts', 'num_of_products'],
    'mobile_logins_30d': ['mobile_logins_30d', 'mobile_logins', 'app_logins', 'logins_30d', 'isactivemember', 'is_active_member'],
    'overdrafts_12m': ['overdrafts_12m', 'overdrafts', 'nsf_fees', 'overdrawn_count'],
    'branch_visits_90d': ['branch_visits_90d', 'branch_visits', 'in_person_visits', 'bank_visits'],

    # eCommerce
    'orders_180d': ['orders_180d', 'orders', 'num_orders', 'purchase_count', 'ordercount', 'order_count'],
    'cart_abandon_rate': ['cart_abandon_rate', 'abandoned_carts', 'cart_abandonment'],
    'avg_order_value_usd': ['avg_order_value_usd', 'aov', 'average_order_value', 'average_spend'],
    'return_rate': ['return_rate', 'refund_rate', 'returns_pct'],
    'app_sessions_30d': ['app_sessions_30d', 'sessions', 'app_visits', 'visits_30d', 'hourspendonapp', 'hour_spend_on_app', 'numberofdeviceregistered'],
    'free_shipping_member': ['free_shipping_member', 'free_shipping', 'premium_shipping', 'vip_shipping'],
    'coupons_used_90d': ['coupons_used_90d', 'coupons_used', 'coupons', 'promo_codes_used', 'couponused', 'coupon_used']
}

def clean_column_name(name: str) -> str:
    """Normalizes column names: lowercase, strip punctuation, underscores, and whitespace."""
    s = str(name).strip().lower()
    s = re.sub(r'[_\-\s]+', '', s)
    return s

def compute_fuzzy_score(col1: str, col2: str) -> float:
    """
    Computes fuzzy match score (0-100) using Levenshtein / Token Sort Ratio.
    Handles exact match and domain synonym overrides.
    """
    c1 = clean_column_name(col1)
    c2 = clean_column_name(col2)
    
    if c1 == c2:
        return 100.0
        
    # Check domain synonym lookup
    for key, syns in SYNONYMS.items():
        if clean_column_name(key) == c2 and any(clean_column_name(syn) == c1 for syn in syns):
            return 95.0
        if clean_column_name(key) == c1 and any(clean_column_name(syn) == c2 for syn in syns):
            return 95.0

    if HAS_RAPIDFUZZ:
        # RapidFuzz Token Sort Ratio handles re-ordered words: 'monthly_spend' vs 'spend_monthly'
        score = fuzz.token_sort_ratio(col1.lower(), col2.lower())
        score_clean = fuzz.ratio(c1, c2)
        return float(max(score, score_clean))
    else:
        # Standard difflib SequenceMatcher fallback
        matcher = difflib.SequenceMatcher(None, c1, c2)
        return float(matcher.ratio() * 100.0)

def detect_industry_fuzzy(headers: list) -> str:
    """
    Detects which industry the uploaded headers best match using fuzzy column matching.
    """
    signatures = {
        'telecom': ['phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity', 'totalcharges', 'roaming', 'droppedcalls', 'datausage'],
        'banking': ['creditscore', 'estimatedsalary', 'numofproducts', 'hascrcard', 'isactivemember', 'avgbalance', 'balance', 'branchvisit', 'overdraft'],
        'saas': ['seatspurchased', 'activeusers', 'featureadoption', 'integrationsconnected', 'adminlogin', 'onboarding'],
        'ecommerce': ['preferredlogindevice', 'citytier', 'preferredpaymentmode', 'hourspendonapp', 'satisfactionscore', 'orderamounthike', 'ordercount', 'daysincelastorder', 'cashbackamount', 'cartabandon'],
        'utilities': ['smartmeter', 'outage', 'monthlyusage', 'moveflag'],
        'insurance': ['policytype', 'premium', 'claim', 'policycount', 'agentcontact', 'renewalday'],
        'healthcare': ['appointment', 'missedappointment', 'patient', 'caregap', 'provider'],
        'hospitality': ['stay', 'nightlyrate', 'guest', 'rewardpoint', 'cancellation'],
        'education': ['course', 'completionrate', 'assignment', 'advisor'],
        'retail': ['loyaltytier', 'basket', 'coupon', 'storepreference']
    }
    
    cleaned_headers = [clean_column_name(h) for h in headers]
    industry_scores = {ind: 0.0 for ind in INDUSTRY_SCHEMAS.keys()}
    
    # Check signature keywords
    for ind, sig_list in signatures.items():
        for sig in sig_list:
            for h in cleaned_headers:
                if sig in h:
                    industry_scores[ind] += 25.0

    # Calculate average best fuzzy match score against each industry schema
    for ind, schema_cols in INDUSTRY_SCHEMAS.items():
        match_scores = []
        for h in headers:
            best_col_score = max(compute_fuzzy_score(h, s_col) for s_col in schema_cols)
            if best_col_score >= 60.0:
                match_scores.append(best_col_score)
        if match_scores:
            industry_scores[ind] += np.sum(match_scores) / len(schema_cols)

    best_industry = max(industry_scores, key=industry_scores.get)
    if industry_scores[best_industry] < 10.0:
        return 'telecom'
    return best_industry

def map_columns_fuzzy(headers: list, target_industry: str, threshold: float = 60.0) -> dict:
    """
    Maps uploaded custom column headers to target industry standard features using Fuzzy Matching.
    Returns:
        {
            uploaded_col: {
                'target': standard_col,
                'confidence': float (0-100),
                'method': 'exact' | 'synonym' | 'fuzzy' | 'id_filter' | 'target_label' | 'unmapped'
            }
        }
    """
    mapping = {}
    standard_cols = INDUSTRY_SCHEMAS.get(target_industry, INDUSTRY_SCHEMAS['telecom'])
    
    # 1. Map ID columns first
    id_syns = SYNONYMS.get('customer_id', [])
    for u_col in headers:
        c = clean_column_name(u_col)
        if any(clean_column_name(syn) == c for syn in id_syns):
            mapping[u_col] = {
                'target': 'customer_id',
                'confidence': 100.0,
                'method': 'id_filter'
            }
            
    # 2. Map Target / Churn label
    churn_syns = SYNONYMS.get('churned', [])
    for u_col in headers:
        if u_col in mapping:
            continue
        c = clean_column_name(u_col)
        if any(clean_column_name(syn) == c for syn in churn_syns):
            mapping[u_col] = {
                'target': 'churned',
                'confidence': 100.0,
                'method': 'target_label'
            }

    # 3. Match remaining features against industry schema
    used_targets = set()
    for u_col in headers:
        if u_col in mapping:
            continue
            
        best_target = None
        best_score = 0.0
        match_method = 'fuzzy'
        
        for s_col in standard_cols:
            if s_col in used_targets:
                continue
                
            score = compute_fuzzy_score(u_col, s_col)
            if score > best_score:
                best_score = score
                best_target = s_col
                if score == 100.0:
                    match_method = 'exact'
                elif score >= 90.0:
                    match_method = 'synonym'
                else:
                    match_method = 'fuzzy'
                    
        if best_target and best_score >= threshold:
            mapping[u_col] = {
                'target': best_target,
                'confidence': round(best_score, 1),
                'method': match_method
            }
            used_targets.add(best_target)
        else:
            mapping[u_col] = {
                'target': None,
                'confidence': round(best_score, 1) if best_target else 0.0,
                'method': 'unmapped'
            }

    return mapping

def map_columns_fuzzy_simple(headers: list, target_industry: str, threshold: float = 60.0) -> dict:
    """Returns simple {uploaded_col: standard_col} mapping for pipeline compatibility."""
    detailed = map_columns_fuzzy(headers, target_industry, threshold)
    return {k: v['target'] for k, v in detailed.items() if v.get('target') is not None}

def map_target_values(series: pd.Series) -> pd.Series:
    """Converts binary targets (Yes/No, True/False, Exited, etc.) to 0/1 integers."""
    val_map = {
        'yes': 1, 'no': 0, 'y': 1, 'n': 0,
        'true': 1, 'false': 0, 't': 1, 'f': 0,
        'exited': 1, 'retained': 0, 'churned': 1,
        'attrited customer': 1, 'existing customer': 0,
        '1': 1, '0': 0, 1: 1, 0: 0
    }
    if series.dtype == object or series.dtype == str or series.dtype == bool:
        return series.astype(str).str.strip().str.lower().map(val_map).fillna(0).astype(int)
    return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)


