import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Standard features dictionaries for each industry
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

# Synonyms for exact matching enhancement
SYNONYMS = {
    # Target column synonyms (for auto-detection of churn labels)
    'churned': ['churned', 'churn', 'exited', 'is_churned', 'attrition', 'attrition_flag', 'churn_label', 'churn_flag', 'target', 'label'],

    # ID / exclusion columns (mapped to None to signal removal)
    'customer_id': ['customer_id', 'customerid', 'user_id', 'userid', 'id', 'cust_id', 'account_id', 'member_id'],

    # Common features
    'signup_year': ['signup_year', 'signup', 'registered_year', 'join_year', 'year_joined', 'registration_year', 'signup_date', 'joining_date', 'registration_date'],
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

    # Telecom specific
    'plan_type': ['plan_type', 'plan', 'telecom_plan', 'tariff_type'],
    'data_usage_gb_30d': ['data_usage_gb_30d', 'data_usage', 'data_gb', 'gb_used', 'internet_usage'],
    'dropped_calls_30d': ['dropped_calls_30d', 'dropped_calls', 'call_drops', 'failed_calls'],
    'network_complaints_90d': ['network_complaints_90d', 'network_issues', 'signal_complaints', 'coverage_complaints'],
    'device_financed': ['device_financed', 'financed', 'phone_installment', 'installment_plan'],
    'international_roaming': ['international_roaming', 'roaming', 'intl_roaming', 'international_plan'],

    # SaaS specific
    'seats_purchased': ['seats_purchased', 'seats', 'seats_count', 'licenses', 'seats_allocated'],
    'active_users_30d': ['active_users_30d', 'active_users', 'users_30d', 'monthly_active_users', 'mau'],
    'feature_adoption_score': ['feature_adoption_score', 'feature_adoption', 'adoption_rate', 'usage_score', 'avg_weekly_usage_hours', 'weekly_usage_hours'],
    'integrations_connected': ['integrations_connected', 'integrations', 'connected_apps', 'plugins_connected'],
    'admin_logins_30d': ['admin_logins_30d', 'admin_logins', 'admin_activity', 'logins_admin'],
    'onboarding_completed': ['onboarding_completed', 'onboarded', 'onboarding_status', 'completed_onboarding'],

    # Retail specific
    'loyalty_tier': ['loyalty_tier', 'loyalty', 'membership_tier', 'rewards_level'],
    'visits_90d': ['visits_90d', 'visits', 'store_visits', 'frequency_of_visits'],
    'avg_basket_usd': ['avg_basket_usd', 'avg_basket', 'basket_size', 'average_order_value', 'aov'],
    'returns_90d': ['returns_90d', 'returns', 'returned_items', 'refunds'],
    'coupons_used_90d': ['coupons_used_90d', 'coupons_used', 'coupons', 'promo_codes_used'],
    'store_preference': ['store_preference', 'preferred_store', 'channel_preference', 'shopping_channel'],

    # Banking specific
    'account_type': ['account_type', 'account', 'banking_plan', 'checking_savings'],
    'avg_balance_usd': ['avg_balance_usd', 'balance', 'average_balance', 'deposits', 'account_balance'],
    'products_count': ['products_count', 'products', 'num_products', 'holdings', 'number_of_products', 'numofproducts', 'num_of_products'],
    'mobile_logins_30d': ['mobile_logins_30d', 'mobile_logins', 'app_logins', 'logins_30d', 'isactivemember', 'is_active_member'],
    'overdrafts_12m': ['overdrafts_12m', 'overdrafts', 'nsf_fees', 'overdrawn_count'],
    'branch_visits_90d': ['branch_visits_90d', 'branch_visits', 'in_person_visits', 'bank_visits'],

    # eCommerce specific
    'orders_180d': ['orders_180d', 'orders', 'num_orders', 'purchase_count', 'ordercount', 'order_count'],
    'cart_abandon_rate': ['cart_abandon_rate', 'abandoned_carts', 'cart_abandonment'],
    'avg_order_value_usd': ['avg_order_value_usd', 'aov', 'average_order_value', 'average_spend'],
    'return_rate': ['return_rate', 'refund_rate', 'returns_pct'],
    'app_sessions_30d': ['app_sessions_30d', 'sessions', 'app_visits', 'visits_30d', 'hourspendonapp', 'hour_spend_on_app', 'numberofdeviceregistered'],
    'free_shipping_member': ['free_shipping_member', 'free_shipping', 'premium_shipping', 'vip_shipping'],
    'coupons_used_90d': ['coupons_used_90d', 'coupons_used', 'coupons', 'promo_codes_used', 'couponused', 'coupon_used'],

    # Education specific
    'program_type': ['program_type', 'program', 'course_category', 'degree'],
    'courses_enrolled': ['courses_enrolled', 'courses', 'enrolled_courses', 'classes'],
    'completion_rate': ['completion_rate', 'course_progress', 'completion_pct'],
    'logins_30d': ['logins_30d', 'student_logins', 'platform_logins'],
    'assignments_late_90d': ['assignments_late_90d', 'late_assignments', 'late_submissions'],
    'advisor_contacts_90d': ['advisor_contacts_90d', 'advisor_meetings', 'support_meetings'],

    # Healthcare specific
    'plan_category': ['plan_category', 'health_plan', 'insurance_category'],
    'appointments_12m': ['appointments_12m', 'appointments', 'doctor_visits'],
    'missed_appointments_12m': ['missed_appointments_12m', 'missed_appointments', 'no_shows'],
    'portal_logins_90d': ['portal_logins_90d', 'portal_logins', 'patient_portal_logins'],
    'care_gap_count': ['care_gap_count', 'care_gaps', 'open_gaps'],
    'primary_provider_assigned': ['primary_provider_assigned', 'pcp_assigned', 'doctor_assigned'],

    # Hospitality specific
    'membership_level': ['membership_level', 'status_level', 'hospitality_tier'],
    'stays_12m': ['stays_12m', 'bookings', 'stays', 'visits_12m'],
    'avg_nightly_rate_usd': ['avg_nightly_rate_usd', 'nightly_rate', 'room_rate_avg'],
    'review_rating': ['review_rating', 'guest_rating', 'feedback_score'],
    'reward_points_balance': ['reward_points_balance', 'points', 'loyalty_points'],
    'cancellations_12m': ['cancellations_12m', 'cancellations', 'cancelled_bookings'],

    # Insurance specific
    'policy_type': ['policy_type', 'coverage_type', 'insurance_type'],
    'premium_usd': ['premium_usd', 'premium', 'monthly_premium', 'cost'],
    'claims_24m': ['claims_24m', 'claims', 'claims_submitted'],
    'policy_count': ['policy_count', 'policies', 'num_policies'],
    'agent_contact_90d': ['agent_contact_90d', 'agent_meetings', 'agent_interactions'],
    'renewal_days_remaining': ['renewal_days_remaining', 'days_to_renewal', 'days_left'],

    # Utilities specific
    'service_type': ['service_type', 'utility_type', 'energy_type'],
    'avg_monthly_usage': ['avg_monthly_usage', 'kwh_usage', 'usage_units'],
    'outages_12m': ['outages_12m', 'outages', 'power_cuts'],
    'smart_meter_enabled': ['smart_meter_enabled', 'smart_meter', 'digital_meter'],
    'paperless_billing': ['paperless_billing', 'digital_billing', 'online_billing'],
    'move_flag_90d': ['move_flag_90d', 'moving', 'relocation_flag']
}

def clean_name(name: str) -> str:
    """Standardizes column names for matching."""
    return str(name).strip().lower().replace('_', '').replace(' ', '')

def get_similarity_score(col1: str, col2: str, vectorizer: TfidfVectorizer) -> float:
    """Calculates semantic similarity using character n-gram TF-IDF and Cosine Similarity."""
    c1_clean = clean_name(col1)
    c2_clean = clean_name(col2)
    
    # Check exact match
    if c1_clean == c2_clean:
        return 1.0
        
    # Check synonym list
    for key, syns in SYNONYMS.items():
        if clean_name(key) == c2_clean and any(clean_name(syn) == c1_clean for syn in syns):
            return 0.95
            
    # Subword TF-IDF Cosine Similarity fallback
    try:
        vecs = vectorizer.transform([c1_clean, c2_clean])
        return float(cosine_similarity(vecs[0], vecs[1])[0, 0])
    except:
        return 0.0

def detect_industry(headers: list) -> str:
    """
    Assumes/Detects which industry the uploaded headers belong to.
    Computes a match score for each industry schema.
    """
    scores = {}
    
    # 1. Identify signature columns for heuristic matching
    signatures = {
        'telecom': ['phoneservice', 'multiplelines', 'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport', 'streamingtv', 'streamingmovies', 'totalcharges', 'roaming', 'droppedcalls', 'datausage'],
        'banking': ['creditscore', 'estimatedsalary', 'numofproducts', 'hascrcard', 'isactivemember', 'avgbalance', 'balance', 'branchvisit', 'overdraft'],
        'saas': ['seatspurchased', 'activeusers', 'featureadoption', 'integrationsconnected', 'adminlogin', 'onboarding'],
        'ecommerce': ['preferredlogindevice', 'citytier', 'preferredpaymentmode', 'hourspendonapp', 'satisfactionscore', 'complain', 'orderamounthike', 'ordercount', 'daysincelastorder', 'cashbackamount', 'cartabandon'],
        'utilities': ['smartmeter', 'outage', 'monthlyusage', 'moveflag'],
        'insurance': ['policytype', 'premium', 'claim', 'policycount', 'agentcontact', 'renewalday'],
        'healthcare': ['appointment', 'missedappointment', 'patient', 'caregap', 'provider'],
        'hospitality': ['stay', 'nightlyrate', 'guest', 'rewardpoint', 'cancellation'],
        'education': ['course', 'completionrate', 'assignment', 'advisor'],
        'retail': ['loyaltytier', 'basket', 'coupon', 'storepreference']
    }
    
    # Exclude ID and target columns to prevent them from skewing feature matching
    id_syns = SYNONYMS.get('customer_id', []) + ['customer_id', 'customerid', 'user_id', 'userid', 'id', 'cust_id', 'account_id', 'member_id', 'rownumber', 'surname']
    id_syns_clean = [clean_name(s) for s in id_syns]
    target_syns = SYNONYMS.get('churned', []) + ['churned', 'churn', 'exited', 'class', 'target', 'label']
    target_syns_clean = [clean_name(s) for s in target_syns]
    
    headers_to_match = []
    sig_counts = {ind: 0 for ind in INDUSTRY_SCHEMAS.keys()}
    for h in headers:
        h_clean = clean_name(h)
        if h_clean in id_syns_clean or h_clean in target_syns_clean:
            continue
        headers_to_match.append(h)
        for ind, sig_list in signatures.items():
            for sig in sig_list:
                if sig in h_clean:
                    sig_counts[ind] += 1
                    
    if not headers_to_match:
        return 'telecom'
    
    # Fit TF-IDF on all schema words to build vocab
    all_vocab = []
    for schema in INDUSTRY_SCHEMAS.values():
        all_vocab.extend([clean_name(col) for col in schema])
    for syns in SYNONYMS.values():
        all_vocab.extend([clean_name(syn) for syn in syns])
    
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    vectorizer.fit(all_vocab + [clean_name(h) for h in headers_to_match])
    
    for industry, standard_cols in INDUSTRY_SCHEMAS.items():
        matched_count = 0
        total_score = 0.0
        
        for u_col in headers_to_match:
            best_sim = 0.0
            for s_col in standard_cols:
                sim = get_similarity_score(u_col, s_col, vectorizer)
                if sim > best_sim:
                    best_sim = sim
            
            # Count it if it is a reasonable match (sim > 0.40)
            if best_sim > 0.40:
                matched_count += 1
                total_score += best_sim
                
        # Weigh industry based on proportion of features matched
        prop = matched_count / len(standard_cols)
        scores[industry] = (total_score * prop) + (sig_counts.get(industry, 0) * 2.0)

    # Return the industry with the highest score
    best_ind = max(scores, key=scores.get)
    # Fallback to telecom if score is extremely low
    if scores[best_ind] < 0.1:
        return 'telecom'
    return best_ind

def map_columns_nlp(headers: list, target_industry: str) -> dict:
    """
    Maps uploaded custom column headers to target industry standard features.
    Returns a mapping dictionary: {uploaded_col: standard_col}
    """
    mapping = {}
    standard_cols = INDUSTRY_SCHEMAS[target_industry]
    
    # 1. Detect and map ID/exclusion columns first to prevent incorrect mapping to feature columns
    id_syns = SYNONYMS.get('customer_id', []) + ['customer_id', 'customerid', 'user_id', 'userid', 'id', 'cust_id', 'account_id', 'member_id', 'rownumber', 'surname', 'customerid']
    id_syns_clean = [clean_name(s) for s in id_syns]
    
    remaining_headers = []
    for u_col in headers:
        if clean_name(u_col) in id_syns_clean:
            mapping[u_col] = 'customer_id'
        else:
            remaining_headers.append(u_col)
            
    # 2. Check exact matches on standard features
    used_targets = set()
    unmapped_headers = []
    for u_col in remaining_headers:
        u_clean = clean_name(u_col)
        matched = False
        for s_col in standard_cols:
            if u_clean == clean_name(s_col):
                mapping[u_col] = s_col
                used_targets.add(s_col)
                matched = True
                break
        if not matched:
            unmapped_headers.append(u_col)
                
    if not unmapped_headers:
        return mapping
        
    all_words = [clean_name(col) for col in standard_cols] + [clean_name(h) for h in headers]
    for syns in SYNONYMS.values():
        all_words.extend([clean_name(syn) for syn in syns])
        
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    vectorizer.fit(all_words)
    
    for u_col in unmapped_headers:
        best_match = None
        best_sim = 0.0
        
        for s_col in standard_cols:
            sim = get_similarity_score(u_col, s_col, vectorizer)
            
            if sim > best_sim:
                best_sim = sim
                best_match = s_col
                
        # Only map if it meets similarity threshold and target hasn't been mapped yet
        if best_sim > 0.35 and best_match not in used_targets:
            mapping[u_col] = best_match
            used_targets.add(best_match)
            
    return mapping


def map_target_values(series):
    """
    Maps common target label formats to binary 0/1.
    Handles: Yes/No, True/False, Existing/Attrited Customer, etc.
    """
    TARGET_VALUE_MAP = {
        'Yes': 1, 'No': 0, 'yes': 1, 'no': 0,
        'True': 1, 'False': 0, 'true': 1, 'false': 0,
        'YES': 1, 'NO': 0,
        'Existing Customer': 0, 'Attrited Customer': 1,
        '1': 1, '0': 0, 1: 1, 0: 0,
    }
    if series.dtype == object:
        return series.map(TARGET_VALUE_MAP).fillna(0).astype(int)
    return pd.to_numeric(series, errors='coerce').fillna(0).astype(int)
