import requests
import os
import pandas as pd
import io

URL = "http://localhost:8000/augment-db/"

# Create a small dummy CSV structure matching Telecom schema
dummy_data = {
    'signup_year': [2024, 2025, 2023, 2024, 2025],
    'region': ['West', 'South', 'Midwest', 'Northeast', 'West'],
    'customer_segment': ['standard', 'premium', 'value', 'standard', 'premium'],
    'age': [35, 42, 28, 51, 40],
    'tenure_months': [12, 24, 3, 36, 18],
    'contract_type': ['annual', 'multi_year', 'month_to_month', 'annual', 'month_to_month'],
    'monthly_spend_usd': [80.0, 120.0, 50.0, 95.0, 110.0],
    'discount_pct': [0.0, 0.1, 0.05, 0.15, 0.0],
    'autopay_enabled': [1, 1, 0, 1, 0],
    'support_tickets_90d': [2, 0, 5, 1, 4],
    'complaints_90d': [0, 0, 2, 0, 1],
    'nps_score': [8, 9, 5, 8, 6],
    'days_since_last_activity': [10, 4, 25, 12, 18],
    'late_payments_12m': [0, 0, 1, 0, 1],
    'acquisition_channel': ['organic', 'paid_search', 'social', 'referral', 'organic'],
    'plan_type': ['postpaid', 'family', 'prepaid', 'postpaid', 'prepaid'],
    'data_usage_gb_30d': [15.0, 25.0, 2.0, 10.0, 30.0],
    'dropped_calls_30d': [1, 0, 6, 2, 8],
    'network_complaints_90d': [0, 0, 1, 0, 2],
    'device_financed': [0, 1, 0, 0, 0],
    'international_roaming': [0, 1, 0, 0, 1],
    'churned': [0, 0, 1, 0, 1]
}

df = pd.DataFrame(dummy_data)

# Save test files
os.makedirs("scratch", exist_ok=True)
train_file_path = "scratch/telecom_test_train_data.csv"
val_file_path = "scratch/telecom_test_val_data.csv"
split_file_path = "scratch/telecom_test_split_data.csv"

df.to_csv(train_file_path, index=False)
df.to_csv(val_file_path, index=False)
df.to_csv(split_file_path, index=False)

def test_file_upload(file_path, expected_ingested_as):
    print(f"\n--- Testing Augmentation for {os.path.basename(file_path)} ---")
    with open(file_path, 'rb') as f:
        files = {'file': (os.path.basename(file_path), f, 'text/csv')}
        response = requests.post(URL, files=files)
        
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        res_json = response.json()
        print("Success: TRUE")
        print(f"Message: {res_json.get('message')}")
        print(f"Ingested As: {res_json.get('ingested_as')}")
        print(f"Train Added Rows: {res_json.get('train_records_added')}")
        print(f"Val Added Rows: {res_json.get('val_records_added')}")
        
        # Validation checks
        assert res_json.get('ingested_as') == expected_ingested_as, f"Expected {expected_ingested_as}, got {res_json.get('ingested_as')}"
        print("PASS: Ingestion type matches expected value!")
    else:
        print(f"FAILED: {response.text}")

try:
    test_file_upload(train_file_path, 'Train Set')
    test_file_upload(val_file_path, 'Validation Set')
    test_file_upload(split_file_path, 'Split Train/Val (80/20)')
    print("\nAll augmentation tests completed successfully!")
finally:
    # Clean up files
    for p in [train_file_path, val_file_path, split_file_path]:
        if os.path.exists(p):
            os.remove(p)
