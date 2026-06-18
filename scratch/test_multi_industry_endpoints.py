import urllib.request
import urllib.parse
import json
import os

BASE_URL = "http://127.0.0.1:8000"

def test_predict_endpoint(industry, data):
    print(f"\n--- Testing Predict Endpoint ({industry.upper()}) ---")
    url = f"{BASE_URL}/predict/?industry={industry}&confidence=0.85"
    
    req = urllib.request.Request(
        url,
        data=json.dumps(data).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            print("Status: SUCCESS")
            print("Churn Probability:", res.get("churn_probability"), "%")
            print("Prediction:", res.get("prediction"))
            print("Risk Level:", res.get("risk_level"))
            print("Drivers:", res.get("drivers"))
            print("Recommendations:", res.get("recommendations"))
            print("UQ Set:", res.get("uq", {}).get("set"))
            print("RAG Sources Count:", len(res.get("rag_sources", [])))
    except Exception as e:
        print("Status: FAILED")
        print("Error:", str(e))

def test_batch_upload(filename, expected_industry):
    print(f"\n--- Testing Batch Upload ({filename}) ---")
    url = f"{BASE_URL}/upload-csv/?confidence=0.85"
    
    # Read first few rows of mock file
    filepath = rf"C:\Users\Saran\Documents\forMock\mock_churn_data\{filename}"
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return
        
    with open(filepath, 'rb') as f:
        file_content = f.read()
        
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode('utf-8') + file_content + f"\r\n--{boundary}--\r\n".encode('utf-8')
    
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            'Content-Type': f'multipart/form-data; boundary={boundary}',
            'Content-Length': str(len(body))
        }
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            print("Status: SUCCESS")
            det_ind = str(res.get("detected_industry")).encode('ascii', errors='replace').decode('ascii')
            print("Detected Industry:", det_ind)
            print("Column Mappings:", res.get("column_mappings"))
            print("Total Records:", res.get("total_records"))
            if res.get("results"):
                print("First record probability:", res.get("results")[0].get("churn_probability (%)"), "%")
                print("First record action:", res.get("results")[0].get("recommended_business_action"))
    except Exception as e:
        print("Status: FAILED")
        print("Error:", str(e))

def main():
    # Telecom test data
    telecom_data = {
        "signup_year": 2024,
        "region": "West",
        "customer_segment": "premium",
        "age": 42,
        "tenure_months": 12,
        "contract_type": "month_to_month",
        "monthly_spend_usd": 120.50,
        "discount_pct": 0.0,
        "autopay_enabled": 0,
        "support_tickets_90d": 6,
        "complaints_90d": 2,
        "nps_score": 5,
        "days_since_last_activity": 22,
        "late_payments_12m": 1,
        "acquisition_channel": "social",
        "plan_type": "family",
        "data_usage_gb_30d": 45.2,
        "dropped_calls_30d": 15,
        "network_complaints_90d": 3,
        "device_financed": 0,
        "international_roaming": 0
    }
    
    # SaaS test data
    saas_data = {
        "signup_year": 2023,
        "region": "Northeast",
        "customer_segment": "enterprise",
        "age": 35,
        "tenure_months": 36,
        "contract_type": "multi_year",
        "monthly_spend_usd": 850.00,
        "discount_pct": 0.10,
        "autopay_enabled": 1,
        "support_tickets_90d": 0,
        "complaints_90d": 0,
        "nps_score": 9,
        "days_since_last_activity": 2,
        "late_payments_12m": 0,
        "acquisition_channel": "sales",
        "seats_purchased": 100,
        "active_users_30d": 95,
        "feature_adoption_score": 0.85,
        "integrations_connected": 8,
        "admin_logins_30d": 20,
        "onboarding_completed": 1
    }
    
    test_predict_endpoint("telecom", telecom_data)
    test_predict_endpoint("saas", saas_data)
    
    # Test batch uploads on actual mock files
    test_batch_upload("telecom_churn_mock_data.csv", "telecom")
    test_batch_upload("saas_churn_mock_data.csv", "saas")
    test_batch_upload("retail_churn_mock_data.csv", "retail")
    test_batch_upload("banking_churn_mock_data.csv", "banking")

if __name__ == "__main__":
    main()
