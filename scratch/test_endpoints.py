import urllib.request
import urllib.parse
import json

def test_single_predict():
    print("=== Testing Single Predict Endpoint ===")
    url = "http://127.0.0.1:8000/predict/?industry=telecom"
    data = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0
    }
    
    req = urllib.request.Request(
        url, 
        data=json.dumps(data).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    
    try:
        with urllib.request.urlopen(req) as response:
            res = json.loads(response.read().decode('utf-8'))
            print("Success!")
            print("Response:", json.dumps(res, indent=2))
    except Exception as e:
        print("Failed to call predict endpoint:", str(e))

def test_batch_upload():
    print("\n=== Testing Batch Upload Endpoint ===")
    url = "http://127.0.0.1:8000/upload-csv/"
    
    # Create a simple mock CSV content
    csv_content = (
        "customer_sex,is_senior,spouse,children,months_active,telephone,more_lines,"
        "internet_service,antivirus,cloud_backup,protection_plan,technical_support,"
        "tv_streaming,movie_streaming,contract_type,digital_bill,billing_method,"
        "monthly_cost,total_cost\n"
        "Male,0,No,No,12,Yes,No,DSL,No,No,No,No,No,No,Month-to-month,No,Electronic check,70.0,840.0\n"
        "Female,1,Yes,No,2,Yes,Yes,Fiber optic,No,Yes,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,105.0,210.0\n"
    )
    
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    # Construct multipart form-data manually
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="test.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
        f"{csv_content}\r\n"
        f"--{boundary}--\r\n"
    ).encode('utf-8')
    
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
            print("Success!")
            det_ind = str(res.get("detected_industry")).encode('ascii', errors='replace').decode('ascii')
            print("Detected Industry:", det_ind)
            print("Column Mappings:", res.get("column_mappings"))
            print("Total Records:", res.get("total_records"))
            print("Sample Result:", json.dumps(res.get("results")[0], indent=2))
    except Exception as e:
        print("Failed to call upload-csv endpoint:", str(e))

def test_batch_upload_saas():
    print("\n=== Testing Batch Upload Endpoint (SaaS) ===")
    url = "http://127.0.0.1:8000/upload-csv/"
    
    csv_content = (
        "subscription_tier,seats_used,support_tickets,usage_frequency,monthly_spend\n"
        "Basic,1,6,Monthly,49.0\n"
        "Enterprise,120,0,Daily,1200.0\n"
    )
    
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="saas_test.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
        f"{csv_content}\r\n"
        f"--{boundary}--\r\n"
    ).encode('utf-8')
    
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
            print("Success!")
            det_ind = str(res.get("detected_industry")).encode('ascii', errors='replace').decode('ascii')
            print("Detected Industry:", det_ind)
            print("Column Mappings:", res.get("column_mappings"))
            print("Total Records:", res.get("total_records"))
            print("Results:")
            for idx, r in enumerate(res.get("results")):
                print(f" Record {idx+1}: Tier={r.get('subscription_tier')}, Tickets={r.get('support_tickets')}, Prob={r.get('churn_probability (%)')}%, Action={r.get('recommended_business_action')}")
    except Exception as e:
        print("Failed to call upload-csv endpoint for SaaS:", str(e))

def test_batch_upload_banking():
    print("\n=== Testing Batch Upload Endpoint (Banking) ===")
    url = "http://127.0.0.1:8000/upload-csv/"
    
    csv_content = (
        "active_client,number_of_products,account_balance,credit_rating,monthly_charges\n"
        "No,1,120.0,490,2500.0\n"
        "Yes,3,145000.0,790,8500.0\n"
    )
    
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="banking_test.csv"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
        f"{csv_content}\r\n"
        f"--{boundary}--\r\n"
    ).encode('utf-8')
    
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
            print("Success!")
            det_ind = str(res.get("detected_industry")).encode('ascii', errors='replace').decode('ascii')
            print("Detected Industry:", det_ind)
            print("Column Mappings:", res.get("column_mappings"))
            print("Total Records:", res.get("total_records"))
            print("Results:")
            for idx, r in enumerate(res.get("results")):
                print(f" Record {idx+1}: Balance={r.get('account_balance')}, Score={r.get('credit_rating')}, Prob={r.get('churn_probability (%)')}%, Action={r.get('recommended_business_action')}")
    except Exception as e:
        print("Failed to call upload-csv endpoint for Banking:", str(e))

def test_root_index():
    print("\n=== Testing Root Index Page (GET /) ===")
    url = "http://127.0.0.1:8000/"
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read().decode('utf-8')
            print("Success! Root index loaded.")
            print("Title check in HTML:", "Multi-Industry Customer Churn Diagnostic Suite" in html)
            print("Confidence Select dropdown check:", "confidenceSelect" in html)
    except Exception as e:
        print("Failed to load root index page:", str(e))

if __name__ == "__main__":
    test_root_index()
    test_single_predict()
    test_batch_upload()
    test_batch_upload_saas()
    test_batch_upload_banking()
