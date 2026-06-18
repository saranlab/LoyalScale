import urllib.request
import urllib.parse
import json

def test_rag_prediction():
    print("=== Testing Single Predict Endpoint with RAG ===")
    url = "http://127.0.0.1:8000/predict/?industry=telecom&confidence=0.85"
    data = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 1140.0
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
            print("Probability:", res.get("churn_probability"))
            print("Conformal Set:", res.get("uq", {}).get("set"))
            print("RAG Sources Count:", len(res.get("rag_sources", [])))
            for idx, src in enumerate(res.get("rag_sources", [])):
                print(f"  Source {idx+1}: Sim={src['similarity']}%, Contract={src['contract']}, Churned={src['churned']}")
            
            assert len(res.get("rag_sources", [])) > 0, "RAG sources should be retrieved for Telecom!"
            print("RAG validation passed!")
    except Exception as e:
        print("Failed prediction test:", str(e))
        raise e

def test_dynamic_augmentation():
    print("\n=== Testing Dynamic Model Augmentation ===")
    url = "http://127.0.0.1:8000/augment-db/"
    
    # 5 mock training instances to append
    csv_content = (
        "customer_sex,is_senior,spouse,children,months_active,telephone,more_lines,"
        "internet_service,antivirus,cloud_backup,protection_plan,technical_support,"
        "tv_streaming,movie_streaming,contract_type,digital_bill,billing_method,"
        "monthly_cost,total_cost,Churn\n"
        "Male,0,No,No,12,Yes,No,DSL,No,No,No,No,No,No,Month-to-month,No,Electronic check,70.0,840.0,No\n"
        "Female,1,Yes,No,2,Yes,Yes,Fiber optic,No,Yes,No,No,Yes,Yes,Month-to-month,Yes,Electronic check,105.0,210.0,Yes\n"
        "Male,0,Yes,Yes,45,Yes,Yes,Fiber optic,Yes,Yes,Yes,Yes,Yes,Yes,Two year,Yes,Credit card (automatic),115.0,5175.0,No\n"
        "Female,0,No,No,1,Yes,No,No,No internet service,No internet service,No internet service,No internet service,No internet service,No internet service,Month-to-month,No,Mailed check,20.0,20.0,Yes\n"
        "Male,1,Yes,No,24,Yes,Yes,Fiber optic,No,No,Yes,No,Yes,No,One year,Yes,Bank transfer (automatic),90.0,2160.0,No\n"
    )
    
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="augment_train.csv"\r\n'
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
            print("Message:", res.get("message"))
            print("Response contains stats:", "stats_json" in res)
            
            stats = json.loads(res.get("stats_json"))
            industry_key = list(stats.keys())[0] if stats else "utilities"
            model_stats = stats.get(industry_key)
            if model_stats:
                print("New Total Customers:", model_stats.get("metrics", {}).get("total_customers"))
                print("New Accuracy:", model_stats.get("metrics", {}).get("model_accuracy"))
                print("New AUC:", model_stats.get("metrics", {}).get("model_auc"))
            else:
                print("No stats found for industry:", industry_key)
            
    except urllib.error.HTTPError as e:
        print("Failed augmentation test. HTTP Code:", e.code)
        try:
            print("Error body:", e.read().decode('utf-8'))
        except Exception as read_err:
            print("Could not read error body:", str(read_err))
        raise e
    except Exception as e:
        print("Failed prediction/augmentation test:", str(e))
        raise e

if __name__ == "__main__":
    try:
        test_rag_prediction()
        test_dynamic_augmentation()
    except Exception as e:
        pass

