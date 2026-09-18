import os
import sys
import django
import pandas as pd
import io

# Setup stdout encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Setup Django environment
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'churn_dashboard.settings')
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory
from dashboard.views import upload_csv

def test_real_world_files():
    files_to_test = {
        'telecom': ('WA_Fn-UseC_-Telco-Customer-Churn.csv', 'telecom'),
        'banking': ('Bank_Churn_Modelling.csv', 'banking'),
        'saas': ('SaaS_customer_subscription_churn_usage_patterns.csv', 'saas'),
        'ecommerce': ('E Commerce Dataset(E Comm).csv', 'ecommerce')
    }
    
    print("=== Testing Uploads with Real-World Datasets ===")
    for industry_name, (filename, expected_industry) in files_to_test.items():
        filepath = os.path.join(CURRENT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        print(f"\n--- Testing {filename} (Expected: {expected_industry}) ---")
        # Load the first 10 rows
        df = pd.read_csv(filepath, nrows=10)
        print("Loaded DataFrame columns:", df.columns.tolist())
        
        # Write to csv buffer
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        print("CSV Data first 100 chars:", csv_data[:100])
        
        uploaded_file = SimpleUploadedFile(filename, csv_data.encode('utf-8'), content_type="text/csv")
        
        # Construct request
        rf = RequestFactory()
        request = rf.post('/dashboard/upload_csv/', {'file': uploaded_file})
        request.FILES['file'] = uploaded_file
        
        # Call upload_csv view
        response = upload_csv(request)
        print("Status code:", response.status_code)
        
        import json
        resp_data = json.loads(response.content.decode('utf-8'))
        if response.status_code == 200:
            print(f"Detected Industry: {resp_data.get('detected_industry')} (Key: {resp_data.get('industry_key')})")
            print("Mapped Columns:")
            print(resp_data.get('column_mappings'))
            print(f"Total Records: {resp_data.get('total_records')}")
            print(f"Batch confident rate: {resp_data.get('batch_confident_rate')}%")
            print(f"Average churn risk: {resp_data.get('avg_churn_risk')}%")
            print("First record preview:")
            if resp_data.get('results'):
                first_rec = resp_data.get('results')[0]
                print(f"  Churn prob: {first_rec.get('churn_probability (%)')}%")
                print(f"  Conformal set: {first_rec.get('conformal_prediction_set')}")
                print(f"  Action: {first_rec.get('recommended_business_action')}")
        else:
            print("Error response:", resp_data)

if __name__ == '__main__':
    test_real_world_files()
