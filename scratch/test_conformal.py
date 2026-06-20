import os
import sys
import django
import pandas as pd
import io

# Setup Django environment
sys.path.append(r'c:\Users\Saran\Documents\test_proj')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'churn_dashboard.settings')
django.setup()

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import RequestFactory
from dashboard.views import upload_csv

def test_upload_csv():
    print("=== Testing upload_csv endpoint ===")
    
    # Create a dummy CSV content matching telecom schema
    csv_data = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges,Churn
7590-VHVEG,Female,0,Yes,No,1,No,No phone service,DSL,No,Yes,No,No,No,No,Month-to-month,Yes,Electronic check,29.85,29.85,No
5575-GNVDE,Male,0,No,No,34,Yes,No,DSL,Yes,No,Yes,No,No,No,One year,No,Mailed check,56.95,1889.5,No
"""
    
    uploaded_file = SimpleUploadedFile("telecom_test.csv", csv_data.encode('utf-8'), content_type="text/csv")
    
    # Construct request
    rf = RequestFactory()
    request = rf.post('/dashboard/upload_csv/', {'file': uploaded_file})
    request.FILES['file'] = uploaded_file
    
    # Seek to 0 so the view can read it
    uploaded_file.seek(0)
    
    # Call view
    response = upload_csv(request)
    print("Status code:", response.status_code)
    # Print first 500 characters of response content
    print("Content:", response.content.decode('utf-8')[:500])

if __name__ == '__main__':
    test_upload_csv()
