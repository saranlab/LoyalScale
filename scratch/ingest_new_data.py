import urllib.request
import urllib.parse
import os
import json

def upload_file(file_path):
    print(f"\n=== Uploading {os.path.basename(file_path)} ===")
    url = "http://127.0.0.1:8000/augment-db/"
    
    # Read file content in binary mode
    with open(file_path, 'rb') as f:
        file_content = f.read()
        
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    
    # Construct multipart form-data
    part_header = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(file_path)}"\r\n'
        f"Content-Type: text/csv\r\n\r\n"
    ).encode('utf-8')
    
    part_footer = f"\r\n--{boundary}--\r\n".encode('utf-8')
    
    body = part_header + file_content + part_footer
    
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
            print("Train rows added:", res.get("train_records_added"))
            print("Val rows added:", res.get("val_records_added"))
    except urllib.error.HTTPError as e:
        print("Failed with HTTP Code:", e.code)
        try:
            print("Error body:", e.read().decode('utf-8'))
        except Exception as read_err:
            print("Could not read error body:", str(read_err))
    except Exception as e:
        print("Failed to call augment-db endpoint:", str(e))

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    files_to_upload = [
        os.path.join(BASE_DIR, "SaaS_customer_subscription_churn_usage_patterns.csv"),
        os.path.join(BASE_DIR, "E Commerce Dataset(E Comm).csv"),
        os.path.join(BASE_DIR, "Bank_Churn_Modelling.csv")
    ]
    
    for fp in files_to_upload:
        if os.path.exists(fp):
            upload_file(fp)
        else:
            print(f"File not found: {fp}")
