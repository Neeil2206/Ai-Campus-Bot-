import urllib.request
import json

url = "http://127.0.0.1:5000/api/status"

try:
    response = urllib.request.urlopen(url)
    res_data = json.loads(response.read().decode("utf-8"))
    print("STATUS API RESPONSE:")
    print(json.dumps(res_data, indent=2))
except Exception as e:
    print("REQUEST FAILED:", e)
