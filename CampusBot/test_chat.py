import urllib.request
import json

url = "http://127.0.0.1:5000/api/chat"
data = json.dumps({"message": "What is the weather tomorrow?"}).encode("utf-8")

req = urllib.request.Request(
    url, 
    data=data, 
    headers={"Content-Type": "application/json"}
)

try:
    response = urllib.request.urlopen(req)
    res_data = json.loads(response.read().decode("utf-8"))
    print("RESPONSE STATUS: SUCCESS")
    print("CHAT BOT RESPONSE:", res_data.get("answer"))
except Exception as e:
    print("REQUEST FAILED:", e)
