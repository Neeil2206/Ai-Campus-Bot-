import os
import sys
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")

print("--- STEP 1 & 2 ---")
print("BASE_DIR =", BASE_DIR)
print("ENV_PATH =", ENV_PATH)
print("ENV EXISTS =", os.path.isfile(ENV_PATH))

# Clear key from os.environ first to see what load_dotenv does
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

load_dotenv(ENV_PATH, override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

print("\n--- STEP 3 ---")
print("API KEY DETECTED:", bool(OPENAI_API_KEY))
print("API KEY LENGTH:", len(OPENAI_API_KEY))

print("\n--- STEP 4 & 5 ---")
env_file = Path(ENV_PATH)
if env_file.exists():
    raw_content = env_file.read_bytes()
    print("RAW FILE BYTES (first 100):", raw_content[:100])
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            parts = line.split("=", 1)
            name = parts[0].strip()
            print("ENV VARIABLE FOUND:", repr(name))

print("\n--- STEP 6 ---")
found_key_line = False
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r", encoding="utf-8-sig") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("OPENAI_API_KEY="):
                found_key_line = True
print("OPENAI_API_KEY line exists:", found_key_line)

print("\n--- STEP 11: TEST OPENAI SEPARATELY ---")
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        print("Initializing OpenAI client...")
        client = OpenAI(api_key=OPENAI_API_KEY)
        # Try a tiny API call
        models = client.models.list()
        print("OpenAI models list success: True")
    except Exception as e:
        print("OpenAI test failed:", type(e).__name__, str(e))
else:
    print("Cannot test OpenAI because key is not detected.")
