import base64
import sys
import requests
import json

# ── Edit these three values ────────────────────────────────────
ENDPOINT = "https://ultrafay--deepseek-ocr-ocrmodel-ocr.modal.run"
TOKEN_ID = "wk-ajj1XFbxJyCbHJlabu8jsy"
TOKEN_SECRET = "ws-LqTMd2hU09mjBvlOLhJ0FE"
# ───────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage: python test_ocr.py <path-to-invoice-image>")
    sys.exit(1)

image_path = sys.argv[1]

print(f"Reading {image_path}...")
with open(image_path, "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

print(f"Sending to Modal endpoint...")
print("(First request will take 30-90s for cold start - model loading into GPU)")

response = requests.post(
    ENDPOINT,
    headers={
        "Content-Type": "application/json",
        "Modal-Key": TOKEN_ID,
        "Modal-Secret": TOKEN_SECRET,
    },
    json={"image_base64": image_b64},
    timeout=300,
)

print(f"\nStatus: {response.status_code}")
print("─" * 60)

if response.status_code == 200:
    data = response.json()
    text = data.get("text", "")

    print("EXTRACTED TEXT:")
    print("─" * 60)
    print(text if text else "(empty)")
    print("─" * 60)
    print(f"\nText length: {len(text)} characters")

    print("\nDEBUG INFO:")
    print(f"  result type: {data.get('debug_result_type')}")
    print(f"  result preview: {data.get('debug_result_preview')}")
    print(f"  files written: {data.get('debug_output_files')}")
else:
    print("ERROR RESPONSE:")
    print(response.text[:2000])