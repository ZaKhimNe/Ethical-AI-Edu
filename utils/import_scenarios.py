import json, os, requests
from dotenv import load_dotenv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
URL = os.environ["SUPABASE_URL"]
KEY = os.environ["SUPABASE_KEY"]

headers = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal,resolution=merge-duplicates"
}

with open(os.path.join(ROOT, "data/pipeline_results/edu_scenarios_prompted.json"), encoding="utf-8") as f:
    data = json.load(f)

rows = [{
    "id":             s["id"],
    "topic":          s["topic"],
    "title":          s["title"],
    "context":        s["enhanced_context"],
    "option_1":       s["option_1"],
    "option_2":       s["option_2"],
    "option_3":       s.get("option_3", ""),
    "option_label":   s["option_label"],
    "bias_direction": s.get("bias_direction", ""),
    "cultural_dimension": s.get("cultural_dimension", ""),
} for s in data["scenarios"]]


resp = requests.post(
    f"{URL}/rest/v1/scenarios",
    headers=headers,
    json=rows
)

if resp.status_code in (200, 201):
    print(f"[OK] Imported {len(rows)} scenarios")
else:
    print(f"[ERR] {resp.status_code}: {resp.text}")