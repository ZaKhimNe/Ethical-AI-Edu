import os, random, requests
from dotenv import load_dotenv

load_dotenv()
URL = os.environ["SUPABASE_URL"]
KEY = os.environ["SUPABASE_KEY"]

headers = {
    "apikey": KEY,
    "Authorization": f"Bearer {KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=minimal"
}

resp_del = requests.delete(
    f"{URL}/rest/v1/assignments?code=neq.null",
    headers=headers
)
print(f"[OK] Đã xóa assignments cũ: {resp_del.status_code}")

N_PEOPLE             = 30
SCENARIOS_PER_PERSON = 10   
MIN_PER_SCENARIO     = 3   


random.seed(42)

pool = list(range(1, 101)) * MIN_PER_SCENARIO
random.shuffle(pool)

assignments = []
for i in range(N_PEOPLE):
    code        = f"VN{i+1:03d}"
    start       = i * SCENARIOS_PER_PERSON
    scenario_ids = pool[start:start + SCENARIOS_PER_PERSON]
    assignments.append({
        "code":         code,
        "scenario_ids": scenario_ids,
    })

resp = requests.post(
    f"{URL}/rest/v1/assignments",
    headers=headers,
    json=assignments
)

if resp.status_code in (200, 201):
    print(f"[OK] Generated {len(assignments)} codes")
    print()
    print("=== DANH SÁCH CODE GỬI CHO BẠN BÈ ===")
    for a in assignments:
        print(f"  {a['code']}")
else:
    print(f"[ERR] {resp.status_code}: {resp.text}")