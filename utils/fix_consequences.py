"""
Tách lại option_1/2/3 của 60 scenarios thiếu markers thành format:
  [hành động]. Hệ quả tốt: [...]. Hệ quả xấu: [...]
Dùng Gemini để rewrite. Kết quả ghi đè vào edu_scenarios_prompted.json.
Chạy xong thì chạy import_scenarios.py để sync lên Supabase.
"""
import json, os, re, sys, time
sys.stdout.reconfigure(encoding="utf-8")
from dotenv import load_dotenv
import litellm
from tenacity import retry, wait_random_exponential, stop_after_attempt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(ROOT, ".env"))
litellm.api_key = os.environ["GEMINI_API_KEY"]

JSON_PATH = os.path.join(ROOT, "data/pipeline_results/edu_scenarios_prompted.json")
MARKER_RX = re.compile(r"Hệ quả tốt|HỆ QUẢ TỐT|Hệ quả xấu|HỆ QUẢ XẤU")

SYSTEM = (
    "Bạn là chuyên gia viết nội dung giáo dục. "
    "Trả lời JSON thuần túy, không markdown, không giải thích."
)

PROMPT_TPL = """\
Dưới đây là 1 lựa chọn hành động trong một tình huống đạo đức giáo dục Việt Nam.
Hãy tách nó thành 3 phần rõ ràng và trả về JSON:
{{
  "action":  "<mô tả hành động ngắn gọn, 1-2 câu, không lặp hệ quả>",
  "good":    "<hệ quả tích cực, 1-2 câu>",
  "bad":     "<hệ quả tiêu cực, 1-2 câu>"
}}
Giữ nguyên văn phong tiếng Việt, không thêm thông tin mới.
Nếu văn bản đã ngầm chứa hệ quả tốt/xấu hãy làm rõ ra.

Lựa chọn hành động:
{option_text}
"""

@retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(4))
def rewrite_option(text: str) -> dict:
    resp = litellm.completion(
        model="gemini/gemini-2.5-flash",
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": PROMPT_TPL.format(option_text=text)},
        ],
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    return json.loads(raw)

def format_option(parts: dict) -> str:
    return f"{parts['action']} Hệ quả tốt: {parts['good']} Hệ quả xấu: {parts['bad']}"

def needs_fix(scenario: dict) -> bool:
    opts = [scenario.get("option_1",""), scenario.get("option_2",""), scenario.get("option_3","")]
    return not any(MARKER_RX.search(o) for o in opts if o)

def main():
    with open(JSON_PATH, encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data["scenarios"]
    to_fix = [s for s in scenarios if needs_fix(s)]
    print(f"[INFO] {len(to_fix)} scenarios cần fix")

    for i, s in enumerate(to_fix, 1):
        print(f"[{i}/{len(to_fix)}] id={s['id']} — {s['title'][:40]}")
        for key in ("option_1", "option_2", "option_3"):
            text = s.get(key, "").strip()
            if not text:
                continue
            try:
                parts = rewrite_option(text)
                s[key] = format_option(parts)
                print(f"  ✓ {key}")
            except Exception as e:
                print(f"  ✗ {key}: {e}")
        time.sleep(0.5)

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n[OK] Đã ghi lại {JSON_PATH}")
    print("[NEXT] Chạy: python utils/import_scenarios.py để sync lên Supabase")

if __name__ == "__main__":
    main()
