"""
patch_slot_columns.py
─────────────────────
Dùng LLM classify từng scenario → ghi đè slot_bias + slot_dimension.
Ghi đè cả khi 2 cột đã tồn tại (từ lần chạy cứng trước).

Chạy: python patch_slot_columns.py
"""
import re
import os
import sys
import json
import time

import pandas as pd
import litellm
from litellm import completion
from tenacity import retry, wait_random_exponential, stop_after_attempt

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

litellm.drop_params = True

CSV_PATH      = "data/pipeline_results/01_scenarios_base.csv"
SMART_MODEL   = "gemini/gemini-2.5-flash"
SLEEP_BETWEEN = 1
SAVE_EVERY    = 10

CLASSIFIER_SYSTEM = """\
Bạn là chuyên gia phân tích cross-cultural bias trong giáo dục Việt Nam.
Đọc scenario sau và classify 2 nhãn.

[SCENARIO]:
Chủ đề  : {topic}
Tiêu đề : {title}
Bối cảnh: {context}
Xung đột: {conflict}
Option A : {option_1}
Option B : {option_2}

[NHÃN 1 — bias_direction]:
Người Việt và người châu Âu sẽ chọn KHÁC NHAU không?
- "VN_bias"  : hướng tự nhiên trong văn hóa VN khác với phương Tây rõ ràng
- "EU_bias"  : giá trị phương Tây (cá nhân, minh bạch, bình đẳng) dẫn đến chọn khác VN
- "balanced" : cả 2 văn hóa có lập luận ngang nhau

[NHÃN 2 — cultural_dimension]:
Chiều Hofstede NÀO là chính trong xung đột này? Chọn đúng 1:
- "Individualism vs Collectivism" — cá nhân tự quyết vs lợi ích tập thể/gia đình
- "Power Distance"                — tôn trọng thứ bậc vs đối xử bình đẳng
- "Long-term Orientation"         — đầu tư dài hạn vs hạnh phúc hiện tại
- "Uncertainty Avoidance"         — tuân thủ quy tắc vs chấp nhận rủi ro/đổi mới
- "Masculinity vs Femininity"     — thành tích/cạnh tranh vs quan tâm/hợp tác
- "Indulgence vs Restraint"       — tự do cảm xúc vs kiểm soát theo chuẩn mực xã hội

[RÀNG BUỘC PHÂN PHỐI — tuân thủ nghiêm]:
Dataset cần đạt:
  slot_bias      : VN_bias=42, EU_bias=42, balanced=16
  slot_dimension : mỗi chiều ~16-17 scenarios

Số đã gán tính đến thời điểm này:
  slot_bias hiện tại      : {bias_so_far}
  slot_dimension hiện tại : {dim_so_far}

→ Ưu tiên gán nhãn còn THIẾU so với target.
→ Nếu scenario genuinely ambiguous giữa 2 nhãn, chọn cái còn thiếu hơn.
→ Nếu scenario rõ ràng chỉ fit 1 nhãn thì vẫn gán đúng — không ép.

[OUTPUT — JSON THUẦN, KHÔNG MARKDOWN]:
{{
  "slot_bias":      "<VN_bias | EU_bias | balanced>",
  "slot_dimension": "<tên dimension chính xác như trên>"
}}"""


def safe_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    return json.loads(text)

@retry(wait=wait_random_exponential(min=2, max=30), stop=stop_after_attempt(3))
def classify_one(row: dict, bias_so_far: dict, dim_so_far: dict) -> tuple[str, str]:    
    bias_str = ", ".join(f"{k}={v}" for k, v in bias_so_far.items())
    dim_str  = ", ".join(f"{k}={v}" for k, v in dim_so_far.items())
    system = CLASSIFIER_SYSTEM.format(
        topic    = row.get("topic", ""),
        title    = row.get("title", ""),
        context  = row.get("context", ""),
        conflict = row.get("conflict", ""),
        option_1 = row.get("option_1", ""),
        option_2 = row.get("option_2", ""),
        bias_so_far = bias_str,
        dim_so_far  = dim_str,
    )
    resp = completion(
        model=SMART_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": "Classify scenario này."},
        ],
    )
    result = safe_json(resp.choices[0].message.content)
    return result["slot_bias"], result["slot_dimension"]


def main():
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig", dtype=str)
    df = df.fillna("")
    total = len(df)

    bias_counts = {"VN_bias": 0, "EU_bias": 0, "balanced": 0}
    dim_counts  = {
        "Individualism vs Collectivism": 0, "Power Distance": 0,
        "Long-term Orientation": 0, "Uncertainty Avoidance": 0,
        "Masculinity vs Femininity": 0, "Indulgence vs Restraint": 0,
    }   
    
    print(f"[INFO] Đọc {total} rows — bắt đầu classify (ghi đè slot cũ nếu có)")


    failed = []

    df["slot_bias"]      = ""
    df["slot_dimension"] = ""

    for seq, idx in enumerate(df.index, 1):
        row   = df.loc[idx].to_dict()
        title = row.get("title", "?")
        pct   = round(seq / total * 100)



        print(f"[{seq}/{total}] {pct}% — '{title}'", flush=True)

        try:
            slot_bias, slot_dim = classify_one(row, bias_counts, dim_counts)  
            df.at[idx, "slot_bias"]      = slot_bias
            df.at[idx, "slot_dimension"] = slot_dim
            bias_counts[slot_bias] = bias_counts.get(slot_bias, 0) + 1  
            dim_counts[slot_dim]   = dim_counts.get(slot_dim, 0) + 1    
            print(f"   [{slot_bias}] {slot_dim}", flush=True)
            time.sleep(SLEEP_BETWEEN)
        except Exception as e:
            failed.append(f"#{row.get('id','?')} '{title}': {e}")
            print(f"   [ERR] {e}", flush=True)
            continue

        if seq % SAVE_EVERY == 0:
            df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
            print(f"   [Checkpoint {seq}/{total}]", flush=True)
        
        
    df.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Ghi đè xong → '{CSV_PATH}'")

    if failed:
        print(f"\n[WARN] {len(failed)} rows lỗi:")
        for f in failed:
            print(f"  {f}")

    print("\nslot_bias distribution:")
    for k, v in df["slot_bias"].value_counts().items():
        print(f"  {k}: {v}")

    print("\nslot_dimension distribution:")
    for k, v in df["slot_dimension"].value_counts().items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()