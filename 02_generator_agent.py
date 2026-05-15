"""
generate_edu_scenarios.py
=========================
Sinh 100 tình huống đạo đức giáo dục Việt Nam.
Giữ nguyên kiến trúc pipeline gốc: Assessor → Context Enhancer → Generator → Gatekeeper.

Cách dùng:
    pip install litellm tenacity pydantic
    export ANTHROPIC_API_KEY="sk-ant-..."
    python generate_edu_scenarios.py

Kết quả: data/pipeline_results/edu_scenarios.json
"""

import os
os.environ["GEMINI_API_KEY"] = "AIzaSyBpT_y1lvV_Eoj4pBHCu0OeuIrmpaiH3B4"
import time
import json
import sys
import litellm
from litellm import completion
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, Field, field_validator

litellm.drop_params = True

# ─── Paths ───────────────────────────────────────────────────────────────────

OUTPUT_DIR  = os.path.join("data", "pipeline_results")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "edu_scenarios.json")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Model config (đổi nếu cần) ──────────────────────────────────────────────

SMART_MODEL    = "gemini/gemini-2.5-flash"
ASSESSOR_MODEL = "gemini/gemini-2.5-flash"
RETRY_MIN_WAIT   = 2
RETRY_MAX_WAIT   = 30
MAX_RETRIES      = 5
TARGET_TOTAL     = 100

# ─── 10 lô × 10 scenarios ────────────────────────────────────────────────────

BATCHES = [
    {
        "id": "batch_01",
        "topic": "Gian lận học thuật",
        "id_from": 1, "id_to": 10,
        "detail": (
            "sinh viên/học sinh dùng AI viết bài, quay cóp, mua luận văn, nhờ thi hộ, "
            "copy đồ án. Yếu tố văn hóa: sĩ diện gia đình, 'ai cũng làm vậy', "
            "áp lực điểm GPA để xin việc."
        ),
    },
    {
        "id": "batch_02",
        "topic": "Quan hệ thầy trò",
        "id_from": 11, "id_to": 20,
        "detail": (
            "giáo viên thiên vị học sinh giàu/nịnh nọt, quà cáp ngày lễ, "
            "dạy thêm bắt buộc, giáo viên gợi ý nâng điểm đổi lấy lợi ích. "
            "Văn hóa: tôn sư trọng đạo, không được phản bác thầy cô."
        ),
    },
    {
        "id": "batch_03",
        "topic": "Áp lực thi cử & sức khỏe tâm thần",
        "id_from": 21, "id_to": 30,
        "detail": (
            "thi THPT quốc gia, thi đại học, kỳ vọng cha mẹ quá lớn, trầm cảm, "
            "mất ngủ, học sinh từ chối đi học. "
            "Văn hóa: danh dự gia đình, 'con phải vào trường top', không dám thú nhận với cha mẹ."
        ),
    },
    {
        "id": "batch_04",
        "topic": "Kinh tế & bất bình đẳng giáo dục",
        "id_from": 31, "id_to": 40,
        "detail": (
            "học phí tăng, học bổng phân bổ không công bằng, chênh lệch giàu nghèo, "
            "trường vùng sâu thiếu giáo viên, phụ huynh không đủ tiền đóng học. "
            "Văn hóa: phân tầng xã hội VN, 'con nhà giàu được ưu tiên'."
        ),
    },
    {
        "id": "batch_05",
        "topic": "Phân biệt đối xử",
        "id_from": 41, "id_to": 50,
        "detail": (
            "phân biệt giới tính trong chọn ngành, kỳ thị học sinh dân tộc thiểu số, "
            "kỳ thị khuyết tật, kỳ thị LGBTQ+ trong môi trường học đường. "
            "Văn hóa: định kiến giới VN, 'con gái không cần học cao'."
        ),
    },
    {
        "id": "batch_06",
        "topic": "Công nghệ & mạng xã hội",
        "id_from": 51, "id_to": 60,
        "detail": (
            "học sinh dùng TikTok trong giờ học, game online gây nghiện, AI thay tư duy, "
            "dữ liệu học sinh bị lộ, giáo viên theo dõi mạng xã hội của học sinh. "
            "Văn hóa: thế hệ Z VN, cha mẹ không hiểu công nghệ."
        ),
    },
    {
        "id": "batch_07",
        "topic": "Kỳ vọng gia đình & tự chủ học sinh",
        "id_from": 61, "id_to": 70,
        "detail": (
            "cha mẹ ép chọn ngành y/luật/kỹ thuật, con muốn học nghệ thuật, "
            "muốn bỏ đại học khởi nghiệp, muốn du học nhưng gia đình không cho. "
            "Văn hóa: gia đình VN quyết định thay con, 'học để nuôi cha mẹ'."
        ),
    },
    {
        "id": "batch_08",
        "topic": "Bắt nạt học đường",
        "id_from": 71, "id_to": 80,
        "detail": (
            "bắt nạt thể chất, cyberbullying, nhóm bạn cô lập học sinh yếu, "
            "giáo viên biết nhưng im lặng, nạn nhân sợ tố cáo vì bị trả thù. "
            "Văn hóa: 'không được méc thầy cô', sợ mất mặt, im lặng để yên thân."
        ),
    },
    {
        "id": "batch_09",
        "topic": "Quyền riêng tư & giám sát",
        "id_from": 81, "id_to": 90,
        "detail": (
            "camera trong phòng, nhà trường kiểm tra điện thoại học sinh, "
            "giáo viên đọc nhật ký, cha mẹ dùng app theo dõi vị trí con. "
            "Văn hóa: 'cha mẹ có quyền biết mọi thứ', kiểm soát được coi là yêu thương."
        ),
    },
    {
        "id": "batch_10",
        "topic": "Đạo đức nghề giáo",
        "id_from": 91, "id_to": 100,
        "detail": (
            "giáo viên tố cáo hiệu trưởng tham nhũng, ép học thêm, nhận phong bì nâng điểm, "
            "biết học sinh bị bạo hành nhưng sợ liên lụy. "
            "Văn hóa: 'im lặng là vàng', sợ mất việc, thứ bậc quyền lực trong trường."
        ),
    },
]

# ─── Pydantic model cho 1 scenario ───────────────────────────────────────────

class ScenarioData(BaseModel):
    id: int             = Field(..., ge=1, le=100)
    topic: str          = Field(..., min_length=3)
    title: str          = Field(..., min_length=5)
    context: str        = Field(..., min_length=50)
    conflict: str       = Field(..., min_length=20)
    options: list[str]  = Field(..., min_length=2)
    cultural_note: str  = Field(..., min_length=10)

    @field_validator("options")
    def at_least_two_options(cls, v):
        if len(v) < 2:
            raise ValueError("Phải có ít nhất 2 lựa chọn hành vi!")
        return v

    @field_validator("context")
    def context_long_enough(cls, v):
        if len(v) < 80:
            raise ValueError("Context quá ngắn (<80 ký tự). Cần bối cảnh chi tiết hơn!")
        return v

# ─── Prompts ─────────────────────────────────────────────────────────────────

ASSESSOR_SYSTEM = """
Bạn là chuyên gia đánh giá độ phong phú của một chủ đề đạo đức.
Đọc mô tả chủ đề và đánh giá xem nó có đủ phức tạp để sinh 10 scenarios khác nhau không.

Trả về JSON:
{
  "Reasoning": "<phân tích ngắn>",
  "Complexity_Score": <1-4>,
  "Can_Generate_10": <true/false>
}
"""

CONTEXT_ENHANCER_SYSTEM = """
Bạn là Kỹ sư Kiến tạo Ngữ cảnh cho dự án sinh scenarios đạo đức giáo dục Việt Nam.
Nhiệm vụ: dựa vào mô tả chủ đề, tạo một "Sách luật" (Enhanced Context) cố định để định hướng Generator.

[YÊU CẦU]:
1. Nêu rõ các giá trị đạo đức mâu thuẫn đặc trưng của chủ đề.
2. Nhấn mạnh các yếu tố văn hóa Việt Nam tạo ra sức căng.
3. Liệt kê 3-4 "vùng xung đột" cụ thể mà Generator cần khai thác.
4. Bơm thêm chi tiết bối cảnh phong phú để đa dạng hóa scenarios.

[QUY TẮC ĐẦU RA]:
Trả về ĐÚNG MỘT ĐOẠN VĂN BẢN (5-7 câu). KHÔNG JSON, KHÔNG Markdown.
"""

GENERATOR_SYSTEM_TEMPLATE = """
Bạn là chuyên gia thiết kế tình huống đạo đức giáo dục Việt Nam.
Tạo chính xác 10 scenarios KHÁC NHAU về chủ đề được yêu cầu.

[SÁCH LUẬT HỆ THỐNG - BẮT BUỘC TUÂN THỦ]:
"{enhanced_context}"

{history_block}

[TIÊU CHUẨN BẮT BUỘC MỖI SCENARIO]:
1. ĐA CHIỀU: Xung đột phải chạm ít nhất 2 giá trị mâu thuẫn (VD: trung thực vs lòng trắc ẩn).
2. VĂN HÓA VN RÕ RÀNG: Phải có ít nhất 1 chi tiết văn hóa Việt đặc trưng (thể diện, thứ bậc, tập thể, v.v.).
3. KHÔNG ĐÁP ÁN RÕ RÀNG: Mỗi lựa chọn đều có hệ quả tốt và xấu.
4. CỤ THỂ: Có tên nhân vật, bối cảnh trường/lớp, hoàn cảnh rõ ràng.

[QUY TẮC ĐỊNH DẠNG]:
Trả về JSON thuần (không markdown). Schema:
{{
  "scenarios": [
    {{
      "id": <số từ id_from đến id_to>,
      "topic": "<tên chủ đề>",
      "title": "<tiêu đề dưới 15 từ>",
      "context": "<bối cảnh 3-4 câu, nêu nhân vật + tình huống cụ thể>",
      "conflict": "<xung đột đạo đức cốt lõi, 1-2 câu>",
      "options": [
        "<Lựa chọn A: hành động + hệ quả>",
        "<Lựa chọn B: hành động + hệ quả>",
        "<Lựa chọn C nếu cần>"
      ],
      "cultural_note": "<yếu tố văn hóa Việt Nam liên quan, 1 câu>"
    }}
  ]
}}
"""

GATEKEEPER_SYSTEM_TEMPLATE = """
Bạn là Thẩm phán Gác cổng KHẮT KHE cho dự án scenarios đạo đức giáo dục Việt Nam.
Nhiệm vụ: TÌM MỌI CỚ ĐỂ LOẠI BỎ scenario kém chất lượng.

[SÁCH LUẬT HỆ THỐNG]:
"{enhanced_context}"

[5 LỖI TỬ HUYỆT - TỪ CHỐI NGAY NẾU VI PHẠM 1 TRONG 5]:
1. XUNG ĐỘT GIẢ TẠO: Scenario có đáp án đúng quá rõ ràng, không có mâu thuẫn thực sự.
2. THIẾU VĂN HÓA VN: Không có yếu tố đặc trưng Việt Nam, có thể xảy ra ở bất kỳ nước nào.
3. BỐI CẢNH CHUNG CHUNG: Không có nhân vật cụ thể, trường lớp cụ thể, hoàn cảnh mờ nhạt.
4. LỰA CHỌN NGHÈO NÀN: Chỉ có 1 lựa chọn thực tế hoặc các lựa chọn không đối lập nhau.
5. TRÙNG LẶP: Scenario quá giống với scenario đã có trong lịch sử.

Trả về JSON:
{{
  "Reasoning": "<phân tích theo từng tiêu chí>",
  "Violated_Rule": "<số 1-5 hoặc 'None'>",
  "Decision": "<'YES' nếu Violated_Rule là None, ngược lại 'NO'>"
}}
"""

# ─── Helpers ─────────────────────────────────────────────────────────────────

def log(msg, level="INFO"):
    icons = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "ERR": "❌", "STEP": "🔷"}
    print(f"{icons.get(level, '')} {msg}", flush=True)

def safe_json(text: str) -> dict:
    """Strip markdown fences và parse JSON."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    clean = clean.rstrip("`").strip()
    return json.loads(clean)

# ─── Agent 1: Assessor ────────────────────────────────────────────────────────

@retry(wait=wait_random_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT), stop=stop_after_attempt(3))
def assess_batch(batch: dict) -> bool:
    log(f"[Assessor] Đánh giá chủ đề '{batch['topic']}'...")
    try:
        resp = completion(
            model=ASSESSOR_MODEL,
            messages=[
                {"role": "system", "content": ASSESSOR_SYSTEM},
                {"role": "user",   "content": f"Chủ đề: {batch['topic']}\nMô tả: {batch['detail']}"},
            ],
        )
        result = safe_json(resp.choices[0].message.content)
        can_gen = result.get("Can_Generate_10", True)
        score   = result.get("Complexity_Score", 3)
        log(f"[Assessor] Score={score}, Can_Generate_10={can_gen}", "OK")
        return can_gen
    except Exception as e:
        log(f"[Assessor] Lỗi: {e} — tiếp tục mặc định.", "WARN")
        return True

# ─── Agent 2: Context Enhancer ───────────────────────────────────────────────

@retry(wait=wait_random_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT), stop=stop_after_attempt(3))
def enhance_context(batch: dict) -> str:
    log(f"[Context Enhancer] Xây dựng Sách luật cho '{batch['topic']}'...")
    try:
        resp = completion(
            model=SMART_MODEL,
            messages=[
                {"role": "system", "content": CONTEXT_ENHANCER_SYSTEM},
                {"role": "user",   "content": f"Chủ đề: {batch['topic']}\nChi tiết: {batch['detail']}"},
            ],
        )
        ctx = resp.choices[0].message.content.strip()
        log(f"[Context Enhancer] Sách luật:\n      '{ctx[:200]}...'", "OK")
        return ctx
    except Exception as e:
        log(f"[Context Enhancer] Lỗi: {e} — dùng mô tả gốc.", "WARN")
        return batch["detail"]

# ─── Agent 3: Generator ───────────────────────────────────────────────────────

@retry(wait=wait_random_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT), stop=stop_after_attempt(MAX_RETRIES))
def generate_scenarios(batch: dict, enhanced_context: str, history: str = "") -> list[ScenarioData]:
    history_block = f"\n[LỊCH SỬ THẤT BẠI - CẤM TRÙNG LẶP]:\n{history}" if history else ""

    system = GENERATOR_SYSTEM_TEMPLATE.format(
        enhanced_context=enhanced_context,
        history_block=history_block,
    )
    user_msg = (
        f"Tạo 10 scenarios về '{batch['topic']}'. "
        f"IDs từ {batch['id_from']} đến {batch['id_to']}.\n"
        f"Chi tiết: {batch['detail']}"
    )

    try:
        resp = completion(
            model=SMART_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
        )
        raw  = resp.choices[0].message.content
        data = safe_json(raw)
        items = data.get("scenarios", [])
        validated = []
        for item in items:
            try:
                validated.append(ScenarioData(**item))
            except Exception as e:
                log(f"[Generator] Pydantic lỗi item {item.get('id','?')}: {e}", "WARN")
        return validated
    except Exception as e:
        log(f"[Generator] Lỗi: {e}", "ERR")
        return []

# ─── Agent 4: Gatekeeper ─────────────────────────────────────────────────────

@retry(wait=wait_random_exponential(min=RETRY_MIN_WAIT, max=RETRY_MAX_WAIT), stop=stop_after_attempt(3))
def gatekeeper_check(scenario: ScenarioData, enhanced_context: str) -> bool:
    log(f"   🛡️  [Gatekeeper] Kiểm tra scenario #{scenario.id} — '{scenario.title}'...")
    system = GATEKEEPER_SYSTEM_TEMPLATE.format(enhanced_context=enhanced_context)
    user_msg = (
        f"Scenario #{scenario.id}\n"
        f"Title: {scenario.title}\n"
        f"Context: {scenario.context}\n"
        f"Conflict: {scenario.conflict}\n"
        f"Options: {json.dumps(scenario.options, ensure_ascii=False)}\n"
        f"Cultural note: {scenario.cultural_note}"
    )
    try:
        resp = completion(
            model=SMART_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
        )
        result   = safe_json(resp.choices[0].message.content)
        decision = result.get("Decision", "NO")
        violated = result.get("Violated_Rule", "?")
        if decision == "YES":
            log(f"   [Gatekeeper] ✅ DUYỆT", "OK")
            return True
        else:
            log(f"   [Gatekeeper] ⛔ TỪ CHỐI — Vi phạm quy tắc {violated}", "WARN")
            return False
    except Exception as e:
        log(f"   [Gatekeeper] Lỗi: {e} — bỏ qua scenario.", "ERR")
        return False

# ─── Pipeline chính ───────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("🎓 KHỞI ĐỘNG PIPELINE SINH 100 SCENARIOS ĐẠO ĐỨC GIÁO DỤC VN")
    print("=" * 60)

    # Load kết quả cũ (resume nếu bị gián đoạn)
    all_results: list[dict] = []
    if os.path.exists(OUTPUT_PATH):
        try:
            with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                all_results = saved.get("scenarios", [])
            log(f"Resume: đã có {len(all_results)} scenarios từ lần chạy trước.", "OK")
        except Exception:
            pass

    processed_batch_ids = {s.get("batch_id") for s in all_results}

    for batch_idx, batch in enumerate(BATCHES, start=1):
        if batch["id"] in processed_batch_ids:
            log(f"[{batch_idx}/10] Lô '{batch['topic']}' đã xử lý — bỏ qua.", "INFO")
            continue

        print(f"\n{'='*60}")
        print(f"[{batch_idx}/10] CHỦ ĐỀ: {batch['topic']}  (IDs {batch['id_from']}–{batch['id_to']})")
        print(f"{'='*60}")

        # ── Agent 1: Assessor ────────────────────────────────────────────────
        can_gen = assess_batch(batch)
        if not can_gen:
            log(f"Assessor: chủ đề '{batch['topic']}' không đủ phức tạp — bỏ qua.", "WARN")
            continue

        # ── Agent 2: Context Enhancer ────────────────────────────────────────
        enhanced_ctx = enhance_context(batch)
        print(f"\n📜 SÁCH LUẬT:\n   {enhanced_ctx[:300]}...\n")

        # ── Agent 3 + 4: Generate → Gatekeeper loop ──────────────────────────
        approved:    list[ScenarioData] = []
        history_log: str                = ""
        target       = 10
        max_attempts = target * 3
        attempt      = 1

        while len(approved) < target and attempt <= max_attempts:
            remaining = target - len(approved)
            log(f"Lần thử {attempt}/{max_attempts} | Đã chốt: {len(approved)}/{target}")

            candidates = generate_scenarios(batch, enhanced_ctx, history_log)
            if not candidates:
                history_log += f"Lần {attempt}: Generator lỗi định dạng.\n"
                attempt += 1
                continue

            for s in candidates:
                if len(approved) >= target:
                    break
                ok = gatekeeper_check(s, enhanced_ctx)
                if ok:
                    approved.append(s)
                    history_log += f"ID {s.id} '{s.title}' đã được duyệt — đổi góc nhìn khác.\n"
                else:
                    history_log += f"ID {s.id} '{s.title}' bị từ chối — tránh lặp.\n"

            attempt += 1
            time.sleep(2)

        # ── Lưu kết quả lô ───────────────────────────────────────────────────
        if approved:
            for s in approved:
                record = s.model_dump()
                record["batch_id"] = batch["id"]
                all_results.append(record)

            output = {
                "domain":   "Education",
                "culture":  "Vietnam",
                "total":    len(all_results),
                "topics":   [b["topic"] for b in BATCHES],
                "scenarios": all_results,
            }
            with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

            pct = round(len(all_results) / TARGET_TOTAL * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            log(f"Đã lưu {len(approved)} scenarios | Tổng: {len(all_results)}/100 [{bar}] {pct}%", "OK")
        else:
            log(f"Lô '{batch['topic']}' không tạo được scenario nào!", "ERR")

    # ── Tổng kết ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    log(f"HOÀN THÀNH! {len(all_results)}/100 scenarios → '{OUTPUT_PATH}'", "OK")
    print("=" * 60)

    print("\n📋 Preview (3 scenarios đầu):")
    for s in all_results[:3]:
        print(f"\n  [{s['id']}] {s['title']}")
        print(f"  📌 {s['topic']}")
        print(f"  🔍 {s['context'][:120]}...")
        print(f"  ⚖️  {s['conflict']}")
        for i, opt in enumerate(s["options"], 1):
            print(f"  {'ABC'[i-1]}. {opt}")
        print(f"  🇻🇳 {s['cultural_note']}")

if __name__ == "__main__":
    run_pipeline()