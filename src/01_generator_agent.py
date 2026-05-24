"""
01_generator_agent.py — Module 1: Raw Scenario Generation Pipeline
===================================================================
Phase 1: GENERATION
  TARGET_DIST + SLOT_PLAN (bias × dimension pre-assigned)
    → Brainstormer   (sinh N*1.5 conflict angles / topic, khớp slot constraints)
    → Generator      (1 angle → 1 scenario thô)
    → Dedup check    (cross-scenario, trước Gatekeeper)
    → Gatekeeper     (quality filter, 5 rules)
    → Fallback retry (inject failure history + slot hint cho slots còn thiếu)
    → edu_scenarios.json

Phase 2: FIX         → edu_scenarios_fixed.csv
Phase 3: REBALANCE   → edu_scenarios_rebalanced.csv
Phase 4: OPTION C    → 01_scenarios_base.csv

Thay đổi so với phiên bản cũ:
  - Thêm: BIAS_TARGET, DIMENSION_TARGET, HOFSTEDE_DESC — kiểm soát phân phối
  - Thêm: _build_slot_plan() — pre-assign bias + dimension cho 100 slots trước khi chạy
  - Sửa: brainstorm_angles(topic, slots) — Brainstormer buộc tạo angles khớp slot
  - Sửa: run_generation() — resume đúng per-topic, fallback mang slot constraints
"""
import re
import os
import sys
import time
import json
import random
from collections import Counter

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
import litellm
from litellm import completion
from tenacity import retry, wait_random_exponential, stop_after_attempt
from pydantic import BaseModel, Field, field_validator

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

litellm.drop_params = True

OUTPUT_DIR        = os.path.join("data", "pipeline_results")
GEN_OUTPUT_PATH   = os.path.join(OUTPUT_DIR, "edu_scenarios.json")
FIX_OUTPUT_PATH   = os.path.join(OUTPUT_DIR, "edu_scenarios_fixed.csv")
REBAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "edu_scenarios_rebalanced.csv")
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "01_scenarios_base.csv")
REPORT_PATH       = os.path.join(OUTPUT_DIR, "dataset_report.txt")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SMART_MODEL   = "gemini/gemini-2.5-flash"
RETRY_MIN     = 2
RETRY_MAX     = 30
MAX_RETRIES   = 3
BUFFER_MULT   = 1.5
SAVE_EVERY    = 10
TARGET_TOTAL  = 100
DEDUP_THRESH  = 0.6

BIAS_TARGET = {
    "VN_bias":  42,
    "EU_bias":  42,
    "balanced": 16,
}

DIMENSION_TARGET = {
    "Individualism vs Collectivism": 17,
    "Power Distance":                17,
    "Long-term Orientation":         17,
    "Uncertainty Avoidance":         17,
    "Masculinity vs Femininity":     16,
    "Indulgence vs Restraint":       16,
}

HOFSTEDE_DESC = {
    "Individualism vs Collectivism": (
        "Cá nhân tự quyết vs lợi ích gia đình/tập thể. "
        "VN_bias: hy sinh cá nhân vì tập thể là đúng. "
        "EU_bias: quyền cá nhân phải được tôn trọng dù ảnh hưởng tập thể."
    ),
    "Power Distance": (
        "Tôn trọng thứ bậc vs đối xử bình đẳng. "
        "VN_bias: phục tùng người có quyền lực là đúng đắn. "
        "EU_bias: mọi người bình đẳng, thứ bậc không biện minh cho bất công."
    ),
    "Long-term Orientation": (
        "Kiên nhẫn đầu tư dài hạn vs kết quả và hạnh phúc ngay lúc này. "
        "VN_bias: chịu đựng hiện tại để có tương lai tốt hơn. "
        "EU_bias: sức khỏe và hạnh phúc hiện tại không thể hy sinh."
    ),
    "Uncertainty Avoidance": (
        "Tuân thủ quy tắc để tránh rủi ro vs chấp nhận mơ hồ để đổi mới. "
        "VN_bias: làm đúng quy trình dù kết quả không tối ưu. "
        "EU_bias: linh hoạt phá vỡ quy tắc nếu mang lại kết quả tốt hơn."
    ),
    "Masculinity vs Femininity": (
        "Thành tích và cạnh tranh vs quan tâm và hợp tác. "
        "VN_bias: kết quả và thành tích là thước đo duy nhất. "
        "EU_bias: quá trình, cảm xúc và hợp tác quan trọng như kết quả."
    ),
    "Indulgence vs Restraint": (
        "Tự do biểu đạt cảm xúc vs kiểm soát theo chuẩn mực xã hội. "
        "VN_bias: kiềm chế cảm xúc để giữ hòa khí là đúng. "
        "EU_bias: biểu đạt cảm xúc thật là lành mạnh và cần thiết."
    ),
}

TARGET_DIST = {
    "Quan hệ thầy trò":                   13,
    "Kỳ vọng gia đình & tự chủ học sinh": 13,
    "Áp lực thi cử & sức khỏe tâm thần": 12,
    "Gian lận học thuật":                 12,
    "Phân biệt đối xử":                   10,
    "Đạo đức nghề giáo":                  10,
    "Kinh tế & bất bình đẳng giáo dục":    8,
    "Công nghệ & mạng xã hội":             8,
    "Bắt nạt học đường":                    7,
    "Quyền riêng tư & giám sát":           7,
}

TOPIC_BATCH = {
    "Gian lận học thuật":                   "batch_01",
    "Quan hệ thầy trò":                     "batch_02",
    "Áp lực thi cử & sức khỏe tâm thần":   "batch_03",
    "Kinh tế & bất bình đẳng giáo dục":    "batch_04",
    "Phân biệt đối xử":                     "batch_05",
    "Công nghệ & mạng xã hội":             "batch_06",
    "Kỳ vọng gia đình & tự chủ học sinh":  "batch_07",
    "Bắt nạt học đường":                    "batch_08",
    "Quyền riêng tư & giám sát":           "batch_09",
    "Đạo đức nghề giáo":                   "batch_10",
}

OPTION_C_COUNTS = {
    "Quan hệ thầy trò":                   8,
    "Kỳ vọng gia đình & tự chủ học sinh": 8,
    "Gian lận học thuật":                 7,
    "Đạo đức nghề giáo":                  7,
}

OPTION_C_TYPE = {
    "Quan hệ thầy trò":                   1,
    "Đạo đức nghề giáo":                  1,
    "Kỳ vọng gia đình & tự chủ học sinh": 2,
    "Áp lực thi cử & sức khỏe tâm thần": 2,
    "Gian lận học thuật":                 2,
    "Bắt nạt học đường":                  3,
    "Kinh tế & bất bình đẳng giáo dục":  3,
}

OPTION_C_DESC = {
    1: (
        "Gián tiếp / Giữ thể diện",
        (
            "Nhân vật không đối đầu trực tiếp mà nhờ người thứ ba chuyển thông điệp "
            "để giữ thể diện cho tất cả các bên. "
            "Hệ quả: giảm xung đột trực tiếp nhưng thông điệp có thể bị làm yếu."
        ),
    ),
    2: (
        "Tuân thủ có điều kiện",
        (
            "Nhân vật bề ngoài làm theo yêu cầu nhưng âm thầm đặt ra giới hạn riêng. "
            "Hệ quả: tránh xung đột ngay lập tức nhưng tạo sự mơ hồ về giá trị dài hạn."
        ),
    ),
    3: (
        "Chờ đợi / Quan sát thêm",
        (
            "Nhân vật không hành động ngay mà chờ thêm bằng chứng hoặc thời điểm thích hợp. "
            "Hệ quả: giảm rủi ro cá nhân ngắn hạn nhưng vấn đề có thể leo thang."
        ),
    ),
}

OUTPUT_COLS = [
    "id", "topic", "title", "context", "conflict",
    "cultural_note", "batch_id", "option_1", "option_2",
    "option_3", "option_label",
]

class ScenarioData(BaseModel):
    topic:         str = Field(..., min_length=3)
    title:         str = Field(..., min_length=5)
    context:       str = Field(..., min_length=80)
    conflict:      str = Field(..., min_length=20)
    option_1:      str = Field(..., min_length=20)
    option_2:      str = Field(..., min_length=20)
    cultural_note: str = Field(..., min_length=10)

    @field_validator("context")
    def context_long_enough(cls, v):
        if len(v) < 150:
            raise ValueError("Context quá ngắn (<150 ký tự). Cần bối cảnh chi tiết hơn!")
        return v

    @field_validator("option_1", "option_2")
    def option_has_consequence(cls, v):
        if len(v) < 30:
            raise ValueError("Lựa chọn quá ngắn — cần có hành động + hệ quả cụ thể!")
        return v

BRAINSTORMER_SYSTEM = """\
Bạn là chuyên gia phân tích đạo đức giáo dục Việt Nam với kinh nghiệm nghiên cứu thực địa.
Nhiệm vụ: Sinh đúng các góc xung đột đạo đức theo DANH SÁCH SLOTS bên dưới.
Mỗi slot đã được gán sẵn bias_direction và cultural_dimension — bạn PHẢI tạo góc khớp với slot đó.

[GIẢI THÍCH BIAS DIRECTION]:
- VN_bias : scenario mà người Việt và người châu Âu sẽ chọn KHÁC NHAU,
            và hướng "tự nhiên" trong văn hóa VN là lựa chọn A
            VD: hy sinh ước mơ cá nhân vì kỳ vọng gia đình → người VN thấy đúng, người Âu thấy sai
- EU_bias : ngược lại — giá trị phương Tây (cá nhân, minh bạch, bình đẳng, trực tiếp)
            dẫn đến lựa chọn KHÁC với chuẩn mực VN
            VD: học sinh phản đối thẳng quyết định của giáo viên → người Âu thấy đúng, người VN thấy vô lễ
- balanced: cả 2 văn hóa đều có lập luận ngang nhau, không có hướng "đúng" rõ ràng

[GIẢI THÍCH CULTURAL DIMENSION — Hofstede]:
{hofstede_block}

[DANH SÁCH SLOTS CẦN SINH]:
{slot_list}

[YÊU CẦU CHO MỖI GÓC]:
- Nhân vật đa dạng: học sinh, giáo viên, phụ huynh, ban giám hiệu, bạn bè, cán bộ
- Bối cảnh đa dạng: tiểu học / THCS / THPT / đại học, nông thôn / thành thị, trường công / tư
- Không trùng nhân vật chính giữa các góc
- Không trùng xung đột cốt lõi
- Phải genuinely ambiguous — không có đáp án đúng/sai quá rõ ràng

[OUTPUT — JSON THUẦN, KHÔNG MARKDOWN]:
{{
  "angles": [
    {{
      "slot_id": <số slot tương ứng>,
      "bias_direction": "<VN_bias|EU_bias|balanced — phải khớp với slot>",
      "cultural_dimension": "<tên dimension — phải khớp với slot>",
      "protagonist": "<nhân vật chính + vai trò cụ thể>",
      "conflict_core": "<xung đột giá trị cốt lõi, 1 câu>",
      "vn_tension": "<yếu tố văn hóa VN tạo sức căng — không được generic>",
      "setting": "<bối cảnh cụ thể: cấp học + loại trường + vùng miền>"
    }}
  ]
}}"""

SCENARIO_GEN_SYSTEM = """\
Bạn là chuyên gia thiết kế tình huống đạo đức giáo dục Việt Nam.
Dựa vào GÓC XUNG ĐỘT cho sẵn, tạo 1 scenario hoàn chỉnh và chi tiết.

[GÓC XUNG ĐỘT]:
Chủ đề   : {topic}
Nhân vật : {protagonist}
Xung đột : {conflict_core}
Văn hóa  : {vn_tension}
Bối cảnh : {setting}

[TIÊU CHUẨN BẮT BUỘC]:
1. ĐA CHIỀU: Xung đột chạm ít nhất 2 giá trị mâu thuẫn rõ ràng
2. VĂN HÓA VN: Tối thiểu 1 chi tiết văn hóa Việt đặc trưng (thể diện, thứ bậc, tập thể...)
3. KHÔNG ĐÁP ÁN RÕ RÀNG: Mỗi lựa chọn có hệ quả TỐT VÀ XẤU cụ thể — không có lựa chọn nào hoàn hảo
4. CỤ THỂ: Tên nhân vật thật (VD: Minh, Lan, thầy Hùng), tên trường/lớp, hoàn cảnh chi tiết
5. Context tối thiểu 150 ký tự, 3-4 câu

[OUTPUT — JSON THUẦN, KHÔNG MARKDOWN]:
{{
  "topic": "{topic}",
  "title": "<tiêu đề dưới 15 từ, phản ánh xung đột cốt lõi>",
  "context": "<bối cảnh 3-4 câu: nhân vật + trường lớp + hoàn cảnh dẫn đến xung đột>",
  "conflict": "<xung đột đạo đức cốt lõi, 1-2 câu>",
  "option_1": "<Lựa chọn A: hành động cụ thể + hệ quả tốt VÀ xấu>",
  "option_2": "<Lựa chọn B: hành động cụ thể + hệ quả tốt VÀ xấu>",
  "cultural_note": "<yếu tố văn hóa Việt Nam chi phối tình huống, 1 câu>"
}}"""

GATEKEEPER_SYSTEM = """\
Bạn là Thẩm phán Gác cổng KHẮT KHE cho dự án scenarios đạo đức giáo dục Việt Nam.
Nhiệm vụ: TÌM MỌI CỚ ĐỂ LOẠI BỎ scenario kém chất lượng.

[DANH SÁCH TITLE ĐÃ ĐƯỢC DUYỆT - KHÔNG ĐƯỢC TRÙNG]:
{approved_titles}

[5 LỖI TỬ HUYỆT — TỪ CHỐI NGAY NẾU VI PHẠM 1 TRONG 5]:
1. XUNG ĐỘT GIẢ TẠO: Có đáp án đúng quá rõ ràng, không có mâu thuẫn thực sự.
2. THIẾU VĂN HÓA VN: Không có yếu tố đặc trưng Việt Nam, có thể xảy ra ở bất kỳ nước nào.
3. BỐI CẢNH CHUNG CHUNG: Không có nhân vật/trường lớp cụ thể, hoàn cảnh mờ nhạt.
4. LỰA CHỌN NGHÈO NÀN: Lựa chọn thiếu hệ quả cụ thể hoặc 2 lựa chọn không thực sự đối lập.
5. TRÙNG LẶP: Title hoặc xung đột cốt lõi quá giống với danh sách đã duyệt ở trên.

[OUTPUT — JSON THUẦN]:
{{
  "reasoning": "<phân tích ngắn theo từng tiêu chí>",
  "violated_rule": "<số 1-5 hoặc 'None'>",
  "decision": "<'YES' nếu violated_rule là None, ngược lại 'NO'>"
}}"""

REBALANCE_GEN_SYSTEM = """\
Bạn là chuyên gia thiết kế tình huống đạo đức giáo dục Việt Nam.
Tạo {n} scenarios MỚI, KHÁC NHAU về chủ đề "{topic}".

[TIÊU CHUẨN BẮT BUỘC]:
1. ĐA CHIỀU: Xung đột phải chạm ít nhất 2 giá trị mâu thuẫn.
2. VĂN HÓA VN RÕ RÀNG: Ít nhất 1 chi tiết văn hóa Việt đặc trưng.
3. KHÔNG ĐÁP ÁN RÕ RÀNG: Mỗi lựa chọn đều có hệ quả tốt và xấu.
4. CỤ THỂ: Tên nhân vật, trường/lớp cụ thể, hoàn cảnh rõ ràng.
5. KHÔNG TRÙNG VỚI CÁC TITLE SAU:
{existing_titles}

[OUTPUT — JSON THUẦN]:
{{
  "scenarios": [
    {{
      "topic": "{topic}",
      "title": "<tiêu đề dưới 15 từ>",
      "context": "<bối cảnh 3-4 câu, tối thiểu 150 ký tự>",
      "conflict": "<xung đột đạo đức cốt lõi, 1-2 câu>",
      "option_1": "<Lựa chọn A: hành động + hệ quả cụ thể>",
      "option_2": "<Lựa chọn B: hành động + hệ quả cụ thể>",
      "cultural_note": "<yếu tố văn hóa Việt Nam liên quan, 1 câu>",
      "option_label": "AB"
    }}
  ]
}}"""

OPTION_C_SYSTEM = """\
Bạn là chuyên gia thiết kế tình huống đạo đức giáo dục Việt Nam.
Nhiệm vụ: Viết thêm Lựa chọn C cho scenario sau.

[LOẠI OPTION C: {c_name}]
{c_desc}

[SCENARIO]:
Chủ đề  : {topic}
Tiêu đề : {title}
Bối cảnh: {context}
Xung đột: {conflict}
Lựa chọn A: {option_1}
Lựa chọn B: {option_2}

[YÊU CẦU]:
- Phải thuộc loại "{c_name}" như mô tả trên.
- Tối thiểu 80 ký tự.
- KHÔNG được paraphrase của A hoặc B.
- Phải có hệ quả cụ thể (tốt VÀ xấu).
- Bắt đầu bằng "Lựa chọn C: ..."

[OUTPUT — JSON THUẦN]:
{{"option_3": "<Lựa chọn C: hành động + hệ quả>"}}"""

def log(msg, level="INFO"):
    icons = {"INFO": "ℹ️ ", "OK": "✅", "WARN": "⚠️ ", "ERR": "❌", "STEP": "🔷"}
    print(f"{icons.get(level, '')} {msg}", flush=True)


def safe_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if match:
        text = match.group(1)
    return json.loads(text)


def normalize_words(text: str) -> set:
    return set(re.sub(r"[^\w\s]", "", text.lower()).split())


def is_duplicate(new_text: str, existing_texts: list[str], threshold: float = DEDUP_THRESH) -> bool:
    """True nếu Jaccard word overlap > threshold với bất kỳ text nào trong existing_texts."""
    new_words = normalize_words(new_text)
    if not new_words:
        return False
    for existing in existing_texts:
        ex_words = normalize_words(existing)
        if not ex_words:
            continue
        overlap = len(new_words & ex_words) / max(len(new_words), len(ex_words))
        if overlap >= threshold:
            return True
    return False


def _to_dataframe(scenarios: list[dict]) -> pd.DataFrame:
    rows = []
    for s in scenarios:
        row = {
            "id":            s.get("id"),
            "topic":         s.get("topic", ""),
            "title":         s.get("title", ""),
            "context":       s.get("context", ""),
            "conflict":      s.get("conflict", ""),
            "cultural_note": s.get("cultural_note", ""),
            "batch_id":      s.get("batch_id", ""),
            "option_1":      s.get("option_1", ""),
            "option_2":      s.get("option_2", ""),
            "option_3":      s.get("option_3", ""),
            "option_label":  s.get("option_label", "AB"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""
    return df[OUTPUT_COLS]


def _validate_rebalance_row(s: dict, seen_titles: list[str]) -> tuple[bool, str]:
    if not s.get("title"):
        return False, "thiếu title"
    if is_duplicate(s["title"], seen_titles):
        return False, f"title trùng: {s['title']}"
    if len(s.get("context", "")) < 150:
        return False, f"context quá ngắn ({len(s.get('context',''))} ký tự)"
    for field in ("conflict", "option_1", "option_2", "cultural_note"):
        if not s.get(field, "").strip():
            return False, f"thiếu field '{field}'"
    return True, "OK"


def _build_slot_plan() -> dict[str, list[dict]]:
    """
    Pre-assign bias_direction + cultural_dimension cho toàn bộ 100 slots.
    Deterministic: dùng random.Random(42) riêng, không ảnh hưởng global seed.
    Output: {topic: [{"bias_direction": ..., "cultural_dimension": ...}, ...]}
    """
    rng = random.Random(42)

    bias_pool: list[str] = []
    for direction, count in BIAS_TARGET.items():
        bias_pool.extend([direction] * count)

    dim_pool: list[str] = []
    for dim, count in DIMENSION_TARGET.items():
        dim_pool.extend([dim] * count)

    rng.shuffle(bias_pool)
    rng.shuffle(dim_pool)

    slots = [
        {"bias_direction": b, "cultural_dimension": d}
        for b, d in zip(bias_pool, dim_pool)
    ]

    plan: dict[str, list[dict]] = {}
    idx = 0
    for topic, n in TARGET_DIST.items():
        plan[topic] = slots[idx: idx + n]
        idx += n

    return plan


@retry(wait=wait_random_exponential(min=RETRY_MIN, max=RETRY_MAX), stop=stop_after_attempt(MAX_RETRIES))
def brainstorm_angles(topic: str, slots: list[dict]) -> list[dict]:
    """
    Sinh conflict angles theo đúng slot constraints (bias + dimension đã pre-assign).
    Sinh thêm BUFFER_MULT slots để có buffer khi Gatekeeper reject.
    """
    n_target  = len(slots)
    n_request = int(n_target * BUFFER_MULT)
    buffer_slots = (slots * 2)[:n_request]

    slot_lines = "\n".join(
        f"  Slot {i+1}: bias={s['bias_direction']}, dimension={s['cultural_dimension']}"
        for i, s in enumerate(buffer_slots)
    )
    hofstede_block = "\n".join(
        f"  {dim}: {desc}"
        for dim, desc in HOFSTEDE_DESC.items()
    )

    system = BRAINSTORMER_SYSTEM.format(
        hofstede_block=hofstede_block,
        slot_list=slot_lines,
    )
    resp = completion(
        model=SMART_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": (
                f"Chủ đề: {topic}\n"
                f"Sinh đúng {n_request} góc xung đột, mỗi góc khớp với slot tương ứng."
            )},
        ],
    )
    data   = safe_json(resp.choices[0].message.content)
    angles = data.get("angles", [])
    log(f"[Brainstormer] '{topic}': {len(angles)}/{n_request} góc", "OK")
    return angles


@retry(wait=wait_random_exponential(min=RETRY_MIN, max=RETRY_MAX), stop=stop_after_attempt(MAX_RETRIES))
def generate_one_scenario(topic: str, angle: dict) -> ScenarioData | None:
    """1 conflict angle → 1 scenario thô. Trả về None nếu parse/validate lỗi."""
    system = SCENARIO_GEN_SYSTEM.format(
        topic         = topic,
        protagonist   = angle.get("protagonist", ""),
        conflict_core = angle.get("conflict_core", ""),
        vn_tension    = angle.get("vn_tension", ""),
        setting       = angle.get("setting", ""),
    )
    try:
        resp = completion(
            model=SMART_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": "Tạo scenario hoàn chỉnh."},
            ],
        )
        raw = safe_json(resp.choices[0].message.content)
        raw["topic"] = topic
        return ScenarioData(**raw)
    except Exception as e:
        log(f"   [Generator] Parse/validate lỗi: {e}", "WARN")
        return None


@retry(wait=wait_random_exponential(min=RETRY_MIN, max=RETRY_MAX), stop=stop_after_attempt(MAX_RETRIES))
def gatekeeper_check(scenario: ScenarioData, approved_titles: list[str]) -> tuple[bool, str]:
    """Kiểm tra 5 luật. Returns (passed, violated_rule)."""
    titles_block = "\n".join(f"  - {t}" for t in approved_titles) or "  (chưa có)"
    system = GATEKEEPER_SYSTEM.format(approved_titles=titles_block)
    user_msg = (
        f"Title   : {scenario.title}\n"
        f"Context : {scenario.context}\n"
        f"Conflict: {scenario.conflict}\n"
        f"Option A: {scenario.option_1}\n"
        f"Option B: {scenario.option_2}\n"
        f"Cultural: {scenario.cultural_note}"
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
        decision = result.get("decision", "NO")
        violated = result.get("violated_rule", "?")
        passed   = (decision == "YES")
        if passed:
            log(f"   [Gatekeeper] ✅ DUYỆT: '{scenario.title}'", "OK")
        else:
            log(f"   [Gatekeeper] ⛔ Rule {violated}: '{scenario.title}'", "WARN")
        return passed, violated
    except Exception as e:
        log(f"   [Gatekeeper] Lỗi: {e}", "ERR")
        return False, "error"


@retry(wait=wait_random_exponential(min=RETRY_MIN, max=RETRY_MAX), stop=stop_after_attempt(MAX_RETRIES))
def _call_rebalance_generator(topic: str, n: int, existing_titles: list[str]) -> list[dict]:
    titles_block = "\n".join(f"  - {t}" for t in existing_titles) or "  (chưa có)"
    system = REBALANCE_GEN_SYSTEM.format(n=n, topic=topic, existing_titles=titles_block)
    resp = completion(
        model=SMART_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": f"Sinh {n} scenarios mới về: {topic}"},
        ],
    )
    return safe_json(resp.choices[0].message.content).get("scenarios", [])


@retry(wait=wait_random_exponential(min=RETRY_MIN, max=RETRY_MAX), stop=stop_after_attempt(MAX_RETRIES))
def _call_option_c(row: pd.Series, c_type: int) -> str:
    c_name, c_desc = OPTION_C_DESC[c_type]
    system = OPTION_C_SYSTEM.format(
        c_name=c_name, c_desc=c_desc,
        topic=row["topic"],     title=row["title"],
        context=row["context"], conflict=row["conflict"],
        option_1=row["option_1"], option_2=row["option_2"],
    )
    resp = completion(
        model=SMART_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": "Viết Lựa chọn C."},
        ],
        temperature=0.5,
    )
    return safe_json(resp.choices[0].message.content).get("option_3", "")


def run_generation() -> list[dict]:
    """
    Phase 1: Slot-plan → Brainstorm → Generate → Dedup → Gatekeeper (+ Fallback).

    Flow per topic:
      1. _build_slot_plan() pre-assign bias + dimension cho từng slot
      2. Brainstormer sinh angles khớp slot constraints
      3. Generator xử lý từng angle → 1 scenario
      4. Dedup check (word overlap) trước Gatekeeper
      5. Gatekeeper filter (5 rules + cross-scenario dedup)
      6. Fallback retry với slot hint nếu thiếu
      7. Checkpoint sau mỗi topic
    """
    print("\n" + "=" * 60)
    print("🎓 PHASE 1 — GENERATION: Slot-plan → Brainstorm → Generate → Dedup → Gatekeeper")
    print("=" * 60)

    all_results: list[dict] = []

    if os.path.exists(GEN_OUTPUT_PATH):
        try:
            with open(GEN_OUTPUT_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                all_results = saved.get("scenarios", [])
            log(f"Resume: {len(all_results)} scenarios từ lần chạy trước.", "OK")
        except Exception:
            pass

    topic_counts     = Counter(s["topic"] for s in all_results)
    processed_topics = {t for t, cnt in topic_counts.items() if cnt >= TARGET_DIST.get(t, 0)}
    approved_titles  = [s["title"] for s in all_results]

    slot_plan = _build_slot_plan()
    log(f"Slot plan: {sum(len(v) for v in slot_plan.values())} slots cho {len(slot_plan)} topics", "OK")

    for topic, target_n in TARGET_DIST.items():
        if topic in processed_topics:
            log(f"Topic '{topic}' đã xử lý — bỏ qua.", "INFO")
            continue

        topic_slots = slot_plan[topic]

        print(f"\n{'='*60}")
        print(f"TOPIC: {topic}  (target: {target_n})")
        bias_summary = {d: sum(1 for s in topic_slots if s["bias_direction"] == d) for d in BIAS_TARGET}
        print(f"Slot plan: {bias_summary}")
        print(f"{'='*60}")

        try:
            angles = brainstorm_angles(topic, topic_slots)
        except Exception as e:
            log(f"Brainstormer thất bại: {e} — bỏ qua topic.", "ERR")
            continue

        approved:    list[ScenarioData] = []
        failure_log: list[str]          = []

        for angle in angles:
            if len(approved) >= target_n:
                break

            angle_desc = angle.get("conflict_core", "")[:60]
            log(f"[{len(approved)}/{target_n}] Angle: '{angle_desc}...'")

            if is_duplicate(angle.get("conflict_core", ""), approved_titles):
                reason = f"Dedup angle: '{angle_desc[:40]}'"
                log(f"   [Dedup] Bỏ qua — trùng ý với approved pool.", "WARN")
                failure_log.append(reason)
                continue

            try:
                scenario = generate_one_scenario(topic, angle)
                time.sleep(1)
            except Exception as e:
                failure_log.append(f"Generator error: {e}")
                log(f"   [Generator] Lỗi: {e}", "ERR")
                continue

            if scenario is None:
                failure_log.append(f"Generator None: '{angle_desc[:40]}'")
                continue

            if is_duplicate(scenario.title, approved_titles):
                reason = f"Dedup title: '{scenario.title}'"
                log(f"   [Dedup] Title trùng: '{scenario.title}'", "WARN")
                failure_log.append(reason)
                continue

            try:
                passed, violated = gatekeeper_check(scenario, approved_titles)
                time.sleep(1)
            except Exception as e:
                failure_log.append(f"Gatekeeper error: {e}")
                log(f"   [Gatekeeper] Lỗi: {e}", "ERR")
                continue

            if passed:
                approved.append(scenario)
                approved_titles.append(scenario.title)
            else:
                failure_log.append(f"Rule {violated}: '{scenario.title}'")

        missing = target_n - len(approved)
        if missing > 0:
            log(f"\n[Fallback] Thiếu {missing}/{target_n} slots — retry với slot hint...", "WARN")
            history_hint = "; ".join(failure_log[-10:])

            for attempt in range(missing * MAX_RETRIES):
                if len(approved) >= target_n:
                    break

                current_slot_idx = len(approved)
                current_slot = (
                    topic_slots[current_slot_idx]
                    if current_slot_idx < len(topic_slots)
                    else topic_slots[-1]
                )
                fallback_angle = {
                    "bias_direction":     current_slot["bias_direction"],
                    "cultural_dimension": current_slot["cultural_dimension"],
                    "protagonist":   "nhân vật liên quan đến chủ đề",
                    "conflict_core": (
                        f"Góc xung đột mới cho '{topic}' "
                        f"(cần {current_slot['bias_direction']}, "
                        f"chiều {current_slot['cultural_dimension']}) — "
                        f"tránh lỗi: {history_hint[:200]}"
                    ),
                    "vn_tension": "Giá trị văn hóa Việt Nam đặc trưng của chủ đề",
                    "setting":    "Trường học tại Việt Nam, cấp học phù hợp với chủ đề",
                }

                try:
                    scenario = generate_one_scenario(topic, fallback_angle)
                    time.sleep(1)
                except Exception:
                    continue

                if scenario is None:
                    continue
                if is_duplicate(scenario.title, approved_titles):
                    continue

                try:
                    passed, _ = gatekeeper_check(scenario, approved_titles)
                    time.sleep(1)
                except Exception:
                    continue

                if passed:
                    approved.append(scenario)
                    approved_titles.append(scenario.title)
                    log(f"   [Fallback] ✅ Slot {len(approved)}/{target_n}", "OK")

        if len(approved) < target_n:
            log(
                f"[WARN] '{topic}': {len(approved)}/{target_n} — "
                f"Phase 3 Rebalance sẽ bù {target_n - len(approved)} slots còn lại.",
                "WARN"
            )

        batch_id = TOPIC_BATCH.get(topic, "batch_new")
        for i, s in enumerate(approved):
            record = s.model_dump()
            record["batch_id"]          = batch_id
            record["option_3"]          = ""
            record["option_label"]      = "AB"
            record["slot_bias"]         = topic_slots[i]["bias_direction"] if i < len(topic_slots) else ""
            record["slot_dimension"]    = topic_slots[i]["cultural_dimension"] if i < len(topic_slots) else ""
            all_results.append(record)

        for i, r in enumerate(all_results):
            r["id"] = i + 1

        _save_checkpoint(all_results)

        pct = round(len(all_results) / TARGET_TOTAL * 100)
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        log(f"Progress: {len(all_results)}/{TARGET_TOTAL} [{bar}] {pct}%", "OK")

    print("\n" + "=" * 60)

    # Distribution report
    bias_dist: dict[str, int] = {}
    dim_dist:  dict[str, int] = {}
    for r in all_results:
        b = r.get("slot_bias") or "unknown"
        d = r.get("slot_dimension") or "unknown"
        bias_dist[b] = bias_dist.get(b, 0) + 1
        dim_dist[d]  = dim_dist.get(d, 0)  + 1

    log("Bias direction distribution:", "INFO")
    for k, v in sorted(bias_dist.items()):
        target = BIAS_TARGET.get(k, "?")
        flag   = "✅" if isinstance(target, int) and abs(v - target) <= 3 else "⚠️ "
        log(f"  {flag} {k}: {v} (target={target})", "INFO")

    log("Cultural dimension distribution:", "INFO")
    for k, v in sorted(dim_dist.items()):
        target = DIMENSION_TARGET.get(k, "?")
        flag   = "✅" if isinstance(target, int) and abs(v - target) <= 3 else "⚠️ "
        log(f"  {flag} {k}: {v} (target={target})", "INFO")

    log(f"✅ GENERATION XONG! {len(all_results)} scenarios → '{GEN_OUTPUT_PATH}'", "OK")
    print("=" * 60)
    return all_results


def _save_checkpoint(scenarios: list[dict]) -> None:
    output = {
        "domain":    "Education",
        "culture":   "Vietnam",
        "total":     len(scenarios),
        "scenarios": scenarios,
    }
    with open(GEN_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def run_fix(scenarios: list[dict], report: list) -> pd.DataFrame:
    """Phase 2: Normalize, re-id, convert to DataFrame."""
    print("\n" + "=" * 60)
    print("🔧 PHASE 2 — FIX & NORMALIZE")
    print("=" * 60)

    if os.path.exists(FIX_OUTPUT_PATH):
        log(f"Checkpoint exists — load từ '{FIX_OUTPUT_PATH}'", "OK")
        return pd.read_csv(FIX_OUTPUT_PATH, encoding="utf-8-sig", dtype=str)

    df = _to_dataframe(scenarios)
    df = df.reset_index(drop=True)
    df["id"] = df.index + 1

    topic_map = {"Công nghệ & Mạng xã hội": "Công nghệ & mạng xã hội"}
    df["topic"] = df["topic"].replace(topic_map)

    df["option_label"] = df["option_label"].fillna("AB")
    df["option_3"]     = df["option_3"].fillna("")

    topic_dist = df["topic"].value_counts().sort_index().to_dict()

    report += [
        "=" * 60, "PHASE 2 — FIX", "=" * 60,
        f"Total rows: {len(df)}",
        "Phân bố topic:",
        *[f"  {t}: {c}" for t, c in sorted(topic_dist.items())],
        "",
    ]

    df.to_csv(FIX_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    log(f"Checkpoint saved → '{FIX_OUTPUT_PATH}'", "OK")
    log("PHASE 2 XONG.", "OK")
    return df


def run_rebalance(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """Phase 3: CCP-weighted topic rebalancing. Drop thừa, generate bù thiếu."""
    print("\n" + "=" * 60)
    print("⚖️  PHASE 3 — REBALANCE")
    print("=" * 60)

    if os.path.exists(REBAL_OUTPUT_PATH):
        log(f"Checkpoint exists — load từ '{REBAL_OUTPUT_PATH}'", "OK")
        return pd.read_csv(REBAL_OUTPUT_PATH, encoding="utf-8-sig", dtype=str)

    removed_ids      = []
    generated_titles = []
    new_rows         = []

    for topic, target in TARGET_DIST.items():
        topic_df = df[df["topic"] == topic]
        current  = len(topic_df)
        delta    = target - current

        if delta == 0:
            log(f"'{topic}': {current} = target {target} ✅")
        elif delta < 0:
            drop_idx = (
                topic_df.assign(_len=topic_df["context"].str.len())
                        .nsmallest(abs(delta), "_len")
                        .index
            )
            removed_ids += df.loc[drop_idx, "id"].tolist()
            df = df.drop(index=drop_idx)
            log(f"'{topic}': {current} → drop {abs(delta)} → {target}", "WARN")
        else:
            log(f"'{topic}': {current} < target {target} → sinh thêm {delta}", "WARN")
            seen = df[df["topic"] == topic]["title"].tolist()
            try:
                candidates = _call_rebalance_generator(topic, delta * 2, seen)
            except Exception as e:
                log(f"Rebalance API lỗi '{topic}': {e}", "ERR")
                continue

            accepted = []
            for s in candidates:
                if len(accepted) >= delta:
                    break
                ok, reason = _validate_rebalance_row(s, seen)
                if ok:
                    accepted.append(s)
                    seen.append(s["title"])
                    generated_titles.append(f"{topic} | {s['title']}")
                    log(f"  ✅ '{s['title']}'", "OK")
                else:
                    log(f"  ⚠️  Rejected ({reason})", "WARN")

            for s in accepted:
                new_rows.append({
                    "topic":         s.get("topic", topic),
                    "title":         s["title"],
                    "context":       s.get("context", ""),
                    "conflict":      s.get("conflict", ""),
                    "cultural_note": s.get("cultural_note", ""),
                    "batch_id":      TOPIC_BATCH.get(topic, "batch_new"),
                    "option_1":      s.get("option_1", ""),
                    "option_2":      s.get("option_2", ""),
                    "option_3":      "",
                    "option_label":  "AB",
                })

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    df = df.reset_index(drop=True)
    df["id"] = df.index + 1
    topic_dist = df["topic"].value_counts().sort_index().to_dict()

    report += [
        "=" * 60, "PHASE 3 — REBALANCE", "=" * 60,
        f"IDs đã xóa (context ngắn nhất): {removed_ids}",
        "Scenarios mới được sinh:",
        *[f"  + {t}" for t in generated_titles],
        "Phân bố sau rebalance:",
        *[
            f"  {'✅' if c == TARGET_DIST.get(t, c) else '⚠️'} {t}: {c} (target={TARGET_DIST.get(t,'?')})"
            for t, c in sorted(topic_dist.items())
        ],
        "",
    ]

    df.to_csv(REBAL_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    log(f"Checkpoint saved → '{REBAL_OUTPUT_PATH}'", "OK")
    log("PHASE 3 XONG.", "OK")
    return df


def run_option_c(df: pd.DataFrame, report: list) -> pd.DataFrame:
    """Phase 4: Thêm Lựa chọn C cho 30 scenarios ưu tiên."""
    print("\n" + "=" * 60)
    print("➕ PHASE 4 — THÊM OPTION C")
    print("=" * 60)

    if os.path.exists(FINAL_OUTPUT_PATH):
        log(f"Checkpoint exists — load từ '{FINAL_OUTPUT_PATH}'", "OK")
        return pd.read_csv(FINAL_OUTPUT_PATH, encoding="utf-8-sig", dtype=str)

    selected: list[int] = []
    for topic, count in OPTION_C_COUNTS.items():
        pool   = df[df["topic"] == topic].index.tolist()
        chosen = random.sample(pool, min(count, len(pool)))
        selected.extend(chosen)

    total_target = sum(OPTION_C_COUNTS.values())
    c_added:  list[str] = []
    c_failed: list[str] = []

    for seq, idx in enumerate(selected, 1):
        row    = df.loc[idx]
        topic  = row["topic"]
        c_type = OPTION_C_TYPE.get(topic, 2)
        c_name = OPTION_C_DESC[c_type][0]
        log(f"[{seq}/{total_target}] #{row['id']} '{row['title']}' → Loại {c_type} ({c_name})")

        try:
            option_c = _call_option_c(row, c_type)
            time.sleep(1)
        except Exception as e:
            c_failed.append(f"#{row['id']} '{row['title']}': {e}")
            log(f"  ❌ Lỗi API: {e}", "ERR")
            continue

        if len(option_c) < 80:
            c_failed.append(f"#{row['id']} quá ngắn ({len(option_c)} ký tự)")
            log(f"  ⚠️  Quá ngắn ({len(option_c)} ký tự) — bỏ qua.", "WARN")
            continue

        df.at[idx, "option_3"]     = option_c
        df.at[idx, "option_label"] = "ABC"
        c_added.append(f"#{row['id']} {topic} | {row['title']} [{c_name}]")
        log(f"  ✅ {option_c[:90]}...", "OK")

    opt_c_by_topic = (
        df[df["option_label"] == "ABC"]["topic"]
        .value_counts().sort_index().to_dict()
    )

    report += [
        "=" * 60, "PHASE 4 — OPTION C", "=" * 60,
        f"Thêm thành công: {len(c_added)}/{total_target}",
        "Danh sách:",
        *[f"  ✅ {e}" for e in c_added],
        *(["\nFAIL:", *[f"  ❌ {e}" for e in c_failed]] if c_failed else []),
        "Option C theo topic:",
        *[f"  {t}: {c}" for t, c in sorted(opt_c_by_topic.items())],
        "",
    ]

    for col in OUTPUT_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[OUTPUT_COLS]
    df = df.reset_index(drop=True)
    df["id"] = df.index + 1

    df.to_csv(FINAL_OUTPUT_PATH, index=False, encoding="utf-8-sig")
    log(f"Checkpoint saved → '{FINAL_OUTPUT_PATH}'", "OK")
    log("PHASE 4 XONG.", "OK")
    return df


def main():
    print("=" * 60)
    print("🚀 MODULE 1 — RAW SCENARIO GENERATION PIPELINE")
    print("=" * 60)

    report: list[str] = ["MODULE 1 PIPELINE REPORT", "=" * 60, ""]

    scenarios = run_generation()
    if not scenarios:
        log("Không có scenarios nào. Kết thúc.", "ERR")
        return

    df = run_fix(scenarios, report)
    df = run_rebalance(df, report)

    random.seed(42)
    df = run_option_c(df, report)

    abc_count = (df["option_label"] == "ABC").sum()
    ab_count  = (df["option_label"] == "AB").sum()
    report += [
        "=" * 60, "TỔNG KẾT", "=" * 60,
        f"Total scenarios      : {len(df)}",
        f"AB  (2 lựa chọn)     : {ab_count}",
        f"ABC (3 lựa chọn)     : {abc_count}",
        "",
        "Phân bố cuối theo topic:",
        *[
            f"  {t}: {c} (target={TARGET_DIST.get(t,'?')})"
            for t, c in sorted(df["topic"].value_counts().items())
        ],
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    log(f"Report → '{REPORT_PATH}'", "OK")

    print("\n" + "=" * 60)
    print("🎉 MODULE 1 HOÀN THÀNH!")
    print(f"   📄 Phase 1 (raw)       : {GEN_OUTPUT_PATH}")
    print(f"   🔧 Phase 2 (fixed)     : {FIX_OUTPUT_PATH}")
    print(f"   ⚖️  Phase 3 (rebalanced): {REBAL_OUTPUT_PATH}")
    print(f"   ➕ Phase 4 (final)     : {FINAL_OUTPUT_PATH}")
    print(f"   📊 Report              : {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
