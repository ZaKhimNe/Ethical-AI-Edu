# Ethical-AI-Edu

## Mục đích dự án

Tạo dataset đạo đức giáo dục Việt Nam gồm 100 scenarios, mỗi scenario là một tình huống đạo đức có 2–3 lựa chọn hành động và 4 nhãn phân tích. Dataset phục vụ huấn luyện / đánh giá mô hình AI hiểu ngữ cảnh văn hóa Việt Nam trong giáo dục.

---

## Cấu trúc file

```
Ethical-AI-Edu/
├── 02_generator_agent.py       ← Pipeline chính (CHẠY FILE NÀY)
├── 03_dataset_refiner.py       ← Script độc lập (giữ nguyên, không xóa)
├── edu_scenarios.csv           ← Dataset gốc 100 rows (root dir)
├── data_100scenarios.xlsx      ← Dữ liệu tham khảo ban đầu
├── data/
│   └── pipeline_results/       ← Tất cả output của pipeline
│       ├── edu_scenarios.json           (Phase 1 checkpoint)
│       ├── edu_scenarios_fixed.csv      (Phase 2 checkpoint)
│       ├── edu_scenarios_rebalanced.csv (Phase 3 checkpoint)
│       ├── edu_scenarios_final.csv      (Phase 4 output)
│       ├── edu_scenarios_labeled.json   (Phase 5 output — final)
│       └── dataset_report.txt           (báo cáo tổng hợp)
└── .env                        ← API key (không commit)
```

---

## Cách chạy

```bash
# 1. Cài dependencies
pip install litellm tenacity pydantic pandas python-dotenv

# 2. Tạo .env với Gemini API key
echo GEMINI_API_KEY=AIzaS... > .env

# 3. Chạy toàn bộ pipeline
python 02_generator_agent.py
```

Pipeline có checkpoint idempotent: nếu bị gián đoạn, chạy lại sẽ resume từ phase cuối đã hoàn thành.

---

## Pipeline 5 pha (`02_generator_agent.py`)

| Phase | Hàm | Input | Output | Checkpoint |
|-------|-----|-------|--------|------------|
| 1 Generation | `run_generation()` | BATCHES config | `list[dict]` | `edu_scenarios.json` |
| 2 Fix | `run_fix()` | `list[dict]` | DataFrame | `edu_scenarios_fixed.csv` |
| 3 Rebalance | `run_rebalance()` | DataFrame | DataFrame | `edu_scenarios_rebalanced.csv` |
| 4 Option C | `run_option_c()` | DataFrame | DataFrame | `edu_scenarios_final.csv` |
| 5 Labeling | `run_labeling()` | DataFrame | JSON | `edu_scenarios_labeled.json` |

### Agents sử dụng (Phase 1)
1. **Assessor** — đánh giá độ phù hợp của batch config
2. **Context Enhancer** — làm giàu ngữ cảnh văn hóa
3. **Generator** — sinh scenarios dạng JSON
4. **Gatekeeper** — kiểm tra chất lượng, từ chối nếu không đạt

### Agents bổ sung (Phase 3–5)
5. **Rebalance Generator** — sinh thêm scenarios cho topic thiếu
6. **Option C Generator** — thêm lựa chọn C cho 30 scenarios ưu tiên
7. **Labeler** — dán 4 nhãn: `eu_cultural_note`, `bias_direction`, `adversarial_strategy`, `cultural_dimension`

---

## Phân phối topic (TARGET_DIST)

| Topic | Target |
|-------|--------|
| Quan hệ thầy trò | 13 |
| Kỳ vọng gia đình & tự chủ học sinh | 13 |
| Áp lực thi cử & sức khỏe tâm thần | 12 |
| Gian lận học thuật | 12 |
| Phân biệt đối xử | 10 |
| Đạo đức nghề giáo | 10 |
| Kinh tế & bất bình đẳng giáo dục | 8 |
| Công nghệ & mạng xã hội | 8 |
| Bắt nạt học đường | 7 |
| Quyền riêng tư & giám sát | 7 |

---

## Quy tắc bảo mật

- **TUYỆT ĐỐI KHÔNG** hardcode API key trong source code
- API key phải được load từ `.env` qua `python-dotenv`
- `.env` phải có trong `.gitignore`
- Model: `gemini/gemini-2.5-flash` qua LiteLLM

---

## Kỹ thuật chính áp dụng

- **Gatekeeper Pattern** — quality gate với 5 rejection rules + failure history injection
- **Retry + Exponential Backoff** — `tenacity.wait_random_exponential` chống thundering herd
- **Pydantic Validation** — `ScenarioData` schema validate JSON output của LLM
- **Idempotent Resume** — mỗi phase check `os.path.exists()` checkpoint trước khi chạy
- **Bridge Pattern** — `_to_dataframe()` và `_reconstruct_options()` chuyển đổi format giữa phases
- **CCP-Weighted Distribution** — Phase 3 drop/generate để đạt TARGET_DIST chính xác
- **Safe JSON Parsing** — `safe_json()` strip markdown code blocks trước `json.loads()`
- **Graceful Degradation** — per-scenario `try/except + continue`, pipeline không crash

---

## Conventions

- Encoding: UTF-8 everywhere (`utf-8-sig` cho CSV, `utf-8` cho JSON)
- `random.seed(42)` trước Phase 4 để đảm bảo deterministic selection
- `03_dataset_refiner.py` là script độc lập, không thay đổi, đọc từ `edu_scenarios.csv`
- Log format: `[INFO]`, `[OK]`, `[WARN]`, `[ERROR]` prefix
