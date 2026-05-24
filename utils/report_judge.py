"""
Module 4 Report Generator
=========================
Input  : data/pipeline_results/06_judge_labels.json
Output : data/pipeline_results/06_judge_report.txt
"""

import json, sys
from collections import defaultdict
from pathlib import Path

if sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

INPUT_PATH  = Path("data/pipeline_results/06_judge_labels.json")
OUTPUT_PATH = Path("data/pipeline_results/06_judge_report.txt")


def pct(n, total):
    return f"{n/total*100:.1f}%" if total else "N/A"


def avg(values):
    vals = [v for v in values if v is not None]
    return f"{sum(vals)/len(vals):.2f}" if vals else "N/A"


def load_records() -> list[dict]:
    with open(INPUT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("records", data) if isinstance(data, dict) else data


def build_report(records: list[dict]) -> str:
    lines = []
    w = lines.append

    labeled  = [r for r in records if r.get("ethical_label_vn") is not None]
    errors   = [r for r in records if r.get("ethical_label_vn") is None]
    total    = len(records)
    n_lab    = len(labeled)

    w("MODULE 4 REPORT — JUDGE LABELER")
    w("=" * 60)
    w(f"Total records          : {total}")
    w(f"Labeled OK             : {n_lab} ({pct(n_lab, total)})")
    w(f"Judge errors           : {len(errors)} ({pct(len(errors), total)})")
    w("")

    # ── VN-ethical overview ────────────────────────────────────────
    vn_ok   = [r for r in labeled if r["ethical_label_vn"] == 1]
    vn_fail = [r for r in labeled if r["ethical_label_vn"] == 0]
    w("VN-ethical alignment:")
    w(f"  ethical_label_vn=1 (phù hợp VN) : {len(vn_ok)} ({pct(len(vn_ok), n_lab)})")
    w(f"  ethical_label_vn=0 (vi phạm VN)  : {len(vn_fail)} ({pct(len(vn_fail), n_lab)})")
    w("")

    # ── Severity distribution (chỉ khi ethical=0) ─────────────────
    sev_counts = defaultdict(int)
    for r in labeled:
        if r.get("severity_vn") is not None:
            sev_counts[r["severity_vn"]] += 1
    w("Severity (chỉ khi ethical_label_vn=0):")
    for s in sorted(sev_counts):
        w(f"  {s}: {sev_counts[s]}")
    w(f"  (total vi phạm có severity): {sum(sev_counts.values())}")
    w("")

    # ── Cultural alignment distribution ───────────────────────────
    cal_counts = defaultdict(int)
    for r in labeled:
        if r.get("cultural_alignment_vn") is not None:
            cal_counts[r["cultural_alignment_vn"]] += 1
    w("Cultural alignment (1=Western ↔ 5=VN):")
    for c in sorted(cal_counts):
        w(f"  {c}: {cal_counts[c]}")
    w(f"  avg: {avg([r.get('cultural_alignment_vn') for r in labeled])}")
    w("")

    # ── EU aligned overview ────────────────────────────────────────
    eu_yes = [r for r in labeled if r.get("eu_aligned") == 1]
    w("EU bias signal alignment:")
    w(f"  eu_aligned=1 : {len(eu_yes)} ({pct(len(eu_yes), n_lab)})")
    w(f"  eu_aligned=0 : {n_lab - len(eu_yes)} ({pct(n_lab - len(eu_yes), n_lab)})")
    w("")

    # ── By model ──────────────────────────────────────────────────
    by_model: dict[str, list] = defaultdict(list)
    for r in labeled:
        by_model[r.get("model_name", r.get("model", "?"))].append(r)

    w("By model:")
    header = f"  {'Model':<24} {'n':>5}  {'VN=1':>7}  {'EU=1':>7}  {'avgCAL':>7}  {'shift_mild':>10}  {'shift_strong':>12}"
    w(header)
    w("  " + "-" * (len(header) - 2))
    for model in sorted(by_model):
        rs = by_model[model]
        n  = len(rs)
        vn1 = sum(1 for r in rs if r["ethical_label_vn"] == 1)
        eu1 = sum(1 for r in rs if r.get("eu_aligned") == 1)
        mild_rs  = [r for r in rs if r.get("bias_shift_mild")  is not None]
        strong_rs = [r for r in rs if r.get("bias_shift_strong") is not None]
        mild_t   = sum(1 for r in mild_rs  if r["bias_shift_mild"])
        strong_t = sum(1 for r in strong_rs if r["bias_shift_strong"])
        cal_avg  = avg([r.get("cultural_alignment_vn") for r in rs])
        w(f"  {model:<24} {n:>5}  {pct(vn1, n):>7}  {pct(eu1, n):>7}  {cal_avg:>7}"
          f"  {pct(mild_t, len(mild_rs)):>10}  {pct(strong_t, len(strong_rs)):>12}")
    w("")

    # ── By prompt type ────────────────────────────────────────────
    by_pt: dict[str, list] = defaultdict(list)
    for r in labeled:
        by_pt[r.get("prompt_type", "?")].append(r)

    w("By prompt type:")
    for pt in ["neutral", "bias", "adversarial"]:
        rs = by_pt.get(pt, [])
        n  = len(rs)
        if n == 0:
            continue
        vn1  = sum(1 for r in rs if r["ethical_label_vn"] == 1)
        eu1  = sum(1 for r in rs if r.get("eu_aligned") == 1)
        cal  = avg([r.get("cultural_alignment_vn") for r in rs])
        w(f"  {pt:<12} n={n:>5}  VN=1={pct(vn1,n):>6}  EU=1={pct(eu1,n):>6}  avgCAL={cal}")
    w("")

    # ── Bias shift by model × prompt type ─────────────────────────
    w("Bias shift (strong) by model:")
    for model in sorted(by_model):
        rs    = by_model[model]
        total_pairs = [r for r in rs if r.get("bias_shift_strong") is not None]
        shifted     = [r for r in total_pairs if r["bias_shift_strong"]]
        w(f"  {model:<24}  {len(shifted):>3} / {len(total_pairs):>3}  ({pct(len(shifted), len(total_pairs))})")
    w("")

    # ── By topic ──────────────────────────────────────────────────
    by_topic: dict[str, list] = defaultdict(list)
    for r in labeled:
        by_topic[r.get("topic", "?")].append(r)

    w("By topic (VN-ethical rate):")
    rows = []
    for topic, rs in by_topic.items():
        n   = len(rs)
        vn1 = sum(1 for r in rs if r["ethical_label_vn"] == 1)
        ss  = [r for r in rs if r.get("bias_shift_strong") is not None]
        st  = sum(1 for r in ss if r["bias_shift_strong"])
        rows.append((topic, n, vn1, pct(vn1, n), pct(st, len(ss))))
    for topic, n, vn1, vn_pct, shift_pct in sorted(rows, key=lambda x: -float(x[3].rstrip("%") if x[3] != "N/A" else 0)):
        w(f"  {topic:<38} n={n:>4}  VN=1={vn_pct:>6}  shift_strong={shift_pct}")
    w("")

    w("=" * 60)
    w(f"Output: {OUTPUT_PATH}")

    return "\n".join(lines)


def main():
    if not INPUT_PATH.exists():
        print(f"[ERR] Không tìm thấy {INPUT_PATH}")
        return
    records = load_records()
    report  = build_report(records)
    print(report)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[OK] Đã ghi → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
