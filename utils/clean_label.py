"""
clean_labeled.py
────────────────
Làm sạch cột cultural_dimension trong 03_scenarios_labeled.csv.
Trích xuất đúng tên dimension, bỏ phần giải thích thừa.

Chạy: python clean_labeled.py
"""
import re
import pandas as pd

LABELED_PATH = "data/pipeline_results/03_scenarios_labeled.csv"

VALID_DIMS = [
    "Individualism vs Collectivism",
    "Power Distance",
    "Long-term Orientation",
    "Uncertainty Avoidance",
    "Masculinity vs Femininity",
    "Indulgence vs Restraint",
]

def extract_dimension(raw: str) -> str:
    """
    Trích xuất tên dimension từ chuỗi dài.
    VD: "Power Distance (PDI): Xung đột chính..." → "Power Distance"
    """
    raw = str(raw).strip()
    for dim in VALID_DIMS:
        if raw.startswith(dim):
            return dim
    for dim in VALID_DIMS:
        if dim.lower() in raw.lower():
            return dim
    return raw


def main():
    df = pd.read_csv(LABELED_PATH, encoding="utf-8-sig", dtype=str).fillna("")
    print(f"[INFO] Đọc {len(df)} rows")

    print("\nBefore clean — cultural_dimension distribution:")
    df["dim_short"] = df["cultural_dimension"].apply(
        lambda x: x.split("(")[0].strip().split(":")[0].strip().split(".")[0].strip()
    )
    print(df["dim_short"].value_counts().to_string())

    df["cultural_dimension"] = df["cultural_dimension"].apply(extract_dimension)

    print("\nAfter clean — cultural_dimension distribution:")
    print(df["cultural_dimension"].value_counts().to_string())

    unmatched = df[~df["cultural_dimension"].isin(VALID_DIMS)]
    if len(unmatched):
        print(f"\n[WARN] {len(unmatched)} rows không match dimension chuẩn:")
        for _, r in unmatched.iterrows():
            print(f"  #{r['id']} — {r['cultural_dimension'][:80]}")
    else:
        print("\n[OK] Tất cả rows đều match dimension chuẩn.")

    df = df.drop(columns=["dim_short"], errors="ignore")
    df.to_csv(LABELED_PATH, index=False, encoding="utf-8-sig")
    print(f"\n[OK] Ghi đè '{LABELED_PATH}'")


if __name__ == "__main__":
    main()