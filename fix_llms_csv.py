import pandas as pd

input_path = "How_I_use_LLMs_unfinished_100%.csv"
output_path = "How_I_use_LLMs_fixed.csv"
dropped_output_path = "How_I_use_LLMs_dropped_no_timing.csv"

df = pd.read_csv(input_path)

label_col = "boundary"

# התאמת שמות עמודות
if "start_s" not in df.columns and "onset" in df.columns:
    df = df.rename(columns={"onset": "start_s"})

if "end_s" not in df.columns and "offset" in df.columns:
    df = df.rename(columns={"offset": "end_s"})

# שמירת התגית המקורית לפני בינאריזציה
df["original_boundary"] = df[label_col]

# ניקוי תגיות
df[label_col] = df[label_col].fillna(0)

df[label_col] = df[label_col].replace({
    2: 0,
    3: 0,
    4: 1
})

df[label_col] = df[label_col].astype(int)

# זיהוי מילים בלי תזמון
missing_time = df["start_s"].isna() | df["end_s"].isna()

dropped_df = df[missing_time].copy()
kept_df = df[~missing_time].copy()

# שמירת השורות שנמחקו לבדיקה ידנית
dropped_df.to_csv(dropped_output_path, index=False)

# המרת זמנים למספרים
kept_df["start_s"] = kept_df["start_s"].astype(float)
kept_df["end_s"] = kept_df["end_s"].astype(float)

# שמירת הקובץ הנקי
kept_df.to_csv(output_path, index=False)

print(f"Saved fixed file to: {output_path}")
print(f"Saved dropped rows to: {dropped_output_path}")

print("\nSummary:")
print(f"Rows before cleaning: {len(df)}")
print(f"Rows kept: {len(kept_df)}")
print(f"Rows dropped due to missing timing: {len(dropped_df)}")

if len(dropped_df) > 0:
    print("\nDropped rows boundary distribution:")
    print(dropped_df[label_col].value_counts().sort_index())

print("\nKept rows boundary distribution:")
print(kept_df[label_col].value_counts().sort_index())