import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

# 1. טעינת נתיבים והגדרות
load_dotenv()
env_path = os.getenv("DATA_PATH")

if not env_path:
    print("ERROR: DATA_PATH not found in .env file.")
    exit()

base_path = Path(env_path)
input_csv = base_path / "indiana_merged_data.csv"
output_csv = base_path / "indiana_reports_with_summary.csv"

# 2. טעינת הטבלה
print(f"Loading data from: {input_csv}...")
try:
    df = pd.read_csv(input_csv)
except FileNotFoundError:
    print("ERROR: Input file not found. Make sure 'indiana_merged_data.csv' exists.")
    exit()


# 3. הפונקציה ליצירת הסיכום (הלב של המשימה)
def generate_clinical_summary(row):
    """
    Generates a structured clinical summary text based on the task requirements.
    Rules:
    - Use strict template headers.
    - Skip empty fields (no hallucinations).
    - Use cautious language for interpretation.
    """

    # שליפת הנתונים וניקוי בסיסי (הופך לטקסט, מסיר רווחים מיותרים)
    findings = str(row.get('findings', '')).strip()
    impression = str(row.get('impression', '')).strip()
    indication = str(row.get('indication', '')).strip()  # נשתמש בזה ל-Clinical Interpretation

    # רשימה שתחזיק את שורות הסיכום
    lines = ["Clinical Summary:"]

    # --- הוספת Imaging Findings ---
    # בודקים שהמידע קיים ושזה לא סתם המילה 'nan'
    if findings and findings.lower() != 'nan':
        lines.append("Imaging Findings:")
        lines.append(findings)

    # --- הוספת Radiological Impression ---
    if impression and impression.lower() != 'nan':
        lines.append("Radiological Impression:")
        lines.append(impression)

    # --- הוספת Clinical Interpretation (עם ניסוח זהיר) ---
    if indication and indication.lower() != 'nan':
        lines.append("Possible Clinical Interpretation:")

        # לוגיקה לניסוח זהיר:
        # אם הטקסט כבר מכיל מילות הסתייגות, נשאיר אותו כמו שהוא.
        # אחרת, נוסיף הקדמה שמבהירה שזה הקשר קליני ולא אבחנה סופית.
        lower_ind = indication.lower()
        safe_words = ['suspect', 'evaluate', 'history', 'pain', 'indication', 'check']

        # בדיקה האם יש מילת קישור בטוחה
        is_safe = any(word in lower_ind for word in safe_words)

        if is_safe:
            lines.append(indication)
        else:
            # אם כתוב סתם "Pneumonia", נהפוך ל-"Clinical context suggests: Pneumonia"
            lines.append(f"Clinical context suggests: {indication}")

    # אם לא נוסף שום מידע מעבר לכותרת הראשית - נחזיר ערך ריק (או שניתן להשאיר רק כותרת)
    if len(lines) == 1:
        return ""

    # איחוד השורות עם ירידת שורה ביניהן
    return "\n".join(lines)


# 4. הפעלת הפונקציה על כל הטבלה
print("Generating summaries for all patients...")
# שימוש ב-apply כדי להריץ את הפונקציה שורה-שורה
df['clinical_summary'] = df.apply(generate_clinical_summary, axis=1)

# 5. בדיקת איכות (QC) - הצגת דוגמה אקראית
print("\n" + "=" * 40)
print("SAMPLE RESULT (Random Patient):")
print("=" * 40)
sample = df[df['clinical_summary'] != ""].sample(1).iloc[0]
print(f"File: {sample.get('filename', 'Unknown')}")
print("-" * 20)
print(sample['clinical_summary'])
print("=" * 40)

# 6. שמירת הקובץ החדש
df.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\nSUCCESS: Saved updated data to: {output_csv}")