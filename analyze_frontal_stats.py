import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from collections import Counter

# 1. טעינת הגדרות
load_dotenv()
env_path = os.getenv("DATA_PATH")

if not env_path:
    print("ERROR: DATA_PATH not found in .env")
    exit()

base_path = Path(env_path)

# --- השינוי: טעינה ישירה של הקובץ המסונן שלך ---
csv_path = base_path / "indiana_frontal.csv"

# 2. טעינת הדאטה
print(f"Loading data from: {csv_path}...")
try:
    df_frontal = pd.read_csv(csv_path)  # טוענים ישר למשתנה df_frontal
except FileNotFoundError:
    print(f"ERROR: File not found at {csv_path}")
    print("Please make sure you ran 'filter_frontal_images.py' first.")
    exit()

print("-" * 40)
print(f"Total Frontal images loaded: {len(df_frontal)}")

# 3. אין צורך בסינון (כבר בוצע בקובץ הקודם)

# 4. ניתוח המחלות (מתוך עמודת Problems או Impression)
target_col = None
# בדיקה גמישה לשם העמודה
if 'problems' in df_frontal.columns:
    target_col = 'problems'
elif 'Problems' in df_frontal.columns:
    target_col = 'Problems'
elif 'impression' in df_frontal.columns:
    target_col = 'impression'

print(f"Analyzing column: '{target_col}'")
print("-" * 40)

all_conditions = []

# מעבר על הנתונים וניקוי
for text in df_frontal[target_col].astype(str).str.lower():
    if text == 'nan':
        continue

    # אם עובדים על Impression
    if target_col == 'impression':
        if 'normal' in text or 'no acute' in text or 'unremarkable' in text:
            all_conditions.append('normal')
            continue

    # פירוק הטקסט למחלות
    clean_text = text.replace(";", ",").replace("/", ",")
    diseases = clean_text.split(",")

    for d in diseases:
        d = d.strip()
        if len(d) > 2 and d != 'nan':
            if 'normal' in d:
                all_conditions.append('normal')
            else:
                all_conditions.append(d)

# 5. הצגת התוצאות
counts = Counter(all_conditions)

print("\nTOP CANDIDATES FOR PROTOTYPE (Frontal Only):")
print("=" * 40)
print(f"1. Normal (Healthy): {counts['normal']}")

print("\nTop Abnormal Conditions:")
i = 1
for disease, count in counts.most_common(20):
    if disease == 'normal':
        continue
    print(f"{i + 1}. {disease.title()}: {count}")
    i += 1

print("=" * 40)