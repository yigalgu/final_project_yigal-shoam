import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv
from collections import Counter

# 1. טעינת הנתיב מהקובץ .env
load_dotenv()
env_path = os.getenv("DATA_PATH")

# בדיקה שהנתיב נטען
if not env_path:
    print("ERROR: Could not find DATA_PATH in .env file")
    exit()

base_path = Path(env_path)

# 2. טעינת הקובץ המאוחד
csv_path = base_path / "indiana_merged_data.csv"
print(f"Loading data from: {csv_path}...")

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    print(f"ERROR: File not found at {csv_path}")
    print("Did you run the previous script successfully?")
    exit()

# 3. ניתוח עמודת Problems
print("-" * 30)
print("Analyzing 'Problems' column...")

all_conditions = []

# הופכים למחרוזת, אותיות קטנות, וממלאים חוסרים
# אנחנו בודקים את עמודת "Problems" (עם P גדולה כמו שראינו בתמונה שלך)
if "Problems" in df.columns:
    problems_series = df["Problems"].astype(str).str.lower().fillna("unknown")
else:
    # למקרה שהשם כתוב בקטן
    problems_series = df["problems"].astype(str).str.lower().fillna("unknown")

for row in problems_series:
    # אם כתוב "normal", נשמור אותו
    if row == "normal":
        all_conditions.append("normal")
        continue

    # אם יש כמה מחלות מופרדות בנקודה-פסיק או פסיק
    # מנקים תווים מיותרים
    clean_row = row.replace(";", ",").replace("/", ",")
    diseases = clean_row.split(",")

    for d in diseases:
        d = d.strip()  # הורדת רווחים מיותרים
        if d != "normal" and d != "nan" and d != "unknown" and len(d) > 2:
            all_conditions.append(d)

# 4. ספירה והצגה
counts = Counter(all_conditions)

print(f"\nTotal 'Normal' images: {counts['normal']}")
print("\nTop 10 Diseases found (Candidates for your model):")
print("-" * 30)

# מציג את 10 המחלות הכי נפוצות
for i, (disease, count) in enumerate(counts.most_common(11)):
    if disease == "normal":
        continue
    print(f"{disease}: {count}")

print("-" * 30)