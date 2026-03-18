import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path

# --- הגדרות נתיבים ---
current_script_path = Path(__file__).parent.absolute()
env_file_path = current_script_path / ".env"
load_dotenv(env_file_path)

data_path_str = os.getenv("DATA_PATH")
if not data_path_str:
    print("❌ Error: 'DATA_PATH' not found in .env file.")
    exit()

base_path = Path(data_path_str)
input_csv = base_path / "indiana_frontal.csv"
output_csv = current_script_path / "indiana_poc_large.csv"

print(f"📂 Loading source data from: {input_csv}")
if not input_csv.exists():
    print(f"❌ Error: File not found at: {input_csv}")
    exit()

df = pd.read_csv(input_csv)

# --- הלוגיקה החדשה (Contains במקום Exact Match) ---
print("\n🔍 Filtering data...")

# 1. Normal: כאן התאמה מדויקת היא טובה (כי בריא הוא רק בריא)
# ממירים לאותיות קטנות ליתר ביטחון
df_normal = df[df['Problems'].str.lower() == 'normal'].copy()
df_normal['label'] = 'Normal'

# 2. Cardiomegaly: חיפוש המילה בתוך הטקסט
df_cardio = df[df['Problems'].str.contains('Cardiomegaly', case=False, na=False)].copy()
df_cardio['label'] = 'Cardiomegaly'

# 3. Opacity: חיפוש המילה בתוך הטקסט
df_opacity = df[df['Problems'].str.contains('Opacity', case=False, na=False)].copy()
df_opacity['label'] = 'Opacity'

# --- טיפול בחפיפות (Duplicates) ---
# יש חולים שיש להם גם וגם. אנחנו ניתן עדיפות ל-Cardiomegaly כי היא הקבוצה הקטנה ביותר.
# נסיר מ-Opacity את כל החולים שכבר נמצאים ב-Cardiomegaly
before_dedup = len(df_opacity)
df_opacity = df_opacity[~df_opacity['filename'].isin(df_cardio['filename'])]
after_dedup = len(df_opacity)

print(f"   (Removed {before_dedup - after_dedup} duplicate images that had both Cardio + Opacity)")

# --- בחירת הכמויות ---
print("\n✂️  Balancing dataset...")

# Normal: לוקחים 700 (או פחות אם אין)
n_normal = min(len(df_normal), 700)
df_normal_sample = df_normal.sample(n=n_normal, random_state=42)

# איחוד הכל
df_final = pd.concat([df_normal_sample, df_cardio, df_opacity])

# ערבוב סופי
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n✅ New Dataset Created: 'indiana_poc_large.csv'")
print("---------------------------------------------")
print(df_final['label'].value_counts())
print("---------------------------------------------")
print(f"Total samples: {len(df_final)}")

# שמירה
print(f"Saving to: {output_csv}")
df_final.to_csv(output_csv, index=False)
print(f"\n💾 Saved successfully!")