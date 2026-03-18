import pandas as pd
from pathlib import Path
import os
from dotenv import load_dotenv

# 1. טעינת הגדרות
load_dotenv()
base_path = Path(os.getenv("DATA_PATH"))

input_csv = base_path / "indiana_frontal.csv"
poc_output_csv = base_path / "indiana_poc_balanced.csv"

RANDOM_SEED = 42

# 2. טעינה
print(f"Loading data from: {input_csv}")
df = pd.read_csv(input_csv)


# 3. תיוג (Labeling Logic)
def assign_label(row):
    text_parts = [
        str(row.get('findings', '')),
        str(row.get('impression', '')),
        str(row.get('Problems', '')),
        str(row.get('problems', ''))
    ]
    text = " ".join(text_parts).lower()

    if 'cardiomegaly' in text:
        return 'Cardiomegaly'
    if 'opacity' in text or 'pneumonia' in text or 'airspace disease' in text:
        return 'Opacity'
    if 'normal' in text or 'no acute' in text:
        return 'Normal'
    return 'Other'


print("Assigning labels...")
df['label'] = df.apply(assign_label, axis=1)

# סינון ראשוני
df_labeled = df[df['label'] != 'Other'].copy()

# 4. יצירת ה-POC המצומצם (150 מכל סוג)
# שיניתי ל-150 לפי בקשתך - לייט ומהיר
SAMPLES_PER_CLASS = 150

print(f"\nBalancing dataset (Taking {SAMPLES_PER_CLASS} from each class)...")

normal_df = df_labeled[df_labeled['label'] == 'Normal']
opacity_df = df_labeled[df_labeled['label'] == 'Opacity']
cardio_df = df_labeled[df_labeled['label'] == 'Cardiomegaly']

# בדיקה שיש לנו מספיק (למרות שאנחנו יודעים שיש)
try:
    df_poc = pd.concat([
        normal_df.sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED),
        opacity_df.sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED),
        cardio_df.sample(n=SAMPLES_PER_CLASS, random_state=RANDOM_SEED)
    ])
except ValueError:
    print("Warning: Not enough data for requested size. Taking maximum possible.")
    min_val = min(len(normal_df), len(opacity_df), len(cardio_df))
    df_poc = pd.concat([
        normal_df.sample(n=min_val, random_state=RANDOM_SEED),
        opacity_df.sample(n=min_val, random_state=RANDOM_SEED),
        cardio_df.sample(n=min_val, random_state=RANDOM_SEED)
    ])

# ערבוב סופי
df_poc = df_poc.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# 5. שמירה
df_poc.to_csv(poc_output_csv, index=False, encoding='utf-8-sig')

print("-" * 30)
print(f"SUCCESS: Created Light POC dataset at: {poc_output_csv}")
print(f"Total images: {len(df_poc)}")
print("Class breakdown:")
print(df_poc['label'].value_counts())
print("-" * 30)