import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. טעינת הגדרות
load_dotenv()
base_path = Path(os.getenv("DATA_PATH"))

# נתיבי קבצים
input_csv = base_path / "indiana_merged_data.csv"
output_csv = base_path / "indiana_frontal.csv"

# 2. טעינת הטבלה המקורית
print(f"Loading data from: {input_csv}")
df = pd.read_csv(input_csv)
original_count = len(df)

# 3. ביצוע הסינון (החלק החשוב)
# שומרים רק שורות שבהן בעמודת projection כתוב 'Frontal'
df_frontal = df[df['projection'] == 'Frontal'].copy()
frontal_count = len(df_frontal)

# 4. הצגת נתונים למשתמש
print("-" * 30)
print(f"Original dataset size: {original_count} images")
print(f"Filtered (Frontal only): {frontal_count} images")
print(f"Removed (Lateral): {original_count - frontal_count} images")
print("-" * 30)

# 5. שמירת הקובץ החדש
df_frontal.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"SUCCESS: New file created at: {output_csv}")