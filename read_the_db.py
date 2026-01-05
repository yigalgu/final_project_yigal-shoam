import pandas as pd
from pathlib import Path
from PIL import Image

# ========= 1) הגדרות נתיבים =========
base_path = Path(r"C:\Users\igal6\OneDrive\שולחן העבודה\project_db")
# שים לב: השתמשתי בשם התיקייה כפי שציינת
images_folder = base_path / "images" / "images_normalized"

# ========= 2) טעינת הנתונים =========
df_proj = pd.read_csv(base_path / "indiana_projections.csv")
df_rep = pd.read_csv(base_path / "indiana_reports.csv")
df = df_proj.merge(df_rep, on="uid", how="left")

# ========= 3) בדיקה: מה כתוב ב-CSV? =========
example_filename = str(df["filename"].iloc[0])
print(f"DEBUG: Example filename from CSV: '{example_filename}'")

# ========= 4) בניית אינדקס תמונות (גמיש) =========
print("Indexing images...")
image_index = {}

# אנחנו נסרוק את התיקייה ונשמור את הנתיב לפי שם הקובץ ללא סיומת
for img_path in images_folder.glob("*.png"):
    # שם הקובץ המלא (למשל: 1_IM-0001-3001.dcm.png)
    full_name = img_path.name

    # ננסה להתאים לפי השם המלא או לפי ה-Base (לפני המקף השני)
    image_index[full_name] = str(img_path)

    # הוספת גרסה מקוצרת (למשל 1_IM-0001)
    parts = full_name.split("-")
    if len(parts) >= 2:
        short_name = f"{parts[0]}-{parts[1]}"
        if short_name not in image_index:
            image_index[short_name] = str(img_path)

# ========= 5) קישור התמונות =========
# ננסה למצוא התאמה ישירה
df["img_path"] = df["filename"].astype(str).map(image_index)

# ========= 6) בדיקה אם זה עבד =========
matches = df["img_path"].notna().sum()
print(f"Matched images: {matches} out of {len(df)}")

if matches == 0:
    print("\n--- ERROR DIAGNOSIS ---")
    print("Could not match any image. Here are some files I found in the folder:")
    found_files = [f.name for f in list(images_folder.glob("*.png"))[:5]]
    for f in found_files: print(f" - Found on disk: {f}")
    print(f"Compare these to the CSV filename: {example_filename}")
else:
    # שמירה והצגת דוגמה רק אם יש התאמות
    output_file = base_path / "indiana_merged_data.csv"
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    sample = df.dropna(subset=["img_path"]).iloc[0]
    print(f"\nSUCCESS! Opening: {sample['img_path']}")
    Image.open(sample["img_path"]).show()