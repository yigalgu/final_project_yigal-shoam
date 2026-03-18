import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm
import pickle

# 1. הגדרות בסיס
load_dotenv()
base_path = Path(os.getenv("DATA_PATH"))

# הקלט: הקובץ המאוזן שיצרנו (450 תמונות)
csv_path = base_path / "indiana_poc_balanced.csv"
# הפלט: קובץ הווקטורים הסופי
output_path = base_path / "image_features.pkl"

# בחירת מעבד (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 2. טעינת המודל: DenseNet121
print("Loading DenseNet121 model...")
weights = models.DenseNet121_Weights.DEFAULT
model = models.densenet121(weights=weights)

# --- החלק החשוב: הסרת הראש ---
# אנחנו רוצים את ה"הבנה" של המודל (הווקטורים), לא את הסיווג הסופי.
# לכן אנחנו מחליפים את שכבת הסיווג ב"כלום" (Identity).
# זה יגרום למודל להוציא וקטור באורך 1024.
model.classifier = nn.Identity()

model = model.to(device)
model.eval()  # מצב קריאה בלבד (לא לומד כרגע)

# 3. הכנת התמונות (Preprocessing) - חובה לפי התקן של DenseNet
preprocess = transforms.Compose([
    transforms.Resize(256),  # הקטנה
    transforms.CenterCrop(224),  # חיתוך לריבוע
    transforms.ToTensor(),  # המרה למספרים
    transforms.Normalize(  # נרמול צבעים
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# 4. ביצוע החילוץ
df = pd.read_csv(csv_path)
print(f"Processing {len(df)} images...")

features_dict = {}
missing_count = 0

# לולאה על כל התמונות עם פס התקדמות
for index, row in tqdm(df.iterrows(), total=len(df)):
    image_name = row['filename']

    # בדיקת נתיבים (תתאים את זה למבנה התיקיות שלך אם צריך)
    # נסיון 1: בתוך תיקיית הנרמול
    img_path = base_path / "images" / "images_normalized" / image_name

    # נסיון 2: ישירות בתיקיית התמונות
    if not os.path.exists(img_path):
        img_path = base_path / "images" / image_name

    # אם עדיין לא מצאנו - מדלגים
    if not os.path.exists(img_path):
        missing_count += 1
        continue

    try:
        # טעינת תמונה
        img = Image.open(img_path).convert('RGB')

        # עיבוד (הקטנה ונרמול)
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        # הרצה במודל וקבלת הווקטור
        with torch.no_grad():
            feature_vector = model(img_tensor)

        # המרה למערך שטוח (במקום טנסור) כדי שאפשר יהיה לשמור
        features_dict[image_name] = feature_vector.flatten().cpu().numpy()

    except Exception as e:
        print(f"Error processing {image_name}: {e}")

# 5. שמירת התוצאה
print(f"\nExtraction Done.")
print(f"Successfully processed: {len(features_dict)} images")
if len(features_dict) > 0:
    # בדיקה שאכן קיבלנו וקטור בגודל 1024
    vector_size = len(next(iter(features_dict.values())))
    print(f"Vector size: {vector_size} (Expected: 1024)")

if missing_count > 0:
    print(f"Warning: {missing_count} images were missing from the folder.")

# שמירה לקובץ
with open(output_path, 'wb') as f:
    pickle.dump(features_dict, f)

print(f"Saved features to: {output_path}")