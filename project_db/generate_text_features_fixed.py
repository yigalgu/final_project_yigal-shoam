import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv
import pickle

# 1. הגדרות ונתיבים
load_dotenv()
env_path = os.getenv("DATA_PATH")
base_path = Path(env_path)

input_csv = base_path / "indiana_reports_with_summary.csv"
# שינוי 1: סיומת pkl במקום npy
output_path = base_path / "text_features.pkl"

# 2. טעינת BERT
print("Loading ClinicalBERT model...")
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)
model.eval()


def get_embedding(text):
    """הופך טקסט בודד לווקטור באורך 768"""
    if not isinstance(text, str) or not text.strip():
        return np.zeros(768)

    # טוקניזציה
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # לקיחת ה-CLS Token
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return embedding


# 3. טעינת הנתונים
df = pd.read_csv(input_csv)
df['clinical_summary'] = df['clinical_summary'].fillna("")

print(f"Processing {len(df)} reports...")

# שינוי 2: שימוש במילון במקום רשימה
features_dict = {}

# אנחנו צריכים לדעת מה שם העמודה של שם הקובץ/המזהה
# נניח שזה 'filename', אם זה 'uid' או 'image_id' צריך לשנות כאן
id_col = 'filename'
if id_col not in df.columns:
    # בדיקה למקרה שהשם שונה
    possible_cols = ['uid', 'image_id', 'id']
    for col in possible_cols:
        if col in df.columns:
            id_col = col
            break

print(f"Using '{id_col}' as key for the dictionary.")

# 4. הרצה
for index, row in tqdm(df.iterrows(), total=len(df)):
    text = row['clinical_summary']
    file_id = row[id_col]

    # וידוא ששם הקובץ מכיל סיומת (כדי שיתאים לתמונות שלך)
    # אם שמות הקבצים ב-CSV הם כבר עם .png, השורה הזו לא תפריע
    file_key = str(file_id)
    if not file_key.endswith(('.png', '.jpg', '.jpeg')):
        file_key = f"{file_key}.png"

    vector = get_embedding(text)

    # שמירה במילון: המפתח הוא שם הקובץ, הערך הוא הווקטור
    features_dict[file_key] = vector

# 5. שמירה כ-Pickle
print(f"\nSaving dictionary with {len(features_dict)} items...")
with open(output_path, 'wb') as f:
    pickle.dump(features_dict, f)

print(f"SUCCESS! Saved to: {output_path}")
print("Now run 'train_multimodal_fusion.py' - it will work automatically.")