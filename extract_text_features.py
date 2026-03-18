import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# --- 1. הגדרות סביבה ---
current_script_path = Path(__file__).parent.absolute()
env_file_path = current_script_path / ".env"
load_dotenv(env_file_path)

project_path = current_script_path
input_csv = project_path / "indiana_poc_large.csv"
output_pkl = project_path / "text_features.pkl"

print(f"🚀 Starting Text Feature Extraction (ClinicalBERT)")
print(f"📂 Reading CSV: {input_csv}")

if not input_csv.exists():
    print("❌ Error: indiana_poc_large.csv not found!")
    exit()

df = pd.read_csv(input_csv)

# --- 2. טעינת ClinicalBERT ---
print("📚 Loading ClinicalBERT tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

# --- 3. לולאת החילוץ ---
text_features = {}
empty_count = 0

print("⏳ Extracting text embeddings...")

for index, row in tqdm(df.iterrows(), total=len(df)):
    filename = str(row['filename'])

    # חיבור הטקסטים (טיפול בערכים ריקים)
    findings = str(row['findings']) if pd.notna(row['findings']) else ""
    impression = str(row['impression']) if pd.notna(row['impression']) else ""

    full_text = f"Findings: {findings}. Impression: {impression}"

    # אם אין טקסט בכלל, מדלגים או שמים אפסים (כאן נדלג)
    if len(full_text.strip()) < 25:  # סינון טקסטים קצרים מדי/ריקים
        empty_count += 1
        continue

    try:
        inputs = tokenizer(full_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # לוקחים את ה-CLS token (הייצוג של כל המשפט)
            cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()

        text_features[filename] = cls_embedding

    except Exception as e:
        print(f"⚠️ Error processing text for {filename}: {e}")

# --- 4. שמירה ---
print(f"\n✅ Finished! Extracted {len(text_features)} text vectors.")
print(f"ℹ️  Skipped {empty_count} records due to missing/short text.")

with open(output_pkl, 'wb') as f:
    pickle.dump(text_features, f)

print(f"💾 Saved features to: {output_pkl}")