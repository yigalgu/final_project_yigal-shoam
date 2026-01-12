import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import os
from pathlib import Path
from dotenv import load_dotenv

# 1. הגדרות ונתיבים
load_dotenv()
env_path = os.getenv("DATA_PATH")
base_path = Path(env_path)
input_csv = base_path / "indiana_reports_with_summary.csv"
output_embeddings = base_path / "text_embeddings.npy"

# 2. טעינת המודל והטוקנייזר (ClinicalBERT)
print("Loading ClinicalBERT model...")
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# העברה ל-GPU אם קיים
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()  # מצב הערכה (ללא Dropout)


def get_embedding(text):
    """הופך טקסט בודד לווקטור באורך 768"""
    if not text or text == "":
        return np.zeros(768)

    # טוקניזציה
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    # אנחנו לוקחים את ה-Hidden State של ה-Token הראשון ([CLS])
    # הוא נחשב למייצג הטוב ביותר של כל המשפט במודלי BERT
    embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    return embedding


# 3. טעינת הנתונים
df = pd.read_csv(input_csv)
# נמלא ערכים ריקים בסיכום ליתר ביטחון
df['clinical_summary'] = df['clinical_summary'].fillna("")

# 4. הרצה על כל הטבלה
print(f"Generating embeddings for {len(df)} reports...")
all_embeddings = []

for text in tqdm(df['clinical_summary']):
    vector = get_embedding(text)
    all_embeddings.append(vector)

# 5. שמירה
embeddings_array = np.array(all_embeddings)
np.save(output_embeddings, embeddings_array)

print(f"\nSUCCESS: Created embeddings array of shape {embeddings_array.shape}")
print(f"Saved to: {output_embeddings}")