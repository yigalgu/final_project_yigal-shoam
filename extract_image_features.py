import torch
import torchxrayvision as xrv
from skimage.io import imread
from skimage.transform import resize  # ייבוא פונקציית שינוי גודל
import pandas as pd
import numpy as np
import pickle
import os
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# --- 1. הגדרות ---
current_script_path = Path(__file__).parent.absolute()
load_dotenv(current_script_path / ".env")

data_path_str = os.getenv("DATA_PATH")
base_path = Path(data_path_str)
input_csv = current_script_path / "indiana_poc_large.csv"
output_pkl = current_script_path / "image_features.pkl"
images_dir = base_path / "images" / "images_normalized"

print(f"🚀 Starting FAST Medical Extraction (Resize to 224x224)")
df = pd.read_csv(input_csv)

# --- 2. טעינת המודל ---
model = xrv.models.DenseNet(weights="densenet121-res224-all")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --- 3. לולאה מהירה ---
features = {}
missing = 0

print("⏳ Processing images...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    filename = row['filename']
    img_path = images_dir / filename

    if not img_path.exists():
        missing += 1
        continue

    try:
        # 1. טעינת תמונה (4K כבד)
        img = imread(img_path)

        # 2. נרמול לטווח שהמודל מכיר
        img = xrv.datasets.normalize(img, 255)

        # 3. טיפול בערוצים (שחור לבן)
        if len(img.shape) > 2:
            img = img.mean(2)

        # --- התיקון הקריטי: הקטנה ל-224x224 ---
        # זה הופך את התמונה לקלה פי 100 ומותאמת למודל
        img = resize(img, (224, 224), anti_aliasing=True)

        # 4. הוספת מימדים
        img = img[None, None, :]

        input_tensor = torch.from_numpy(img).float().to(device)

        with torch.no_grad():
            feats = model.features(input_tensor)
            feats = torch.nn.functional.adaptive_avg_pool2d(feats, (1, 1))
            vec = feats.view(feats.size(0), -1).cpu().numpy().flatten()

        features[filename] = vec

    except Exception as e:
        print(f"Error {filename}: {e}")

print(f"✅ Done. Saved to {output_pkl}")
with open(output_pkl, 'wb') as f: pickle.dump(features, f)