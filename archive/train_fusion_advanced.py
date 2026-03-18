import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import os
from dotenv import load_dotenv
from pathlib import Path

# --- הגדרות ---
load_dotenv()
env_path = os.getenv("DATA_PATH")
base_path = Path(env_path) if env_path else Path("..")

print("🚀 Starting Advanced Multimodal Training (PCA + Random Forest)...")

# 1. טעינת הנתונים
csv_path = base_path / "indiana_poc_balanced.csv"
img_pkl_path = base_path / "image_features.pkl"
txt_pkl_path = base_path / "text_features.pkl"

# בדיקת קבצים
if not img_pkl_path.exists() or not txt_pkl_path.exists():
    print("❌ Error: Missing feature files.")
    exit()

df = pd.read_csv(csv_path)
with open(img_pkl_path, 'rb') as f: img_features = pickle.load(f)
with open(txt_pkl_path, 'rb') as f: txt_features = pickle.load(f)

# 2. סידור הנתונים (Alignment)
X_img = []
X_txt = []
y = []
valid_count = 0

print("Aligning data...")
for index, row in df.iterrows():
    fname = str(row['filename'])
    label = row['label']

    # שליפת תמונה (טיפול בסיומות)
    img_vec = img_features.get(fname)
    if img_vec is None: img_vec = img_features.get(fname.replace('.png', ''))

    # שליפת טקסט (טיפול בסיומות)
    txt_vec = txt_features.get(fname)
    if txt_vec is None: txt_vec = txt_features.get(fname.replace('.png', ''))
    if txt_vec is None: txt_vec = txt_features.get(fname + ".png")

    # רק אם יש את שניהם - מוסיפים לרשימה
    if img_vec is not None and txt_vec is not None:
        X_img.append(img_vec)
        X_txt.append(txt_vec)
        y.append(label)
        valid_count += 1

print(f"✅ Aligned {valid_count} samples successfully.")

X_img = np.array(X_img)
X_txt = np.array(X_txt)
y = np.array(y)

# קידוד התגיות
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = list(le.classes_)

# 3. חלוקה לאימון ומבחן
# שים לב: אנחנו שומרים את האינדקסים כדי לפצל את התמונות והטקסט באותו אופן
idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.2, random_state=42, stratify=y_encoded
)

y_train, y_test = y_encoded[idx_train], y_encoded[idx_test]

# --- שלב השיפורים (Preprocessing Pipeline) ---
print("Applying Scaling & PCA...")

# א. נרמול (Scaling) - מביא את כולם לאותה סקאלה
scaler_img = StandardScaler()
scaler_txt = StandardScaler()

# לומדים רק מהאימון!
X_img_train = scaler_img.fit_transform(X_img[idx_train])
X_img_test = scaler_img.transform(X_img[idx_test])

X_txt_train = scaler_txt.fit_transform(X_txt[idx_train])
X_txt_test = scaler_txt.transform(X_txt[idx_test])

# ב. הפחתת מימדים (PCA) לתמונות
# מכווצים 1024 תכונות ל-50 התכונות הכי משמעותיות כדי להפחית רעש
pca = PCA(n_components=50, random_state=42)
X_img_train_pca = pca.fit_transform(X_img_train)
X_img_test_pca = pca.transform(X_img_test)

print(f"Compressed Image Vector from 1024 to {X_img_train_pca.shape[1]} features.")

# ג. יצירת המיזוג (Fusion)
# מחברים את התמונה המכווצת + הטקסט המנורמל
X_fusion_train = np.concatenate([X_img_train_pca, X_txt_train], axis=1)
X_fusion_test = np.concatenate([X_img_test_pca, X_txt_test], axis=1)


# 4. אימון והשוואה (מודלים)

def train_and_eval(name, X_tr, X_te):
    # שימוש ב-Random Forest במקום Logistic Regression
    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    clf.fit(X_tr, y_train)
    pred = clf.predict(X_te)
    acc = accuracy_score(y_test, pred)
    return acc, clf


print("\n--- Training Models ---")

# אימון תמונה (אחרי PCA)
acc_img, _ = train_and_eval("Image (PCA)", X_img_train_pca, X_img_test_pca)
print(f"🎯 Image Accuracy: {acc_img:.2%}")

# אימון טקסט (מנורמל)
acc_txt, _ = train_and_eval("Text Only", X_txt_train, X_txt_test)
print(f"🎯 Text Accuracy:  {acc_txt:.2%}")

# אימון משולב
acc_fusion, best_model = train_and_eval("Fusion (Multimodal)", X_fusion_train, X_fusion_test)
print(f"🎯 Fusion Accuracy: {acc_fusion:.2%}")

# 5. יצירת גרף התוצאות
results = {'Image Only': acc_img, 'Text Only': acc_txt, 'Fusion (Ours)': acc_fusion}

plt.figure(figsize=(9, 6))
# צבעים: כחול, כתום, ירוק
bars = plt.bar(results.keys(), results.values(), color=['#3498db', '#f39c12', '#2ecc71'])

# הוספת אחוזים
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.1%}', ha='center', fontweight='bold', fontsize=12)

plt.title('Improved Benchmark: PCA + Random Forest', fontsize=16)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis='y', alpha=0.3)

save_path = base_path / "benchmark_results_advanced.png"
plt.savefig(save_path, dpi=300)
print(f"\n📊 Graph saved to: {save_path}")

# דו"ח מפורט למודל המנצח
print("\n--- Fusion Model Detailed Report ---")
preds = best_model.predict(X_fusion_test)
print(classification_report(y_test, preds, target_names=classes))

# שמירת המודל הסופי
model_data = {
    "model": best_model,
    "pca": pca,
    "scaler_img": scaler_img,
    "scaler_txt": scaler_txt,
    "encoder": le,
    "classes": classes
}
with open(base_path / "best_fusion_model.pkl", 'wb') as f:
    pickle.dump(model_data, f)
    print("🏆 Best model saved.")

plt.show()