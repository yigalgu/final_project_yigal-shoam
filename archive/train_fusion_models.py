import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import os
from dotenv import load_dotenv
from pathlib import Path

# --- הגדרות ---
load_dotenv()
env_path = os.getenv("DATA_PATH")
if not env_path:
    # ברירת מחדל למקרה שה-env לא נטען
    print("Warning: DATA_PATH not found in .env, using current directory.")
    base_path = Path("..")
else:
    base_path = Path(env_path)

print("🚀 Starting Multimodal Training Benchmark...")

# 1. טעינת הנתונים
csv_path = base_path / "indiana_poc_balanced.csv"
img_pkl_path = base_path / "image_features.pkl"
txt_pkl_path = base_path / "text_features.pkl"

# בדיקה שהקבצים קיימים
if not img_pkl_path.exists():
    print(f"❌ Error: Missing {img_pkl_path}")
    exit()
if not txt_pkl_path.exists():
    print(f"❌ Error: Missing {txt_pkl_path}")
    exit()

print("Loading data files...")
df = pd.read_csv(csv_path)

with open(img_pkl_path, 'rb') as f:
    img_features = pickle.load(f)

with open(txt_pkl_path, 'rb') as f:
    txt_features = pickle.load(f)

print(f"Loaded {len(img_features)} image vectors and {len(txt_features)} text vectors.")

# 2. יצירת הדאטה-סט המשותף (Data Alignment)
X_img = []
X_txt = []
X_fusion = []
y = []

valid_count = 0
missing_text_count = 0
missing_img_count = 0

print("Aligning data (Matching Image + Text)...")

for index, row in df.iterrows():
    fname = str(row['filename'])
    label = row['label']

    # ניסיון למצוא את המפתח במילונים (עם ובלי סיומת .png)
    # זה מנגנון חכם שמונע תקלות של שמות קבצים

    # חיפוש בתמונות
    img_vec = None
    if fname in img_features:
        img_vec = img_features[fname]
    elif fname.replace('.png', '') in img_features:  # נסה בלי סיומת
        img_vec = img_features[fname.replace('.png', '')]

    # חיפוש בטקסט
    txt_vec = None
    if fname in txt_features:
        txt_vec = txt_features[fname]
    elif fname.replace('.png', '') in txt_features:  # נסה בלי סיומת
        txt_vec = txt_features[fname.replace('.png', '')]
    elif fname + ".png" in txt_features:  # נסה להוסיף סיומת
        txt_vec = txt_features[fname + ".png"]

    # אם מצאנו את שניהם - נחבר!
    if img_vec is not None and txt_vec is not None:
        # יצירת וקטור משולב (Concatenation)
        # זה המימוש של "היתוך המידע" מהשקף שלך
        vec_f = np.concatenate([img_vec, txt_vec])

        X_img.append(img_vec)
        X_txt.append(txt_vec)
        X_fusion.append(vec_f)
        y.append(label)
        valid_count += 1
    else:
        if img_vec is None: missing_img_count += 1
        if txt_vec is None: missing_text_count += 1

print(f"✅ Ready for training! Aligned {valid_count} samples.")
print(f"(Skipped: {missing_img_count} missing images, {missing_text_count} missing texts)")

# המרה למערכי Numpy (כדי שהמודל יבין)
X_img = np.array(X_img)
X_txt = np.array(X_txt)
X_fusion = np.array(X_fusion)
y = np.array(y)

# קידוד המחלות למספרים (Pneumonia -> 0, Normal -> 1...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = list(le.classes_)
print(f"Classes: {classes}")

# 3. חלוקה לקבוצות אימון/מבחן (Split)
# שימוש ב-random_state קבוע כדי שההשוואה תהיה הוגנת
if len(y) < 10:
    print("❌ Error: Not enough data to split. Check your pickles!")
    exit()

idx_train, idx_test, y_train, y_test = train_test_split(
    np.arange(len(y)), y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)


# פונקציית עזר לאימון
def train_and_evaluate(name, X_data):
    print(f"\n--- Training {name} Model ---")
    X_tr, X_te = X_data[idx_train], X_data[idx_test]

    # Logistic Regression פשוט ומהיר
    clf = LogisticRegression(max_iter=3000, random_state=42)
    clf.fit(X_tr, y_train)

    preds = clf.predict(X_te)
    acc = accuracy_score(y_test, preds)
    print(f"🎯 {name} Accuracy: {acc:.2%}")
    return acc, clf


# 4. הרצת התחרות
acc_img, model_img = train_and_evaluate("Image-Only", X_img)
acc_txt, model_txt = train_and_evaluate("Text-Only", X_txt)
acc_fusion, model_fusion = train_and_evaluate("Fusion (Multi-modal)", X_fusion)

# 5. ציור גרף הניצחון
results = {
    'Image Only': acc_img,
    'Text Only': acc_txt,
    'Fusion (Ours)': acc_fusion
}

plt.figure(figsize=(10, 6))
# צבעים: כחול, כתום, ירוק (ירוק למנצח)
colors = ['#3498db', '#f39c12', '#2ecc71']
bars = plt.bar(results.keys(), results.values(), color=colors)

# הוספת אחוזים מעל העמודות
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.1%}', ha='center', fontweight='bold', fontsize=12)

plt.title('Performance Comparison: Unimodal vs Multimodal', fontsize=16)
plt.ylabel('Accuracy', fontsize=12)
plt.ylim(0, 1.15)
plt.grid(axis='y', alpha=0.3)

save_path = base_path / "benchmark_results.png"
plt.savefig(save_path, dpi=300)
print(f"\n📊 Graph saved to: {save_path}")

# 6. הצגת הבלבול של המודל המנצח (כדי לדעת איפה הוא טועה)
print("\n--- Fusion Model Detailed Report ---")
fusion_preds = model_fusion.predict(X_fusion[idx_test])
print(classification_report(y_test, fusion_preds, target_names=classes))

# 7. שמירת המודל הסופי
final_model_data = {
    "model": model_fusion,
    "encoder": le,
    "classes": classes,
    "accuracy": acc_fusion,
    "description": "Logistic Regression on Concatenated Features (DenseNet121 + ClinicalBERT)"
}
with open(base_path / "best_fusion_model.pkl", 'wb') as f:
    pickle.dump(final_model_data, f)
    print("🏆 Best model saved as 'best_fusion_model.pkl'")

plt.show()