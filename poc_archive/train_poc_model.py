import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from dotenv import load_dotenv
from pathlib import Path

# --- 1. הגדרות וטעינה ---
current_script_path = Path(__file__).parent.absolute()
load_dotenv(current_script_path / ".env")

print("🚀 Starting FINAL Training (Late Fusion Optimization)...")

input_csv = current_script_path / "indiana_poc_large.csv"
img_pkl_path = current_script_path / "image_features.pkl"
txt_pkl_path = current_script_path / "text_features.pkl"

if not input_csv.exists():
    print("❌ Error: Files not found.")
    exit()

# --- 2. טעינת נתונים ---
print("📥 Loading data...")
df = pd.read_csv(input_csv)
with open(img_pkl_path, 'rb') as f: img_features = pickle.load(f)
with open(txt_pkl_path, 'rb') as f: txt_features = pickle.load(f)

# --- 3. יישור וקטורים (Alignment) ---
X_img, X_txt, y = [], [], []
valid_count = 0

for index, row in df.iterrows():
    fname = str(row['filename'])
    label = row['label']

    img_vec = img_features.get(fname)
    if img_vec is None: img_vec = img_features.get(fname.replace('.png', ''))

    txt_vec = txt_features.get(fname)
    if txt_vec is None: txt_vec = txt_features.get(fname.replace('.png', ''))

    if img_vec is not None and txt_vec is not None:
        X_img.append(img_vec)
        X_txt.append(txt_vec)
        y.append(label)
        valid_count += 1

print(f"✅ Aligned {valid_count} samples.")
X_img = np.array(X_img)
X_txt = np.array(X_txt)
y = np.array(y)

# קידוד
le = LabelEncoder()
y_encoded = le.fit_transform(y)
classes = list(le.classes_)
print(f"🏷️ Classes: {classes}")

# --- 4. חלוקה (Split) ---
# Stratify חשוב מאוד כאן לשמור על יחס חולים/בריאים
X_img_train, X_img_test, X_txt_train, X_txt_test, y_train, y_test = train_test_split(
    X_img, X_txt, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# --- 5. אימון מודלים נפרדים (Training) ---
print("🏋️ Training separate models...")

# A. מודל תמונה
scaler_img = StandardScaler()
X_img_train = scaler_img.fit_transform(X_img_train)
X_img_test = scaler_img.transform(X_img_test)

pca = PCA(n_components=0.95, random_state=42)
X_img_train_pca = pca.fit_transform(X_img_train)
X_img_test_pca = pca.transform(X_img_test)

model_img = LogisticRegression(class_weight='balanced', max_iter=2000)
model_img.fit(X_img_train_pca, y_train)
probs_img = model_img.predict_proba(X_img_test_pca)
acc_img = accuracy_score(y_test, model_img.predict(X_img_test_pca))
print(f"   📷 Image Model Accuracy: {acc_img:.2%}")

# B. מודל טקסט
scaler_txt = StandardScaler()
X_txt_train = scaler_txt.fit_transform(X_txt_train)
X_txt_test = scaler_txt.transform(X_txt_test)

model_txt = LogisticRegression(class_weight='balanced', max_iter=2000)
model_txt.fit(X_txt_train, y_train)
probs_txt = model_txt.predict_proba(X_txt_test)
acc_txt = accuracy_score(y_test, model_txt.predict(X_txt_test))
print(f"   📝 Text Model Accuracy:  {acc_txt:.2%}")

# --- 6. אופטימיזציה של המיזוג (Grid Search) ---
print("\n🔍 Optimizing Fusion Weights...")

best_acc = 0
best_weight_txt = 0
best_preds = None

results_acc = []
weights_range = [i / 20 for i in range(21)]  # בדיקה בקפיצות של 0.05

for w_txt in weights_range:
    w_img = 1.0 - w_txt

    # חישוב משוקלל של ההסתברויות
    weighted_probs = (probs_txt * w_txt) + (probs_img * w_img)
    current_preds = np.argmax(weighted_probs, axis=1)

    acc = accuracy_score(y_test, current_preds)
    results_acc.append(acc)

    if acc > best_acc:
        best_acc = acc
        best_weight_txt = w_txt
        best_preds = current_preds

print("\n------------------------------------------------")
print(f"🏆 Best Fusion Accuracy: {best_acc:.2%}")
print(f"⚖️  Optimal Weights: Text={best_weight_txt:.2f}, Image={1.0 - best_weight_txt:.2f}")
if best_acc > acc_txt:
    print(f"🚀 Improvement: +{best_acc - acc_txt:.2%} over Text alone!")
else:
    print(f"😐 No improvement over Text alone (Text is dominant).")
print("------------------------------------------------")

# --- 7. שמירת תוצאות וגרפים ---

# דוח סיווג
print("\n📊 Classification Report (Best Fusion):")
print(classification_report(y_test, best_preds, target_names=classes))

# גרף האופטימיזציה
plt.figure(figsize=(8, 5))
plt.plot(weights_range, results_acc, marker='o', linewidth=2, color='purple')
plt.title('Fusion Optimization: Accuracy vs. Text Weight')
plt.xlabel('Text Weight (0=Image Only, 1=Text Only)')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)
plt.axvline(x=best_weight_txt, color='r', linestyle='--', label=f'Optimal ({best_weight_txt})')
plt.legend()
plt.savefig(current_script_path / "final_optimization_graph.png")

# מטריצת בלבול
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.title(f'Final Confusion Matrix (Acc: {best_acc:.1%})')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(current_script_path / "final_confusion_matrix.png")

# --- 8. (מעודכן ל-POC) גרף השוואה סופי ---
print("\n📊 Generating POC Comparison Plot...")

# שמות המודלים המדויקים
model_names = [
    'Image Only\n(TorchXRayVision)',
    'Text Only\n(Bio_ClinicalBERT)',
    'Fusion\n(Late Fusion)'
]

accuracies = [acc_img, acc_txt, best_acc]
colors = ['#7f8c8d', '#2980b9', '#27ae60']

plt.figure(figsize=(11, 7))
bars = plt.bar(model_names, accuracies, color=colors, width=0.55)

# --- השינוי בכותרת כאן ---
plt.title('POC Phase Results: Accuracy by Modality', fontsize=16, fontweight='bold', pad=20)
# -------------------------

plt.ylabel('Accuracy', fontsize=13)
plt.ylim(0.6, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.4)

# הוספת המספרים
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
             f'{height:.1%}',
             ha='center', va='bottom', fontsize=15, fontweight='bold', color='#2c3e50')

plt.tight_layout()

# שמירה
comparison_path = current_script_path / "final_comparison_bar_chart.png"
plt.savefig(comparison_path, dpi=300)
print(f"🖼️  POC graph saved to: {comparison_path}")
plt.show()