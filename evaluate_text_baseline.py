import os
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# טעינת משתני הסביבה (קובץ ה-.env)
load_dotenv()


def main():
    # משיכת נתיב הנתונים מקובץ ה-.env (שים לב לשם המשתנה - DATA_PATH)
    data_dir_str = os.getenv("DATA_PATH")
    if not data_dir_str:
        raise ValueError("DATA_PATH is not defined in the .env file.")

    data_dir = Path(data_dir_str)

    # 1. הגדרת הנתיבים לקבצים הדרושים
    features_file = data_dir / "text_features.pt"
    dataset_file = data_dir / "clean_frontal_dataset.csv"

    print("Loading data...")
    # טעינת הווקטורים (ה-X שלנו)
    if not features_file.exists():
        raise FileNotFoundError(f"Could not find text features at {features_file}")

    # טוענים את הטנזור וממירים אותו ל-NumPy (הפורמט ש-scikit-learn מצפה לו)
    X = torch.load(features_file, map_location="cpu").numpy()

    # טעינת התוויות (ה-Y שלנו)
    if not dataset_file.exists():
        raise FileNotFoundError(f"Could not find clean dataset at {dataset_file}")
    df = pd.read_csv(dataset_file)
    y = df['Target_Label'].values

    # נוודא שמספר הווקטורים תואם למספר התוויות
    if len(X) != len(y):
        raise ValueError(f"Mismatch! Found {len(X)} features but {len(y)} labels.")

    print(f"Dataset shape: X={X.shape}, y={y.shape}")

    # 2. חלוקת הנתונים (80% אימון, 20% מבחן)
    # random_state=42 מבטיח שאם נריץ שוב, נקבל את אותה החלוקה בדיוק
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nTraining Logistic Regression model on text features...")
    # 3. אימון המודל הפשוט (Linear Probing)
    # max_iter=1000 כי למודלים על טקסט לפעמים לוקח קצת זמן להתכנס
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    print("Evaluating model...")
    # 4. בדיקת ביצועים על נתוני המבחן
    y_pred = clf.predict(X_test)

    # חישוב והדפסת המדדים
    accuracy = accuracy_score(y_test, y_pred)

    print("\n" + "=" * 40)
    print("📊 BASELINE TEXT EVALUATION RESULTS 📊")
    print("=" * 40)
    print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

    # פירוט מלא של הביצועים לכל מחלקה
    target_names = ['Normal (0)', 'Cardiomegaly (1)', 'Opacity (2)', 'Other (3)']
    print("Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == "__main__":
    main()