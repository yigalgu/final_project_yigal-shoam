import os
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# טעינת משתני הסביבה (קובץ ה-.env)
load_dotenv()


def main():
    # משיכת נתיב הנתונים מקובץ ה-.env
    data_dir_str = os.getenv("DATA_PATH")
    if not data_dir_str:
        raise ValueError("DATA_PATH is not defined in the .env file.")
    data_dir = Path(data_dir_str)

    # 1. טעינת הנתונים
    dataset_file = data_dir / "clean_frontal_dataset.csv"
    print(f"Loading data from: {dataset_file}")
    df = pd.read_csv(dataset_file)

    # 2. פיצול הנתונים *בדיוק* כמו שעשינו בבייסליין (כדי להשוות תפוחים לתפוחים)
    _, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Target_Label'])

    texts = test_df['indication'].tolist()
    true_labels = test_df['Target_Label'].tolist()

    # 3. טעינת המודל המאומן שלנו (המומחה החדש שיצרנו)
    model_dir = data_dir / "best_text_model"
    print(f"Loading fine-tuned model from: {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)

    # העברה למאיץ הגרפי (MPS במק שלך)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # מצב הערכה

    print(f"Running on device: {device}")
    print("Running inference on the test set...")

    predictions = []

    # 4. הרצת המודל על נתוני המבחן (במנות של 16 משפטים כדי לא להעמיס על הזיכרון)
    batch_size = 16
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_texts = [str(t) for t in batch_texts]  # הבטחה שזה טקסט

        # טוקניזציה
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # חילוץ התחזית
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            # לוקחים את המחלקה עם הציון הגבוה ביותר
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            predictions.extend(preds)

    # 5. הדפסת התוצאות
    print("\n" + "=" * 40)
    print("📊 FINE-TUNED TEXT EVALUATION RESULTS 📊")
    print("=" * 40)

    acc = accuracy_score(true_labels, predictions)
    print(f"Overall Accuracy: {acc * 100:.2f}%\n")

    target_names = ['Normal (0)', 'Cardiomegaly (1)', 'Opacity (2)', 'Other (3)']
    print("Detailed Classification Report:")
    print(classification_report(true_labels, predictions, target_names=target_names))


if __name__ == "__main__":
    main()