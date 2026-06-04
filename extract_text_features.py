import os
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel

# טעינת משתני הסביבה (קובץ ה-.env)
load_dotenv()


def main():
    # משיכת נתיב הנתונים מקובץ ה-.env
    data_dir_str = os.getenv("DATA_PATH")
    if not data_dir_str:
        raise ValueError("DATA_PATH is not defined in the .env file. Please check your variable name.")

    data_dir = Path(data_dir_str)

    # הגדרת נתיב הקלט ונתיב הפלט החדש (כדי לא לדרוס את הבייסליין)
    input_file = data_dir / "clean_frontal_dataset.csv"
    output_features_file = data_dir / "text_features_finetuned.pt"

    print(f"Reading clean dataset from: {input_file}")
    df = pd.read_csv(input_file)

    if 'indication' not in df.columns:
        raise KeyError("The column 'indication' was not found in the dataset.")

    # טעינת המודל המאומן מתיקיית הפרויקט
    print("Loading FINE-TUNED Bio_ClinicalBERT model...")
    model_dir = data_dir / "best_text_model"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    # שימוש ב-AutoModel (ולא בסיווג) כדי לקבל את הווקטור (768 ממדים)
    model = AutoModel.from_pretrained(model_dir)

    # העברה למאיץ גרפי
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # מצב הערכה

    print(f"Running on device: {device}")

    text_features = []

    print("Extracting text features from 'indication' column using the fine-tuned model...")
    # לולאה שעוברת על סיבות ההפניה ומייצרת וקטור לכל שורה
    for idx, text in enumerate(df['indication']):
        if idx % 500 == 0:
            print(f"Processing row {idx}/{len(df)}...")

        inputs = tokenizer(str(text), return_tensors="pt", truncation=True, max_length=128, padding="max_length")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # לוקחים את הוקטור של טוקן ה-[CLS]
            cls_embedding = outputs.last_hidden_state[0, 0, :].cpu()
            text_features.append(cls_embedding)

    # איחוד לטנזור אחד
    text_features_tensor = torch.stack(text_features)

    print(f"Features extraction complete. Shape: {text_features_tensor.shape}")

    # שמירה
    torch.save(text_features_tensor, output_features_file)
    print(f"Saved FINETUNED text features tensor to: {output_features_file}")


if __name__ == "__main__":
    main()