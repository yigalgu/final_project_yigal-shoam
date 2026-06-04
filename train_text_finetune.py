import os
import torch
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import evaluate
import numpy as np

# טעינת משתני הסביבה (קובץ ה-.env)
load_dotenv()


def main():
    # משיכת נתיב הנתונים מקובץ ה-.env
    data_dir_str = os.getenv("DATA_PATH")
    if not data_dir_str:
        raise ValueError("DATA_PATH is not defined in the .env file.")
    data_dir = Path(data_dir_str)

    # 1. טעינת הנתונים הנקיים
    dataset_file = data_dir / "clean_frontal_dataset.csv"
    print(f"Loading data from: {dataset_file}")
    df = pd.read_csv(dataset_file)

    # נוודא שהעמודות קיימות
    if 'indication' not in df.columns or 'Target_Label' not in df.columns:
        raise ValueError("Missing 'indication' or 'Target_Label' in the dataset.")

    # 2. פיצול הנתונים ושמירה כפורמט Dataset של HuggingFace
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Target_Label'])

    # המרה למבנה ש-Trainer מבין
    train_dataset = Dataset.from_pandas(
        train_df[['indication', 'Target_Label']].rename(columns={'Target_Label': 'label'}))
    val_dataset = Dataset.from_pandas(val_df[['indication', 'Target_Label']].rename(columns={'Target_Label': 'label'}))

    dataset = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })

    # 3. טעינת מודל וטוקנייזר ל-Fine-Tuning
    model_name = "emilyalsentzer/Bio_ClinicalBERT"
    print(f"Loading tokenizer and model for sequence classification: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # נטען מודל עם "ראש" של 4 מחלקות
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)

    # 4. פונקציית טוקניזציה שתופעל על כל הנתונים
    def tokenize_function(examples):
        return tokenizer(examples["indication"], truncation=True, max_length=128)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # 5. הגדרות האימון (Hyperparameters)
    output_dir = data_dir / "text_finetuned_model"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",  # נבדוק דיוק בסוף כל אופוק (סיבוב)
        learning_rate=2e-5,  # Learning rate נמוך קלאסי ל-Fine-tuning
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,  # בדרך כלל 3-5 סיבובים מספיקים ל-Transformers
        weight_decay=0.01,  # למניעת Overfitting
        save_strategy="epoch",  # שומרים מודל בסוף כל סיבוב
        load_best_model_at_end=True,  # מחזירים את המודל הכי טוב שהיה
        metric_for_best_model="accuracy"
    )

    # 6. הגדרת מדדי ההערכה (Metrics)
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # Data collator דואג לרפד (Padding) את כל המשפטים באותו ה-batch לאותו אורך
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 7. יצירת ה-Trainer והפעלת האימון
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n🚀 Starting Fine-Tuning...")
    trainer.train()

    print("\n✅ Fine-Tuning Complete! Evaluating best model...")
    eval_results = trainer.evaluate()
    print(f"Final Validation Accuracy: {eval_results['eval_accuracy'] * 100:.2f}%")

    # 8. שמירת המודל המאומן
    final_model_dir = data_dir / "best_text_model"
    trainer.save_model(str(final_model_dir))
    print(f"Best model saved to: {final_model_dir}")


if __name__ == "__main__":
    main()