import os
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# טעינת משתני הסביבה מתוך קובץ ה-.env
load_dotenv()


# הגדרת פונקציית התיוג
def assign_label(problem_str):
    if pd.isna(problem_str):
        return 3
    if problem_str == 'normal':
        return 0
    if 'Cardiomegaly' in problem_str:
        return 1
    if 'Opacity' in problem_str:
        return 2
    return 3


def main():
    # 1. משיכת הנתיב מקובץ ה-.env
    data_dir_str = os.getenv("DATA_PATH")
    if not data_dir_str:
        raise ValueError("CRITICAL ERROR: DATA_DIR is not defined in the .env file.")

    # Path מתרגם אוטומטית את הנתיב למבנה הנכון (מק או ווינדוס)
    data_dir = Path(data_dir_str)

    # 2. הגדרת נתיבי הקבצים
    input_file = data_dir / "indiana_frontal.csv"
    output_file = data_dir / "clean_frontal_dataset.csv"

    print(f"Loading data from:\n{input_file}")

    # בדיקה שהקובץ אכן קיים לפני שמנסים לקרוא אותו
    if not input_file.exists():
        raise FileNotFoundError(f"Could not find the file at {input_file}")

    df = pd.read_csv(input_file)

    print("Assigning labels and cleaning data...")
    df['Target_Label'] = df['Problems'].apply(assign_label)
    df['indication'] = df['indication'].fillna('No indication provided')

    columns_to_keep = ['filename', 'uid', 'indication', 'Target_Label', 'img_path']
    clean_df = df[columns_to_keep]

    # 3. שמירת הקובץ החדש באותה תיקיית נתונים
    clean_df.to_csv(output_file, index=False)

    print(f"\nSuccess! Clean dataset saved to:\n{output_file}")
    print("\nLabel Distribution:")
    print(clean_df['Target_Label'].value_counts())


if __name__ == "__main__":
    main()