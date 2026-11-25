from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "Dataset" / "ebay_data_clean.csv"
CLEAN_PATH = BASE_DIR / "Dataset" / "cleaned_ebay_data.csv"
df = pd.read_csv(RAW_PATH)
print("Original Data shape:", df.shape)
df = df[df["Condition"] != "New"]
df = df[df["Condition"] != "For parts or not working"]
print("After filtering conditions:", df.shape)
df = df[df["Ram Size (GB)"] >= 1]
df = df.drop(columns=["Item Number"])
for column in df.select_dtypes(include=["object", "str"]).columns:
    df[column] = df[column].str.strip()
print("Final dataset shape:", df.shape)
df.to_csv(CLEAN_PATH, index=False)
print("Dataset cleaned successfully ->", CLEAN_PATH)