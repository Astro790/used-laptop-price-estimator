from pathlib import Path
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "used_laptop_estimator.joblib"

model = joblib.load(MODEL_PATH)

sample_laptops = pd.DataFrame([
    {
        "Screen Size (in)": 11.6,
        "Ram Size (GB)": 4.0,
        "SSD Capacity (GB)": 32.0,
        "Processor Speed (GHz)": 1.6,
        "Hard Drive Capacity (GB)": 32,
        "Brand": "Acer",
        "Processor": "Intel Celeron",
        "GPU": "Intel HD Graphics",
        "Resolution Size": "Small",
        "OS": "Chrome OS",
        "Storage Type": "eMMC",
        "Condition": "Used",
    },
    {
        "Screen Size (in)": 14.0,
        "Ram Size (GB)": 16.0,
        "SSD Capacity (GB)": 512.0,
        "Processor Speed (GHz)": 2.6,
        "Hard Drive Capacity (GB)": 512,
        "Brand": "Dell",
        "Processor": "Intel Core i5",
        "GPU": "Intel UHD Graphics 620",
        "Resolution Size": "Medium",
        "OS": "Windows 11 Pro",
        "Storage Type": "SSD",
        "Condition": "Very Good",
    },
    {
        "Screen Size (in)": 15.6,
        "Ram Size (GB)": 32.0,
        "SSD Capacity (GB)": 1000.0,
        "Processor Speed (GHz)": 3.8,
        "Hard Drive Capacity (GB)": 1000,
        "Brand": "HP",
        "Processor": "Intel Core i7",
        "GPU": "NVIDIA Quadro P3200",
        "Resolution Size": "Medium",
        "OS": "Windows 11 Pro",
        "Storage Type": "SSD",
        "Condition": "Excellent",
    },
])

predictions = model.predict(sample_laptops)

for i, price in enumerate(predictions):
    row = sample_laptops.iloc[i]
    print(f"{row['Brand']} | {row['Processor']} | {int(row['Ram Size (GB)'])}GB RAM | {row['Condition']} -> Estimated Price: ${price:.2f}")