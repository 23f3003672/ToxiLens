import os 
import urllib.request 
from typing import Dict, List 
import pandas as pd 

# the 12 official Tox21 toxicity endpoints
TOX21_ASSAYS: List[str] = [
    "NR-AR",
    "NR-AR-LBD",
    "NR-AhR",
    "NR-Aromatase",
    "NR-ER",
    "NR-ER-LBD",
    "NR-PPAR-gamma",
    "SR-ARE",
    "SR-ATAD5",
    "SR-HSE",
    "SR-MMP",
    "SR-p53",
]

TOX21_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"

def load_tox21(raw_data_dir: str = "data/raw") -> pd.DataFrame:
    os.makedirs(raw_data_dir, exist_ok=True)
    gz_path = os.path.join(raw_data_dir, "tox21.csv.gz")

    if not os.path.exists(gz_path):
        print(f"Downloading Tox21 dataset to {gz_path}...")
        urllib.request.urlretrieve(TOX21_URL, gz_path)
        print("Downlaod complate")

    df = pd.read_csv(gz_path)
    return df 

def get_assay_summary(df: pd.DataFrame) -> pd.DataFrame:

    total_rows = len(df)
    summary_records = []

    for assay in TOX21_ASSAYS:
        if assay not in df.columns:
            continue 
        series = df[assay]
        missing_count = int(series.isna().sum())
        labeled_series = series.dropna()
        labeled_count = len(labeled_series)

        positives = int((labeled_series == 1.0).sum())
        negatives = int((labeled_series == 0.0).sum())

        pos_ratio = (positives/labeled_count * 100.0) if labeled_count > 0 else 0.0

        summary_records.append({
            "Assay": assay,
            "Total": total_rows,
            "Labeled": labeled_count,
            "Missing": missing_count,
            "Missing %": round((missing_count/ total_rows)*100.0, 1),
            "Toxic (1)": positives,
            "Non-toxic(0)": negatives,
            "Toxic %": round(pos_ratio,2),
        })
    return pd.DataFrame(summary_records)