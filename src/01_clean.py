import pandas as pd, numpy as np
from pathlib import Path

log = []
def note(msg):
    print(msg)
    log.append(msg)

# Load, one file at a time, with smaller dtypes to save memory
dfs = []
for f in sorted(Path("data/raw").glob("*.csv")):
    df = pd.read_csv(f, encoding="latin1", low_memory=False)
    df.columns = df.columns.str.strip()
    df["source_file"] = f.name
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    note(f"{f.name}: {df.shape}")
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
del dfs
note(f"\nTotal rows loaded: {len(data)}")

# Clean, logging how many rows each step removes
n = len(data)
data = data.replace([np.inf, -np.inf], np.nan).dropna()
note(f"Removed inf/NaN rows: {n - len(data)}")

n = len(data)
feature_cols = [c for c in data.columns if c != "source_file"]
data = data.drop_duplicates(subset=feature_cols)
note(f"Removed duplicate rows: {n - len(data)}")

const_cols = [c for c in data.columns if data[c].nunique() <= 1]
data = data.drop(columns=const_cols)
note(f"Dropped constant columns: {const_cols}")

note(f"\nFinal shape: {data.shape}")
note("\nLabel counts:\n" + data["Label"].value_counts().to_string())

data.to_parquet("data/processed/cic2017_clean.parquet")
Path("reports/cleaning_log.txt").write_text("\n".join(log))