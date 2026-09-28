import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = pd.read_parquet("data/processed/cic2017_clean.parquet")

# Fix garbled web attack labels, then save the fixed version
data["Label"] = data["Label"].str.replace(r"^Web Attack\s+\S+\s+", "Web Attack - ", regex=True)
data.to_parquet("data/processed/cic2017_clean.parquet")

# Q1: flows per label
counts = data["Label"].value_counts()
print("\n=== Q1: Label counts ===\n", counts)
counts.plot(kind="barh", logx=True, figsize=(8, 6), title="Flows per label (log scale)")
plt.tight_layout()
plt.savefig("reports/label_counts.png", dpi=150)
plt.close()

# Q2: which attacks appear on which day/file
print("\n=== Q2: Labels per file ===")
print(pd.crosstab(data["Label"], data["source_file"]).to_string())

# Q3: suspicious negative values
num = data.select_dtypes("number")
neg = (num < 0).sum()
print("\n=== Q3: Columns with negative values ===\n", neg[neg > 0])

# Q4: highly correlated features (on a sample, for speed)
corr = num.sample(500_000, random_state=42).corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high = [c for c in upper.columns if (upper[c] > 0.95).any()]
print(f"\n=== Q4: {len(high)} features correlated > 0.95 with another ===")
print(high)