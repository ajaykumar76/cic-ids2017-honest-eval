import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

data = pd.read_parquet("data/processed/cic2017_clean.parquet").reset_index(drop=True)
grp = data.groupby("source_file")
pos = grp.cumcount() / grp["Label"].transform("size")

def cap(df, n=50_000):
    parts = [g.sample(min(len(g), n), random_state=42) for _, g in df.groupby("Label")]
    return pd.concat(parts, ignore_index=True)

train = cap(data[pos < 0.7])
test = data[pos >= 0.7].sample(3000, random_state=42)

drop = ["Label", "source_file"]
X_tr, y_tr = train.drop(columns=drop), (train["Label"] != "BENIGN").astype(int)
X_te = test.drop(columns=drop)

model = XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist",
                      n_jobs=4, random_state=42)
model.fit(X_tr, y_tr)

explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(X_te)

shap.summary_plot(sv, X_te, show=False, max_display=15)
plt.tight_layout()
plt.savefig("reports/shap_summary.png", dpi=150)
print("Saved reports/shap_summary.png")