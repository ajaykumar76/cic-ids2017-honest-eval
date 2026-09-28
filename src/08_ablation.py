import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

data = pd.read_parquet("data/processed/cic2017_clean.parquet").reset_index(drop=True)
data["day"] = data["source_file"].str.split("-").str[0]
grp = data.groupby("source_file")
pos = grp.cumcount() / grp["Label"].transform("size")

splits = {
    "Mon-Wed -> Thu-Fri": (data["day"].isin(["Monday", "Tuesday", "Wednesday"]),
                           data["day"].isin(["Thursday", "Friday"])),
    "Ordered 70/30": (pos < 0.7, pos >= 0.7),
}
variants = {
    "all features": [],
    "no Destination Port": ["Destination Port"],
    "no port/window/min_seg": ["Destination Port", "Init_Win_bytes_forward",
                               "Init_Win_bytes_backward", "min_seg_size_forward"],
}

def cap(df, n=50_000):
    parts = [g.sample(min(len(g), n), random_state=42) for _, g in df.groupby("Label")]
    return pd.concat(parts, ignore_index=True)

rows = []
for sname, (trm, tem) in splits.items():
    train, test = cap(data[trm]), data[tem]
    for vname, extra in variants.items():
        drop = ["Label", "source_file", "day"] + extra
        X_tr, y_tr = train.drop(columns=drop), (train["Label"] != "BENIGN").astype(int)
        X_te, y_te = test.drop(columns=drop), (test["Label"] != "BENIGN").astype(int)
        m = XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist",
                          n_jobs=4, random_state=42).fit(X_tr, y_tr)
        p = m.predict(X_te)
        tn, fp, fn, tp = confusion_matrix(y_te, p, labels=[0, 1]).ravel()
        rows.append({"split": sname, "features": vname,
                     "precision": round(precision_score(y_te, p), 4),
                     "recall": round(recall_score(y_te, p), 4),
                     "f1": round(f1_score(y_te, p), 4),
                     "FPR": round(fp / (fp + tn), 5)})
        print(rows[-1])

out = pd.DataFrame(rows)
print("\n", out.to_string(index=False))
out.to_csv("reports/08_ablation.csv", index=False)