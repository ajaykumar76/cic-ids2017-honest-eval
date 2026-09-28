import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

data = pd.read_parquet("data/processed/cic2017_clean.parquet").reset_index(drop=True)
data["day"] = data["source_file"].str.split("-").str[0]
grp = data.groupby("source_file")
pos = grp.cumcount() / grp["Label"].transform("size")

splits = {
    "Mon-Wed -> Thu-Fri (unseen attacks)": (
        data["day"].isin(["Monday", "Tuesday", "Wednesday"]),
        data["day"].isin(["Thursday", "Friday"]),
    ),
    "Ordered 70/30 within files": (pos < 0.7, pos >= 0.7),
}

def cap(df, n=50_000):
    parts = [g.sample(min(len(g), n), random_state=42) for _, g in df.groupby("Label")]
    return pd.concat(parts, ignore_index=True)

drop = ["Label", "source_file", "day"]
rows = []

for split_name, (tr_mask, te_mask) in splits.items():
    train = cap(data[tr_mask])
    test = data[te_mask]
    X_tr, y_tr = train.drop(columns=drop), (train["Label"] != "BENIGN").astype(int)
    X_te, y_te = test.drop(columns=drop), (test["Label"] != "BENIGN").astype(int)
    spw = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

    models = {
        "RandomForest": RandomForestClassifier(n_estimators=50, n_jobs=4, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist",
                                 n_jobs=4, random_state=42),
        "XGBoost (weighted)": XGBClassifier(n_estimators=200, max_depth=6, tree_method="hist",
                                            n_jobs=4, random_state=42, scale_pos_weight=spw),
    }
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
        rows.append({
            "split": split_name, "model": name,
            "precision": round(precision_score(y_te, pred, zero_division=0), 4),
            "recall": round(recall_score(y_te, pred), 4),
            "f1": round(f1_score(y_te, pred), 4),
            "FPR": round(fp / (fp + tn), 5),
        })
        print(rows[-1])

result = pd.DataFrame(rows)
print("\n", result.to_string(index=False))
result.to_csv("reports/06_results.csv", index=False)