import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_parquet("data/processed/cic2017_clean.parquet").reset_index(drop=True)

# Position of each row within its own file, from 0.0 to 1.0
grp = data.groupby("source_file")
pos = grp.cumcount() / grp["Label"].transform("size")

train = data[pos < 0.7]
test = data[pos >= 0.7].copy()

parts = [g.sample(min(len(g), 50_000), random_state=42)
         for _, g in train.groupby("Label")]
train = pd.concat(parts, ignore_index=True)

print("Train labels:\n", train["Label"].value_counts())
print("\nTest labels:\n", test["Label"].value_counts())

drop = ["Label", "source_file"]
X_tr, y_tr = train.drop(columns=drop), (train["Label"] != "BENIGN").astype(int)
X_te, y_te = test.drop(columns=drop), (test["Label"] != "BENIGN").astype(int)

clf = RandomForestClassifier(n_estimators=50, n_jobs=4, random_state=42)
clf.fit(X_tr, y_tr)
test["pred"] = clf.predict(X_te)

print("\n=== Overall ===")
print(classification_report(y_te, test["pred"], digits=4))
print(confusion_matrix(y_te, test["pred"]))

print("\n=== Per label (BENIGN mean = false positive rate; attacks = recall) ===")
print(test.groupby("Label")["pred"].agg(["mean", "size"]).round(4).to_string())