import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data = pd.read_parquet("data/processed/cic2017_clean.parquet")

# Cap each class at 50k rows so training is quick on your laptop
parts = [g.sample(min(len(g), 50_000), random_state=42)
         for _, g in data.groupby("Label")]
data = pd.concat(parts, ignore_index=True)

print("Sample size:", len(data))

y = (data["Label"] != "BENIGN").astype(int)
X = data.drop(columns=["Label", "source_file"])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=50, n_jobs=4, random_state=42)
clf.fit(X_tr, y_tr)
pred = clf.predict(X_te)

print(classification_report(y_te, pred, digits=4))
print(confusion_matrix(y_te, pred))