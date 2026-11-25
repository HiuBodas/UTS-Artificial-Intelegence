import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
import fitur as F

df = F.bersihkan(pd.read_csv("dataset/kasus03_penguins.csv"))
y = df["species"].astype(str)
X = df[F.FITUR_MENTAH + ["year"]]
Xtr, _, ytr, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)


def skor(num, kat, rasio=True):
    pre = ColumnTransformer([
        ("num", Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler())]), num),
        ("kat", Pipeline([("i", SimpleImputer(strategy="constant", fill_value="unknown")),
                          ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), kat),
    ])
    langkah = ([("f", FunctionTransformer(F.tambah_fitur, validate=False))] if rasio else [])
    langkah += [("p", pre), ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    s = cross_validate(Pipeline(langkah), Xtr, ytr, cv=cv, scoring="f1_macro")
    return round(float(s["test_score"].mean()), 4)


print("Semua fitur (baseline) :", skor(F.FITUR_NUM, F.FITUR_KAT))
print("+ fitur year           :", skor(F.FITUR_NUM + ["year"], F.FITUR_KAT))
print("Tanpa fitur island     :", skor(F.FITUR_NUM, ["sex"]))
print("Tanpa fitur rasio      :", skor(F.FITUR_NUM_MENTAH, F.FITUR_KAT, rasio=False))
