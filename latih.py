from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import fitur as F

RANDOM_STATE = 42
BASE = Path(__file__).parent
DATA_PATH = BASE / "dataset" / "kasus03_penguins.csv"
MODEL_PATH = BASE / "models" / "penguins.joblib"


def buat_preprocess() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), F.FITUR_NUM),
        ("kat", Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=F.KATEGORI_SEX_HILANG)),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), F.FITUR_KAT),
    ])


def buat_pipeline(clf) -> Pipeline:
    return Pipeline([
        ("fitur", FunctionTransformer(F.tambah_fitur, validate=False)),
        ("prep", buat_preprocess()),
        ("clf", clf),
    ])


def main() -> None:
    # 1. Load & audit data
    df_mentah = pd.read_csv(DATA_PATH)
    print("=== Audit Data Mentah ===")
    for k, v in F.audit_data(df_mentah).items():
        print(f"  {k}: {v}")

    df = F.bersihkan(df_mentah)
    print(f"\nSetelah pembersihan: {df_mentah.shape} -> {df.shape}")

    X = df[F.FITUR_MENTAH].copy()
    y = df[F.TARGET].astype(str)

    # 2. Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Data Train: {len(X_train)} | Data Test: {len(X_test)}")

    # 3. Bandingkan model dengan cross-validation
    kandidat = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    hasil_cv = {}

    print("\n=== Cross-Validation (5-Fold) ===")
    for nama, clf in kandidat.items():
        s = cross_validate(buat_pipeline(clf), X_train, y_train, cv=cv,
                           scoring=["accuracy", "f1_macro", "balanced_accuracy"])
        hasil_cv[nama] = {
            "accuracy": float(s["test_accuracy"].mean()),
            "f1_macro": float(s["test_f1_macro"].mean()),
            "f1_macro_std": float(s["test_f1_macro"].std()),
            "balanced_accuracy": float(s["test_balanced_accuracy"].mean()),
        }
        h = hasil_cv[nama]
        print(f"  {nama:<20} Accuracy: {h['accuracy']:.4f} | F1-macro: {h['f1_macro']:.4f} (+/-{h['f1_macro_std']:.4f})")

    terbaik = max(hasil_cv, key=lambda n: hasil_cv[n]["f1_macro"])
    print(f"\nModel terbaik: {terbaik}")

    # 4. Latih model terbaik dan evaluasi pada data test
    model = buat_pipeline(kandidat[terbaik]).fit(X_train, y_train)
    kelas = list(model.classes_)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    cm = confusion_matrix(y_test, y_pred, labels=kelas)
    metrik = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "roc_auc_ovr_macro": float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro", labels=kelas)),
        "confusion_matrix": cm.tolist(),
        "laporan": classification_report(y_test, y_pred, labels=kelas, output_dict=True),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    print("\n=== Evaluasi Data Test ===")
    print(classification_report(y_test, y_pred, labels=kelas, digits=3))
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=kelas, columns=kelas))

    # 5. Simpan bundle model
    bundle = {
        "pipeline": model,
        "model_terpilih": terbaik,
        "fitur_mentah": F.FITUR_MENTAH,
        "kelas": kelas,
        "opsi": {
            "island": sorted(X_train["island"].dropna().unique().tolist()),
            "sex": sorted(X_train["sex"].dropna().unique().tolist()) + [F.KATEGORI_SEX_HILANG],
        },
        "rentang": {c: [float(X_train[c].min()), float(X_train[c].max()),
                        float(X_train[c].median())] for c in F.FITUR_NUM_MENTAH},
        "metrik_test": metrik,
        "hasil_cv": hasil_cv,
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"\nModel tersimpan di: {MODEL_PATH}")


if __name__ == "__main__":
    main()
