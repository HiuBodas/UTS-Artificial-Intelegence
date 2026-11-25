import numpy as np
import pandas as pd

TARGET = "species"

FITUR_NUM_MENTAH = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
FITUR_KAT_MENTAH = ["island", "sex"]
FITUR_MENTAH = FITUR_KAT_MENTAH + FITUR_NUM_MENTAH

FITUR_TURUNAN = ["bill_ratio", "flipper_mass_ratio"]
FITUR_NUM = FITUR_NUM_MENTAH + FITUR_TURUNAN
FITUR_KAT = FITUR_KAT_MENTAH

KOLOM_DIBUANG = ["year"]
KATEGORI_SEX_HILANG = "unknown"


def audit_data(df: pd.DataFrame) -> dict:
    num = df[FITUR_NUM_MENTAH]
    return {
        "bentuk": df.shape,
        "duplikat_baris_penuh": int(df.duplicated().sum()),
        "missing_per_kolom": df.isna().sum()[lambda s: s > 0].to_dict(),
        "baris_semua_ukuran_kosong": int(num.isna().all(axis=1).sum()),
        "nilai_tidak_positif": int((num <= 0).sum().sum()),
        "tipe_data": df.dtypes.astype(str).to_dict(),
        "kategori_sex": df["sex"].value_counts(dropna=False).to_dict(),
        "distribusi_kelas": df[TARGET].value_counts().to_dict(),
    }


def bersihkan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["island", "sex", TARGET]:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
            df[c] = df[c].str.strip()
    df["sex"] = df["sex"].str.lower()
    df = df.drop_duplicates()
    df[FITUR_NUM_MENTAH] = df[FITUR_NUM_MENTAH].where(df[FITUR_NUM_MENTAH] > 0)
    df = df[~df[FITUR_NUM_MENTAH].isna().all(axis=1)]
    return df.reset_index(drop=True)


def tambah_fitur(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    depth = X["bill_depth_mm"].replace(0, np.nan)
    mass = X["body_mass_g"].replace(0, np.nan)
    X["bill_ratio"] = X["bill_length_mm"] / depth
    X["flipper_mass_ratio"] = X["flipper_length_mm"] / mass * 1000
    return X
