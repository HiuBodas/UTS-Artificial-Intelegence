from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import fitur

MODEL_PATH = Path(__file__).parent / "models" / "penguins.joblib"

LABEL = {
    "island": "Pulau",
    "sex": "Jenis Kelamin",
    "bill_length_mm": "Panjang Paruh (mm)",
    "bill_depth_mm": "Kedalaman Paruh (mm)",
    "flipper_length_mm": "Panjang Sirip (mm)",
    "body_mass_g": "Massa Tubuh (gram)",
}
SEX_LABEL = {"female": "Betina", "male": "Jantan", "unknown": "Tidak diketahui"}


@st.cache_resource
def load_bundle():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


st.set_page_config(page_title="Prediksi Spesies Penguin", page_icon="🐧", layout="wide")
st.title("🐧 Prediksi Spesies Penguin")
st.caption("Klasifikasi Spesies Penguin (Adelie, Chinstrap, Gentoo)")

b = load_bundle()
if b is None:
    st.error("Model belum tersedia. Jalankan `python latih.py` terlebih dahulu.")
    st.stop()

tab_pred, tab_eval = st.tabs(["Prediksi", "Evaluasi Model"])

# 1. Tab Prediksi
with tab_pred:
    col1, col2 = st.columns(2)
    with col1:
        island = st.selectbox(LABEL["island"], b["opsi"]["island"])
        sex = st.selectbox(LABEL["sex"], b["opsi"]["sex"], format_func=lambda s: SEX_LABEL.get(s, s))

    nilai = {}
    with col2:
        for kolom in fitur.FITUR_NUM_MENTAH:
            lo, hi, med = b["rentang"][kolom]
            nilai[kolom] = st.number_input(
                LABEL[kolom], min_value=0.0, value=float(round(med, 1)), step=0.1, key=kolom
            )

    if st.button("Prediksi", type="primary"):
        baris = pd.DataFrame([{"island": island, "sex": sex, **nilai}])[b["fitur_mentah"]]
        proba = b["pipeline"].predict_proba(baris)[0]
        hasil = pd.Series(proba, index=b["kelas"], name="Probabilitas")
        spesies = hasil.idxmax()

        st.success(f"Hasil Prediksi: **{spesies}** ({hasil.max():.1%})")

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Probabilitas per Spesies")
            st.dataframe(hasil.map("{:.2%}".format).to_frame())
        with c2:
            st.bar_chart(hasil)

        with st.expander("Fitur Rasio"):
            turunan = fitur.tambah_fitur(baris)[fitur.FITUR_TURUNAN]
            st.dataframe(turunan.round(3))

# 2. Tab Evaluasi Model
with tab_eval:
    m = b["metrik_test"]
    st.subheader(f"Model: {b['model_terpilih']}")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Accuracy", f"{m['accuracy']:.3f}")
    k2.metric("Balanced Accuracy", f"{m['balanced_accuracy']:.3f}")
    k3.metric("F1-Macro", f"{m['f1_macro']:.3f}")
    k4.metric("ROC-AUC", f"{m['roc_auc_ovr_macro']:.3f}")

    st.subheader("Confusion Matrix")
    cm = pd.DataFrame(m["confusion_matrix"], index=[f"Aktual {k}" for k in b["kelas"]],
                      columns=[f"Prediksi {k}" for k in b["kelas"]])
    st.dataframe(cm)

    st.subheader("Laporan Klasifikasi")
    lap = pd.DataFrame(m["laporan"]).T.loc[b["kelas"] + ["macro avg", "weighted avg"]]
    st.dataframe(lap[["precision", "recall", "f1-score", "support"]].round(3))

    st.subheader("Perbandingan Model (Cross-Validation)")
    cvdf = pd.DataFrame(b["hasil_cv"]).T[["f1_macro", "f1_macro_std", "accuracy", "balanced_accuracy"]]
    st.dataframe(cvdf.round(4))
