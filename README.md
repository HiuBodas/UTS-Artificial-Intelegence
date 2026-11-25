# Klasifikasi Spesies Penguin (Palmer Penguins)

## Deskripsi
Proyek ini mengklasifikasikan 3 spesies penguin (*Adelie*, *Chinstrap*, *Gentoo*) berdasarkan ukuran morfologi tubuh (panjang paruh, kedalaman paruh, panjang sirip, massa tubuh) serta pulau pengamatan.

Pipeline mencakup data cleaning, feature engineering (rasio paruh dan rasio tubuh), preprocessing (imputasi, scaling, dan one-hot encoding), evaluasi model (Logistic Regression, Random Forest, KNN) dengan stratified 5-fold cross-validation, serta antarmuka web interaktif menggunakan Streamlit.

## Struktur Direktori
```
├── dataset/
│   └── kasus03_penguins.csv
├── models/
│   └── penguins.joblib
├── notebook/
│   └── kasus03_penguins_colab.ipynb
├── ablasi.py
├── app.py
├── fitur.py
├── latih.py
├── requirements.txt
└── README.md
```

## Cara Menjalankan

### 1. Instalasi Dependensi
```bash
pip install -r requirements.txt
```

### 2. Pelatihan & Evaluasi Model
```bash
python latih.py
```

### 3. Menjalankan Aplikasi Streamlit
```bash
streamlit run app.py
```
