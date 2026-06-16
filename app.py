import streamlit as st
import pandas as pd
import joblib

# ==========================
# KONFIGURASI HALAMAN
# ==========================

st.set_page_config(
    page_title="Prediksi Harga Rumah",
    page_icon="🏠",
    layout="wide"
)

# ==========================
# LOAD MODEL & ENCODER
# ==========================

lokasi_encoder = joblib.load("lokasi_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Model klasifikasi
MODEL_PATHS = {
    "🌳 Decision Tree (80,01%)": "decision_tree_model.pkl",
    "👥 K-Nearest Neighbors (78,12%)": "knn_model.pkl",
    "🧠 Neural Network (68,01%)": "nn_model.pkl",
    "📈 Support Vector Machine (65,71%)": "svm_model.pkl"
}

# Model regresi
REGRESSION_MODELS = {
    "👥 KNN Regression": "knn_regresi_model.pkl",
    "🌳 Decision Tree Regression": "decision_tree_regresi_model.pkl",
    "📈 Support Vector Regression": "svr_regresi_model.pkl",
    "🧠 Neural Network Regression": "nn_regresi_model.pkl"
}
scaler_regresi = joblib.load("scaler_regresi.pkl")

# ==========================
# CUSTOM CSS
# ==========================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #FFF9F5 0%, #FFF4EF 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #EED8D5 0%, #FFF9F5 100%);
    border-right: 1px solid rgba(203, 97, 111, 0.15);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.header-card {
    position: relative;
    overflow: hidden;
    background: rgba(255,255,255,0.85);
    border-radius: 30px;
    padding: 40px;
    border: 1px solid rgba(203,97,111,0.15);
    box-shadow: 0 12px 30px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}

.header-card::before {
    content: "";
    position: absolute;
    width: 220px;
    height: 220px;
    background: rgba(234,204,131,0.25);
    border-radius: 50%;
    top: -90px;
    right: -70px;
}

.header-card::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    background: rgba(203,97,111,0.12);
    border-radius: 50%;
    bottom: -80px;
    left: -50px;
}

.title {
    color: #4A3F35;
    font-size: 42px;
    font-weight: 700;
}

.subtitle {
    color: #7A6A63;
    font-size: 18px;
    margin-top: 8px;
}

.metric-card {
    background: rgba(255,255,255,0.85);
    border-radius: 24px;
    padding: 20px;
    text-align: center;
    border: 1px solid rgba(203,97,111,0.15);
    box-shadow: 0 8px 20px rgba(0,0,0,0.04);
}

.metric-title {
    color: #7A6A63;
    font-size: 14px;
}

.metric-value {
    color: #4A3F35;
    font-size: 22px;
    font-weight: 600;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #EACC83, #CB616F);
    color: white;
    border: none;
    border-radius: 18px;
    padding: 14px;
    font-size: 18px;
    font-weight: 600;
}

.stButton > button:hover {
    opacity: 0.95;
}

[data-testid="stAlert"] {
    border-radius: 18px;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# SIDEBAR
# ==========================

with st.sidebar:

    st.markdown("## 🏠 Prediksi Harga Rumah")

    mode = st.radio(
        "Pilih Jenis Prediksi",
        ["Klasifikasi", "Regresi"]
    )

    st.markdown("---")

    st.markdown("📊 Dashboard")
    st.markdown("🧠 Model")
    st.markdown("🏡 Properti")
    st.markdown("⚙️ Pengaturan")

    st.markdown("---")

    st.caption("Dasar Ilmu Data (GIK2GAB3)")
    st.caption("Kelompok 4")

# ==========================
# HEADER
# ==========================

subtitle = (
    "Prediksi kategori harga rumah berdasarkan karakteristik properti menggunakan machine learning."
    if mode == "Klasifikasi"
    else
    "Estimasi harga rumah dalam rupiah berdasarkan karakteristik properti menggunakan machine learning."
)

st.markdown(f"""
<div class="header-card">
    <div class="title">🏠 Aplikasi Prediksi Harga Rumah di Depok 2026</div>
    <div class="subtitle">
        {subtitle}
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================
# INFO CARD
# ==========================

col1, col2, col3, col4 = st.columns(4)

if mode == "Klasifikasi":
    cards = [
        ("🧠", "Model Terbaik", "Decision Tree"),
        ("🎯", "Akurasi", "80,01%"),
        ("📍", "Fitur", "6 Variabel"),
        ("🏡", "Kategori", "3 Kelas")
    ]
else:
    cards = [
        ("📈", "Model", "4 Algoritma"),
        ("📉", "RMSE Terbaik", "16,88 M"),
        ("📍", "Fitur", "6 Variabel"),
        ("💰", "Output", "Harga (Rp)")
    ]

for col, (icon, title, value) in zip([col1, col2, col3, col4], cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:28px;">{icon}</div>
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

# ==========================
# PILIH MODEL
# ==========================

if mode == "Klasifikasi":

    st.subheader("🧠 Pilih Metode Prediksi")

    selected_model = st.selectbox(
        "Pilih algoritma machine learning",
        list(MODEL_PATHS.keys())
    )

    model = joblib.load(MODEL_PATHS[selected_model])

    st.info(f"Model aktif: {selected_model}")

else:

    st.subheader("📈 Pilih Metode Prediksi")

    selected_model = st.selectbox(
        "Pilih algoritma regresi",
        list(REGRESSION_MODELS.keys())
    )

    model = joblib.load(REGRESSION_MODELS[selected_model])

    st.info(f"Model aktif: {selected_model}")

# ==========================
# INPUT DATA
# ==========================

st.subheader("📝 Informasi Properti")

col1, col2 = st.columns(2)

with col1:
    kamar_tidur = st.number_input(
        "🛏️ Jumlah Kamar Tidur",
        min_value=1,
        value=3
    )

    garasi = st.number_input(
        "🚗 Jumlah Garasi",
        min_value=0,
        value=1
    )

    luas_bangunan = st.number_input(
        "🏢 Luas Bangunan (m²)",
        min_value=1,
        value=90
    )

with col2:
    kamar_mandi = st.number_input(
        "🛁 Jumlah Kamar Mandi",
        min_value=1,
        value=2
    )

    luas_tanah = st.number_input(
        "🌳 Luas Tanah (m²)",
        min_value=1,
        value=120
    )

    lokasi = st.selectbox(
        "📍 Lokasi",
        lokasi_encoder.classes_
    )

st.write("")

button_text = (
    "🔮 Prediksi Kategori Harga"
    if mode == "Klasifikasi"
    else
    "💰 Prediksi Harga Rumah"
)

prediksi = st.button(button_text)

# ==========================
# PROSES PREDIKSI
# ==========================

if prediksi:

    lokasi_encoded = lokasi_encoder.transform([lokasi])[0]

    input_data = pd.DataFrame({
        "Kamar Tidur": [kamar_tidur],
        "Kamar Mandi": [kamar_mandi],
        "Garasi": [garasi],
        "Luas Tanah": [luas_tanah],
        "Luas Bangunan": [luas_bangunan],
        "Lokasi": [lokasi_encoded]
    })

    if mode == "Klasifikasi":

        if "Decision Tree" not in selected_model:
            input_prediksi = scaler.transform(input_data)
        else:
            input_prediksi = input_data

        hasil = model.predict(input_prediksi)[0]

        kategori = {
            0: "🟢 Murah",
            1: "🟡 Sedang",
            2: "🔴 Mahal"
        }

        st.success(
            f"Kategori Harga Rumah: **{kategori[hasil]}**"
        )

    else:

        # Decision Tree Regression tidak memerlukan scaling
        if "Decision Tree" in selected_model:
            input_prediksi = input_data
        else:
            input_prediksi = scaler_regresi.transform(input_data)

        hasil = model.predict(input_prediksi)[0]

        st.success(
            f"💰 Estimasi Harga Rumah: **Rp {hasil:,.0f}**"
        )
