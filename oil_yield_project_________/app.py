import streamlit as st
import pandas as pd
import joblib
import base64
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Oil Yield Predictor",
    layout="wide"
)

# ---------------- BASE DIRECTORY ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------- LOGO FUNCTION ----------------
def get_base64(img_path):
    full_path = os.path.join(BASE_DIR, img_path)
    if not os.path.exists(full_path):
        return ""
    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo1 = get_base64("logo1.png")
logo2 = get_base64("logo2.png")
logo3 = get_base64("logo3.png")

# ---------------- HEADER UI ----------------
st.markdown(f"""
<style>
.logo-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background-color: #f2f2f2;
    padding: 20px 60px;
}}

.logo-bar img {{
    height: 120px;
    object-fit: contain;
}}

.title-section {{
    text-align: center;
    padding: 20px 0px 10px 0px;
}}

.main-title {{
    font-size: 32px;
    font-weight: 700;
}}

.sub-title {{
    font-size: 18px;
    color: gray;
}}
</style>

<div class="logo-bar">
    <img src="data:image/png;base64,{logo1}">
    <img src="data:image/png;base64,{logo2}">
    <img src="data:image/png;base64,{logo3}">
</div>

<div class="title-section">
    <div class="main-title">
    🌿 Oil Yield & Volume Prediction System 🌿
    </div>
    <div class="sub-title">
    Design and Development of a High Yield Cost-effective Portable Oil Extractor
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- PROJECT INFO ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <b>Funding Agency</b><br><br>
    RGSTC, Govt. of Maharashtra<br>
    DBATU, Lonere
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <b>Principal Investigator</b><br><br>
    Dr. B. M. Patil<br>
    Prof. G. N. Deshpande
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <b>Institute</b><br><br>
    Shreeyash College of Engineering<br>
    Chhatrapati Sambhajinagar
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------- LOAD MODEL ----------------
model_path = os.path.join(BASE_DIR, "model", "model.pkl")
data_path = os.path.join(BASE_DIR, "data", "data.csv")

# Safety check
if not os.path.exists(model_path):
    st.error(f"Model file not found: {model_path}")
    st.stop()

if not os.path.exists(data_path):
    st.error(f"Data file not found: {data_path}")
    st.stop()

model = joblib.load(model_path)
df = pd.read_csv(data_path)

# ---------------- DROPDOWNS ----------------
season_list = sorted(df["season"].dropna().unique())
design_list = sorted(df["design"].dropna().unique())
plant_type_list = sorted(df["plant_type"].dropna().unique())
part_list = sorted(df["part"].dropna().unique())
condition_list = sorted(df["condition"].dropna().unique())

# ---------------- SIDEBAR ----------------
st.sidebar.header("Input Parameters")

season = st.sidebar.selectbox("Season", season_list)
design = st.sidebar.selectbox("Design", design_list)
plant_type = st.sidebar.selectbox("Plant Type", plant_type_list)
part = st.sidebar.selectbox("Plant Part", part_list)
condition = st.sidebar.selectbox("Condition", condition_list)

batch_size = st.sidebar.number_input("Batch Size (kg)", 0.1, 100.0, 10.0)
room_temp = st.sidebar.number_input("Room Temperature (°C)", 0.0, 100.0, 25.0)

# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>
<b>Developed By</b><br>
Ganesh Ghare<br>
B.Tech CSE (Data Science)
</center>
""", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
input_df = pd.DataFrame([{
    "season": season,
    "design": design,
    "batch_size": batch_size,
    "room_temp": room_temp,
    "plant_type": plant_type,
    "part": part,
    "condition": condition
}])

if st.button("Predict"):
    prediction = model.predict(input_df)

    yield_val = round(prediction[0][0], 2)
    volume_val = round(prediction[0][1], 2)

    st.success(f"Yield: {yield_val} ml/kg")
    st.success(f"Volume: {volume_val} ml")
