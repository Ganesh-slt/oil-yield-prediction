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

# ---------------- LOGO FUNCTION ----------------

BASE_DIR = os.path.dirname(__file__)

def get_base64(img_path):
    full_path = os.path.join(BASE_DIR, img_path)
    with open(full_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo1 = get_base64("logo1.png")
logo2 = get_base64("logo2.png")
logo3 = get_base64("logo3.png")
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


# ---------------- PROJECT INFO SECTION ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-box">
    <b>Funding Agency</b><br><br>
    Rajiv Gandhi Science and Technology Commission (RGSTC), Govt. of Maharashtra<br>
    Dr. Babasaheb Ambedkar Technological University (DBATU), Lonere
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-box">
    <b>Principal Investigator</b><br><br>
    Dr. B. M. Patil<br>
    Co-PI: Prof. G. N. Deshpande
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="info-box">
    <b>Institute</b><br><br>
    Shreeyash College of Engineering and Technology<br>
    Chhatrapati Sambhajinagar<br><br>
    
    """, unsafe_allow_html=True)

st.markdown("---")

# ---------------- LOAD MODEL ----------------
BASE_DIR = os.path.dirname(__file__)

model_path = os.path.join(BASE_DIR, "oil_yield_project____", "model", "model.pkl")
data_path = os.path.join(BASE_DIR, "oil_yield_project____", "data", "data.csv")

model = joblib.load(model_path)

# ---------------- LOAD DATA FOR DROPDOWN ----------------S
df = pd.read_csv(data_path)

season_list = sorted(df["season"].dropna().unique())
design_list = sorted(df["design"].dropna().unique())
plant_type_list = sorted(df["plant_type"].dropna().unique())
part_list = sorted(df["part"].dropna().unique())
condition_list = sorted(df["condition"].dropna().unique())

# ---------------- SIDEBAR INPUTS ----------------
st.sidebar.header("Input Parameters")

season = st.sidebar.selectbox("Season", season_list)
design = st.sidebar.selectbox("Design", design_list)
plant_type = st.sidebar.selectbox("Plant Type", plant_type_list)
part = st.sidebar.selectbox("Plant Part", part_list)
condition = st.sidebar.selectbox("Condition", condition_list)

batch_size = st.sidebar.number_input(
    "Batch Size (kg)", min_value=0.1, value=10.0
)

room_temp = st.sidebar.number_input(
    "Room Temperature (°C)", min_value=0.0, value=25.0
)

st.markdown("""
<hr style="margin-top:30px; margin-bottom:20px; border: 0.5px solid #E0E0E0;">

<div style="
text-align:center;
padding:12px;
color:#444;
font-size:15px;
">

<b>System Designed & Developed By</b><br>
Ganesh Ghare<br>
B.Tech – Computer Science & Engineering (Data Science) | 3rd Year

</div>
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

    st.markdown(f"""
    <div class="result-card">
        <h3>Prediction Results</h3>
        <p><b>Yield (ml/kg):</b> {yield_val}</p>
        <p><b>Volume (ml):</b> {volume_val}</p>
    </div>
    """, unsafe_allow_html=True)


